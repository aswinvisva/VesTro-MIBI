import datetime
import random
from collections import Counter
import os

import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from feature_extraction.sift_feature_gen import SIFTFeatureGen
from models.clustering_helper import ClusteringHelper
from models.flowsom_clustering import ClusteringFlowSOM
from models.vessel_net import VesselNet, vis_tensor
from topic_generation.cnn_lda import CNNLDA
from utils.construct_microenvironments import construct_partitioned_microenvironments_from_contours, \
    construct_vessel_relative_area_microenvironments_from_contours
from utils.mibi_reader import read, get_all_point_data
from topic_generation.lda_topic_generation import LDATopicGen
from image_segmentation.sliding_window_segmentation import split_image
from image_segmentation.extract_cell_events import *
from image_segmentation.acwe_segmentation import *
from image_segmentation.sliding_window_segmentation import *
from feature_extraction import bag_of_cells_feature_gen
from feature_extraction.cnn_feature_gen import CNNFeatureGen
from feature_extraction.markers_feature_gen import *
from topic_generation.cnnfcluster import CNNFCluster
from topic_generation.lda_topic_generation import LDATopicGen
from topic_generation.lda import LDA, lda_learning
from utils.cnn_dataloader import VesselDataset, preprocess_input

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def vessel_region_plots(n_expansions, interval, markers_names, expansion_data):
    """
    Vessel region line plots

    :param n_expansions:
    :param interval:
    :param markers_names:
    :param expansion_data:
    :return:
    """
    marker_clusters = {
        "Nucleus": ["HH3"],
        "Microglia": ["CD45", "HLADR", "Iba1"],
        "Disease": ["CD47", "ABeta42", "polyubiK48", "PHFTau", "8OHGuanosine"],
        "Vessels": ["SMA", "CD31", "CollagenIV", "TrkA", "GLUT1", "Desmin", "vWF", "CD105"],
        "Astrocytes": ["S100b", "GlnSyn", "Cx30", "EAAT2", "CD44", "GFAP", "Cx43"],
        "Synapse": ["CD56", "Synaptophysin", "VAMP2", "PSD95"],
        "Oligodendrocytes": ["MOG", "MAG"],
        "Neurons": ["Calretinin", "Parvalbumin", "MAP2", "Gephyrin"]
    }

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    points = [1, 7, 26, 30, 43, 48]

    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    # Change in Marker Expression w.r.t pixel expansion per vessel (All bins)
    for point in points:
        output_dir = "results/point_%s_vessels_%s_interval_%s_expansions_allbins" % (
            str(point), str(interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        n_vessels = len(expansion_data[0][point - 1])

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():
                plt.plot([], [], color=colors[key], label=key)

            for marker, marker_name in enumerate(markers_names):
                y = []
                color = None

                for key in marker_clusters.keys():
                    if marker_name in marker_clusters[key]:
                        color = colors[key]
                        break

                for i in range(n_expansions):
                    vessel_data = expansion_data[i][point - 1][vessel]
                    y.append(vessel_data[marker])

                plt.plot(pixel_expansions, y, color=color)
            plt.xticks(pixel_expansions, fontsize=7, rotation=90)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Vessel ID: %s, Point %s All Bins" % (str(vessel), str(point)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/vessel_%s_allbins.png' % str(vessel), bbox_inches='tight')
            plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per vessel (Average bins)
    for point in points:
        output_dir = "results/point_%s_vessels_%s_interval_%s_expansions_averagebins" % (
            str(point), str(interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        n_vessels = len(expansion_data[0][point - 1])

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():
                plt.plot([], [], color=colors[key], label=key)

            for key in marker_clusters.keys():
                color = colors[key]
                y_tot = []
                for marker, marker_name in enumerate(markers_names):
                    if marker_name not in marker_clusters[key]:
                        continue

                    y = []

                    for i in range(n_expansions):
                        vessel_data = expansion_data[i][point - 1][vessel]
                        y.append(vessel_data[marker])

                    y_tot.append(y)

                plt.plot(pixel_expansions, np.mean(np.array(y_tot), axis=0), color=color)
            plt.xticks(pixel_expansions, fontsize=7, rotation=90)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Vessel ID: %s, Point %s Average Bins" % (str(vessel), str(point)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/vessel_%s_averagebins.png' % str(vessel), bbox_inches='tight')
            plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per vessel (One plot per bin)
    for point in points:
        output_dir = "results/point_%s_vessels_%s_interval_%s_expansions_perbin" % (
            str(point), str(interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        n_vessels = len(expansion_data[0][point - 1])

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():

                for marker, marker_name in enumerate(markers_names):
                    if marker_name not in marker_clusters[key]:
                        continue

                    y = []

                    for i in range(n_expansions):
                        vessel_data = expansion_data[i][point - 1][vessel]
                        y.append(vessel_data[marker])

                    plt.plot(pixel_expansions, y, label=marker_name)
                plt.xticks(pixel_expansions, fontsize=7, rotation=90)
                plt.xlabel("# of Pixels Expanded")
                plt.ylabel("Mean Pixel Expression")
                plt.title("Vessel ID: %s, Point %s %s" % (str(vessel), str(point), key))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(output_dir + '/vessel_%s_%s.png' % (str(vessel), key), bbox_inches='tight')
                plt.clf()


def point_region_plots(n_expansions, interval, markers_names, expansion_data):
    """
    Point region line plots

    :param n_expansions:
    :param interval:
    :param markers_names:
    :param expansion_data:
    :return:
    """
    n_points = 48

    marker_clusters = {
        "Nucleus": ["HH3"],
        "Microglia": ["CD45", "HLADR", "Iba1"],
        "Disease": ["CD47", "ABeta42", "polyubiK48", "PHFTau", "8OHGuanosine"],
        "Vessels": ["SMA", "CD31", "CollagenIV", "TrkA", "GLUT1", "Desmin", "vWF", "CD105"],
        "Astrocytes": ["S100b", "GlnSyn", "Cx30", "EAAT2", "CD44", "GFAP", "Cx43"],
        "Synapse": ["CD56", "Synaptophysin", "VAMP2", "PSD95"],
        "Oligodendrocytes": ["MOG", "MAG"],
        "Neurons": ["Calretinin", "Parvalbumin", "MAP2", "Gephyrin"]
    }

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    # Points Plots

    output_dir = "results/points_%s_interval_%s_expansions" % (str(interval), str(n_expansions - 1))
    mkdir_p(output_dir)

    # Change in Marker Expression w.r.t pixel expansion per point (All bins)
    for point in range(n_points):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for marker, marker_name in enumerate(markers_names):
            y = []
            color = None

            for key in marker_clusters.keys():
                if marker_name in marker_clusters[key]:
                    color = colors[key]
                    break

            for i in range(n_expansions):
                current = []
                point_data = expansion_data[i][point]

                for vessel in point_data:
                    current.append(vessel[marker])

                average_expression = sum(current) / len(current)
                y.append(average_expression)
            plt.plot(pixel_expansions, y, color=color)
        plt.xticks(pixel_expansions, fontsize=7, rotation=90)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Point %s" % str(point + 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/point_%s_allbins.png' % str(point + 1), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per point (Average bins)
    for point in range(n_points):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for key in marker_clusters.keys():
            color = colors[key]
            y_tot = []
            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    point_data = expansion_data[i][point]

                    for vessel in point_data:
                        current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)
                y_tot.append(y)

            plt.plot(pixel_expansions, np.mean(np.array(y_tot), axis=0), color=color)
        plt.xticks(pixel_expansions, fontsize=7, rotation=90)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Point %s" % str(point + 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/point_%s_averagebins.png' % str(point + 1), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per point (One bin per plot)
    for point in range(n_points):

        for key in marker_clusters.keys():

            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    point_data = expansion_data[i][point]

                    for vessel in point_data:
                        current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)

                plt.plot(pixel_expansions, y, label=marker_name)
            plt.xticks(pixel_expansions, fontsize=7, rotation=90)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Point %s" % str(point + 1))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/point_%s_%s.png' % (str(point + 1), str(key)), bbox_inches='tight')
            plt.clf()


def brain_region_plots(n_expansions, interval, markers_names, expansion_data):
    """
    Brain region expansion line plots
    :param n_expansions:
    :param interval:
    :param markers_names:
    :param expansion_data:
    :return:
    """
    brain_regions = [(1, 16), (17, 32), (33, 48)]

    marker_clusters = {
        "Nucleus": ["HH3"],
        "Microglia": ["CD45", "HLADR", "Iba1"],
        "Disease": ["CD47", "ABeta42", "polyubiK48", "PHFTau", "8OHGuanosine"],
        "Vessels": ["SMA", "CD31", "CollagenIV", "TrkA", "GLUT1", "Desmin", "vWF", "CD105"],
        "Astrocytes": ["S100b", "GlnSyn", "Cx30", "EAAT2", "CD44", "GFAP", "Cx43"],
        "Synapse": ["CD56", "Synaptophysin", "VAMP2", "PSD95"],
        "Oligodendrocytes": ["MOG", "MAG"],
        "Neurons": ["Calretinin", "Parvalbumin", "MAP2", "Gephyrin"]
    }

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    region_names = [
        "MFG",
        "HIP",
        "CAUD"
    ]

    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    output_dir = "results/brain_regions_%s_interval_%s_expansions" % (str(interval), str(n_expansions - 1))
    mkdir_p(output_dir)

    # Brain Region Plots

    # Change in Marker Expression w.r.t pixel expansion per brain region (All bins in one graph)
    for idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        x = np.arange(n_expansions)
        for marker, marker_name in enumerate(markers_names):
            y = []
            color = None
            for key in marker_clusters.keys():
                if marker_name in marker_clusters[key]:
                    color = colors[key]
                    break

            for i in range(n_expansions):
                current = []
                for point_data in expansion_data[i][region[0] - 1: region[1] - 1]:
                    for vessel in point_data:
                        current.append(vessel[marker])

                average_expression = sum(current) / len(current)
                y.append(average_expression)
            plt.plot(pixel_expansions, y, color=color)

        plt.xticks(pixel_expansions, fontsize=7, rotation=90)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s - All Bins" % str(region_names[idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/region_%s_allbins.png' % str(region_names[idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (Bins averaged into one line)
    for idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for key in marker_clusters.keys():
            color = colors[key]
            y_tot = []
            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    for point_data in expansion_data[i][region[0] - 1: region[1] - 1]:
                        for vessel in point_data:
                            current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)
                y_tot.append(y)

            plt.plot(pixel_expansions, np.mean(np.array(y_tot), axis=0), color=color)
        plt.xticks(pixel_expansions, fontsize=7, rotation=90)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s - Average Bins" % str(region_names[idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/region_%s_averagebins.png' % str(region_names[idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (One bin per plot)
    for idx, region in enumerate(brain_regions):
        x = np.arange(n_expansions)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    for point_data in expansion_data[i][region[0] - 1: region[1] - 1]:
                        for vessel in point_data:
                            current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)

                plt.plot(pixel_expansions, y, label=marker_name)
            plt.xticks(pixel_expansions, fontsize=7, rotation=90)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Brain Region - %s - %s" % (str(region_names[idx]), str(key)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/region_%s_%s.png' % (str(region_names[idx]), str(key)), bbox_inches='tight')
            plt.clf()


def pixel_expansion_ring_plots(n_expansions=11, interval=10, mask="allvessels", point_num=33):
    """
    Pixel expansion "Ring" plots

    :param n_expansions: # of expansions
    :param interval: Pixel expansion
    :param mask: Mask type ex. allvessels, largevessels etc.
    :param point_num: Point number to visualize
    :return:
    """

    # Convert point number to index from 0
    point_num -= 1

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data(segmentation_type=mask)

    contour_data_multiple_points = []
    contour_images_multiple_points = []
    expansions = [2, 4, 6, 8]

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    current_interval = interval
    expansion_image = np.zeros(markers_data[0][0].shape, np.uint8)

    for x in range(n_expansions):

        contours = contour_data_multiple_points[point_num]
        expansion_ring_plots(contours,
                             expansion_image,
                             pixel_expansion_amount=current_interval,
                             prev_pixel_expansion_amount=current_interval - interval)
        print("Current interval %s, previous interval %s" % (str(current_interval), str(current_interval - interval)))

        if x + 1 in expansions:
            cv.imwrite("results/expansion_plot_%s_interval_%s_expansion.png" % (str(interval), str(x + 1)),
                       expansion_image)

        current_interval += interval


def pixel_expansion_plots(n_expansions=4, interval=10, mask="allvessels"):
    """
    Pixel Expansion line plots

    :param n_expansions: # of expansions
    :param interval: Pixel interval
    :param mask: Mask type ex. allvessels, largevessels etc.
    :return:
    """

    n_expansions += 1  # Intuitively, 5 expansions means 5 expansions excluding the original composition of the
    # vessel, but we mean 5 expansions including the original composition - thus 4 expansions. Therefore lets add 1
    # so we are on the same page.
    expansions = [1, 2, 3, 4]

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data(segmentation_type=mask)

    contour_data_multiple_points = []
    contour_images_multiple_points = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    expansion_data = []
    current_interval = interval
    expansion_image = np.zeros(markers_data[0][0].shape, np.uint8)

    for x in range(n_expansions):
        current_expansion_data = []

        for i in range(len(contour_data_multiple_points)):
            contours = contour_data_multiple_points[i]
            # construct_vessel_relative_area_microenvironments_from_contours(contours, marker_segmentation_masks[i])
            marker_data = markers_data[i]
            start_expression = datetime.datetime.now()

            if x == 0:
                data = calculate_marker_composition_single_vessel(marker_data, contours,
                                                                  expression_type="mean",
                                                                  plot=False,
                                                                  vessel_id_plot=True,
                                                                  vessel_id_label="Point_%s" % str(i + 1))
            else:
                if i != 13:
                    data, _ = calculate_microenvironment_marker_expression_single_vessel(marker_data, contours,
                                                                                         expression_type="mean",
                                                                                         plot=False,
                                                                                         pixel_expansion_amount=current_interval,
                                                                                         prev_pixel_expansion_amount=current_interval - interval,
                                                                                         n_markers=len(markers_names))
                else:
                    data, _ = calculate_microenvironment_marker_expression_single_vessel(marker_data, contours,
                                                                                         expression_type="mean",
                                                                                         plot=False,
                                                                                         pixel_expansion_amount=current_interval,
                                                                                         prev_pixel_expansion_amount=current_interval - interval,
                                                                                         expansion_image=expansion_image,
                                                                                         n_markers=len(markers_names))

                    if x + 1 in expansions:
                        cv.imwrite("results/expansion_plot_%s_interval_%s_expansion.png" % (str(interval),
                                                                                            str(x + 1)),
                                   expansion_image)

            end_expression = datetime.datetime.now()

            print("Finished calculating expression %s in %s" % (str(i), end_expression - start_expression))
            print(
                "Current interval %s, previous interval %s" % (str(current_interval), str(current_interval - interval)))

            current_expansion_data.append(data)

        expansion_data.append(current_expansion_data)

        current_interval += interval

    for x in expansions:
        # brain_region_plots(x + 1, interval, markers_names, expansion_data)
        # point_region_plots(x + 1, interval, markers_names, expansion_data)
        # vessel_region_plots(x + 1, interval, markers_names, expansion_data)
        pass


def extract_vessel_heterogeneity(n=56,
                                 feature_extraction_method="vesselnet"):
    '''
    Extract vessel heterogeneity
    :return:
    '''

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data()

    contour_data_multiple_points = []
    contour_images_multiple_points = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    if feature_extraction_method == "vesselnet":
        data_loader = VesselDataset(contour_data_multiple_points, markers_data, batch_size=16, n=n)

        # vn = VesselNet(n=n)
        # vn.fit(data_loader, epochs=150)
        # vn.visualize_filters()
        # torch.save(vn.state_dict(), "trained_models/vessel_net_100.pth")

        vn = VesselNet(n=n)
        vn.load_state_dict(torch.load("trained_models/vessel_net_100.pth"))
        vn.to(torch.device("cuda"))
        # vn.visualize_filters()

        encoder_output = []
        marker_expressions = []

        for i in range(len(contour_data_multiple_points)):
            for x in range(len(contour_data_multiple_points[i])):
                contours = [contour_data_multiple_points[i][x]]
                marker_data = markers_data[i]
                expression, expression_images = calculate_microenvironment_marker_expression_single_vessel(marker_data,
                                                                                                           contours,
                                                                                                           plot=False)

                expression_image = preprocess_input(expression_images, n)
                expression_image = torch.unsqueeze(expression_image, 0)

                reconstructed_img, output = vn.forward(expression_image)

                y_pred_numpy = reconstructed_img.cpu().data.numpy()
                y_true_numpy = expression_image.cpu().data.numpy()

                row_i = np.random.choice(y_pred_numpy.shape[0], 1)
                random_pic = y_pred_numpy[row_i, :, :, :]
                random_pic = random_pic.reshape(34, n, n)

                true = y_true_numpy[row_i, :, :, :]
                true = true.reshape(34, n, n)

                # for w in range(len(random_pic)):
                #     cv.imshow("Predicted", random_pic[w] * 255)
                #     cv.imshow("True", true[w] * 255)
                #     cv.waitKey(0)

                output = output.cpu().data.numpy()
                encoder_output.append(output.reshape(2048))
                marker_expressions.append(expression[0])

    elif feature_extraction_method == "sift":
        flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

        sift = SIFTFeatureGen()
        encoder_output = sift.generate(flat_list)
    elif feature_extraction_method == "resnet":
        flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

        cnn = CNNFeatureGen()
        encoder_output = cnn.generate(flat_list)

    km = ClusteringHelper(encoder_output, n_clusters=10, metric="cosine", method="kmeans")
    indices, frequency = km.fit_predict()

    print(len(indices))

    indices = [[i, indices[i]] for i in range(len(indices))]
    values = set(map(lambda y: y[1], indices))

    grouped_indices = [[y[0] for y in indices if y[1] == x] for x in values]

    marker_expressions_grouped = []

    for i, cluster in enumerate(grouped_indices):
        temp = []
        for idx in cluster:
            temp.append(marker_expressions[idx])

        average = np.mean(temp, axis=0)

        marker_expressions_grouped.append(average)

    marker_expressions_grouped = np.array(marker_expressions_grouped)
    print(marker_expressions_grouped.shape)

    km.plot(x=marker_expressions_grouped, labels=markers_names)

    k = 10

    sampled_indices = []

    for cluster in grouped_indices:
        sampled_indices.append(random.choices(cluster, k=k))

    flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

    for i, cluster in enumerate(sampled_indices):
        for idx in cluster:
            cv.imshow("Cluster %s" % i, flat_list[idx])
            cv.waitKey(0)


def show_vessel_areas(show_outliers=False):
    '''
    Create visualizations of vessel areas

    :return: None
    '''

    masks = [
        'astrocytes',
        'BBB',
        'largevessels',
        'microglia',
        'myelin',
        'plaques',
        'tangles',
        'allvessels'
    ]

    region_names = [
        "MFG",
        "HIP",
        "CAUD"
    ]

    total_areas = [[], [], []]

    brain_regions = [(1, 16),
                     (17, 32),
                     (33, 48)]

    for segmentation_type in masks:
        current_point = 1
        current_region = 0

        marker_segmentation_masks, markers_data, markers_names = get_all_point_data(segmentation_type=segmentation_type)

        contour_images_multiple_points = []
        contour_data_multiple_points = []

        for segmentation_mask in marker_segmentation_masks:
            contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
            contour_images_multiple_points.append(contour_images)
            contour_data_multiple_points.append(contours)

        vessel_areas = plot_vessel_areas(contour_data_multiple_points, marker_segmentation_masks,
                                         segmentation_type=segmentation_type,
                                         show_outliers=show_outliers)

        for point_vessel_areas in vessel_areas:
            current_point += 1

            if not (brain_regions[current_region][0] <= current_point <= brain_regions[current_region][1]):
                current_region += 1

            if current_region < len(total_areas):
                total_areas[current_region].extend(sorted(point_vessel_areas))

    colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']

    fig = plt.figure(1, figsize=(9, 6))
    plt.title("All Objects Across Brain Regions - All Vessels")

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(total_areas, showfliers=show_outliers, patch_artist=True, labels=region_names)

    for w, region in enumerate(brain_regions):
        patch = bp['boxes'][w]
        patch.set(facecolor=colors[w])

    plt.show()


if __name__ == '__main__':
    pixel_expansion_ring_plots()
    # pixel_expansion_plots()

    # extract_vessel_heterogeneity()

    # show_vessel_areas(show_outliers=False)

    # counts = run_complete(use_flowsom=True, show_plots=True)
