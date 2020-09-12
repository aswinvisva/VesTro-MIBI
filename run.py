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


def pixel_expansion_plots(n_expansions=10, interval=5):
    n_expansions += 1  # Intuitively, 5 expansions means 5 expansions excluding the original composition of the
    # vessel, but we mean 5 expansions including the original composition - thus 4 expansions. Therefore lets add 1
    # so we are on the same page.

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data()

    contour_data_multiple_points = []
    contour_images_multiple_points = []
    n_points = 48

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
                                                                  plot=False)
            else:
                if i != 2:
                    data, _ = calculate_microenvironment_marker_expression_single_vessel(marker_data, contours,
                                                                                         expression_type="mean",
                                                                                         plot=False,
                                                                                         pixel_expansion_amount=current_interval,
                                                                                         prev_pixel_expansion_amount=current_interval - interval)
                else:
                    data, _ = calculate_microenvironment_marker_expression_single_vessel(marker_data, contours,
                                                                                         expression_type="mean",
                                                                                         plot=False,
                                                                                         pixel_expansion_amount=current_interval,
                                                                                         prev_pixel_expansion_amount=current_interval - interval,
                                                                                         expansion_image=expansion_image)

            end_expression = datetime.datetime.now()

            print("Finished calculating expression %s in %s" % (str(i), end_expression - start_expression))
            print(
                "Current interval %s, previous interval %s" % (str(current_interval), str(current_interval - interval)))

            current_expansion_data.append(data)

        expansion_data.append(current_expansion_data)

        current_interval += interval

    cv.imshow("Expansion Rings", expansion_image)
    cv.waitKey(0)

    brain_regions = [(1, 16), (17, 32), (33, 48)]

    marker_clusters = {
        "Nucleus": ["HH3"],
        "Microglia": ["CD45", "HLADR", "Iba1"],
        "Disease": ["CD47", "ABeta42", "polyubiK48", "PHFTau", "8OHGuanosine"],
        "Vessels": ["SMA", "CD31", "CollagenIV", "TrkA", "GLUT1", "Desmin", "vWF", "CD105"],
        "Astrocytes": ["S100beta", "GluSyn", "Cx30", "EAAT2", "CD44", "GFAP", "Cx43"],
        "Synapse": ["CD56", "Synaptophysin", "VAMP2", "PSD95"],
        "Oligodendrocytes": ["MOG", "MAG"],
        "Neurons": ["Calretinin", "Parvalbumin", "MAP2", "Gephrin"]
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

        plt.xticks(pixel_expansions)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s" % str(region_names[idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("results/region_%s_allbins.png" % str(region_names[idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (Bins averaged into one line)
    for idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        x = np.arange(n_expansions)

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
        plt.xticks(pixel_expansions)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s" % str(region_names[idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("results/region_%s_averagebins.png" % str(region_names[idx]), bbox_inches='tight')
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
            plt.xticks(pixel_expansions)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Brain Region - %s" % str(region_names[idx]))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig("results/region_%s_%s.png" % (str(region_names[idx]), str(key)), bbox_inches='tight')
            plt.clf()

    # Points Plots

    # Change in Marker Expression w.r.t pixel expansion per point
    for point in range(n_points):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        print("Point %s" % str(point + 1))
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
                point_data = expansion_data[i][point]

                for vessel in point_data:
                    current.append(vessel[marker])

                average_expression = sum(current) / len(current)
                y.append(average_expression)
            plt.plot(pixel_expansions, y, color=color)
        plt.xticks(pixel_expansions)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Point %s" % str(point + 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("results/point_%s.png" % str(point + 1), bbox_inches='tight')
        plt.clf()


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


def run_complete(size=512,
                 no_environments=5,
                 point="multiple",
                 no_phenotypes=12,
                 use_flowsom=True,
                 use_cnnfcluster=False,
                 pretrained=False,
                 show_plots=True):
    """
    Run execution to get segmented image with cell labels

    :param no_environments:
    :param use_cnnfcluster:
    :param use_flowsom: Use FlowSOM for marker clustering
    :param size: (If using sliding window) Window side length
    :param point: Point number to get data
    :param no_phenotypes: Number of clusters for K-Means
    :param pretrained: Is K-Means model pre-trained?
    :param show_plots: Should show plots?
    :return:
    """

    begin_time = datetime.datetime.now()

    if point == "multiple":
        marker_segmentation_masks, markers_data, markers_names = get_all_point_data()
    else:
        segmentation_mask, marker_data, marker_names = read(point_name=point)

    if point == "multiple":
        contour_images_multiple_points = []
        contour_data_multiple_points = []

        for segmentation_mask in marker_segmentation_masks:
            contour_images, contours = extract_cell_contours(segmentation_mask)
            contour_images_multiple_points.append(contour_images)
            contour_data_multiple_points.append(contours)

        if show_plots:
            plot_vessel_areas(contour_data_multiple_points, marker_segmentation_masks)
    else:
        contour_images, contours = extract_cell_contours(segmentation_mask)

    if point == "multiple":
        points_expression = None

        for i in range(len(contour_data_multiple_points)):
            contours = contour_data_multiple_points[i]
            # construct_vessel_relative_area_microenvironments_from_contours(contours, marker_segmentation_masks[i])
            marker_data = markers_data[i]
            start_expression = datetime.datetime.now()
            data = calculate_marker_composition_single_vessel(marker_data, contours, plot=False)
            end_expression = datetime.datetime.now()

            print("Finished calculating expression %s in %s" % (str(i), end_expression - start_expression))

            if points_expression is None:
                points_expression = data
            else:
                points_expression = np.append(points_expression, data, axis=0)

        print("There are %s samples" % points_expression.shape[0])

    else:
        data = calculate_marker_composition_single_vessel(marker_data, contours)

        print("There are %s samples" % len(contours))

    if not use_flowsom:
        if point == "multiple":
            marker_names = markers_names[0]

            model = ClusteringHelper(points_expression,
                                     point,
                                     marker_names,
                                     clusters=no_phenotypes,
                                     pretrained=pretrained,
                                     show_plots=show_plots)
            model.elbow_method()
            model.fit_model()
            indices, cell_counts = model.generate_embeddings()
        else:
            model = ClusteringHelper(data,
                                     point,
                                     marker_names,
                                     clusters=no_phenotypes,
                                     pretrained=pretrained,
                                     show_plots=show_plots)
            model.elbow_method()
            model.fit_model()
            indices, cell_counts = model.generate_embeddings()
    else:
        if point == "multiple":
            marker_names = markers_names[0]

            model = ClusteringFlowSOM(points_expression,
                                      point,
                                      marker_names,
                                      clusters=no_phenotypes,
                                      explore_clusters=0,
                                      pretrained=pretrained,
                                      show_plots=show_plots)
            model.fit_model()
            indices, cell_counts = model.predict()
        else:
            model = ClusteringFlowSOM(data,
                                      point,
                                      marker_names,
                                      clusters=no_phenotypes,
                                      pretrained=pretrained,
                                      show_plots=show_plots)
            model.fit_model()
            indices, cell_counts = model.predict()

    if point == "multiple":
        prev_index = 0
        instance_segmentation_masks_multiple_markers = []
        complete_data = []

        for x in range(len(contour_data_multiple_points)):
            contours = contour_data_multiple_points[x]
            i = indices[prev_index:prev_index + len(contours)]

            segmentation_mask = marker_segmentation_masks[x]

            segmented_image, data = label_image_watershed(segmentation_mask, contours, i,
                                                          show_plot=show_plots,
                                                          no_topics=no_phenotypes)
            instance_segmentation_masks_multiple_markers.append(segmented_image)
            complete_data.append(data)
            prev_index = len(contours)
    else:
        segmented_image, data = label_image_watershed(segmentation_mask, contours, indices,
                                                      show_plot=show_plots,
                                                      no_topics=no_phenotypes)

    end_time = datetime.datetime.now()

    print("Segmentation finished. Time taken:", end_time - begin_time)

    if point == "multiple":
        split_marker_segmentation_masks = []
        complete_split_data = []

        for i in range(len(instance_segmentation_masks_multiple_markers)):
            segmented_image = instance_segmentation_masks_multiple_markers[i]
            data = complete_data[i]

            split_segmented_images = split_image(segmented_image, n=size)
            split_data = split_image(data, n=size)

            split_marker_segmentation_masks.extend(split_segmented_images)
            complete_split_data.append(split_data)
    else:
        segmented_images = split_image(segmented_image, n=size)
        split_data = split_image(data, n=size)

    if point == "multiple":
        bag_of_words_data = None
        for split_data in complete_split_data:

            data = bag_of_cells_feature_gen.generate(split_data, vector_size=no_phenotypes)

            if bag_of_words_data is None:
                bag_of_words_data = data
            else:
                bag_of_words_data = np.append(bag_of_words_data, data, axis=0)
    else:
        bag_of_words_data = bag_of_cells_feature_gen.generate(split_data, vector_size=no_phenotypes)

    if use_cnnfcluster:
        vec_map = {}

        cnn = CNNFeatureGen(n=size)

        for i in range(len(segmented_images)):
            img = segmented_images[i]
            print("Data", bag_of_words_data[i])
            v = cnn.generate(img)
            vec_map[str(bag_of_words_data[i].tolist())] = v

        cnnfcluster = CNNFCluster()
        label_list, clusters = cnnfcluster.fit_predict(bag_of_words_data, vec_map)
        label_image(segmented_image, label_list, n=size, topics=clusters)

    else:
        if point == "multiple":
            X = []

            for img in split_marker_segmentation_masks:
                # img = preprocess_input(img)
                X.append(img)

            X = np.array(X).reshape((len(split_marker_segmentation_masks), size, size, 3))

            cnnlda = CNNLDA(K=no_environments, n=size, phenotypes=no_phenotypes)
            cnnlda.fit(X, bag_of_words_data)
            cnnlda.plot()
            cnnlda.view_microenvironments()

            # lda = LDATopicGen(bag_of_words_data, topics=no_environments)
            # topics = lda.fit_predict()
            # lda.plot()
            #
            # indices = []
            #
            # for idx, topic in enumerate(topics):
            #     indices.append(np.where(topic == topic.max())[0][0])
            #
            # label_image(segmented_image, indices, n=size, topics=no_environments)
        else:
            lda = LDATopicGen(bag_of_words_data, topics=no_environments)
            topics = lda.fit_predict()
            lda.plot()

            indices = []

            for idx, topic in enumerate(topics):
                indices.append(np.where(topic == topic.max())[0][0])

            label_image(segmented_image, indices, n=size, topics=no_environments)

    return cell_counts


def run_TNBC_dataset_test(square_side_length=50,
                          no_topics=10,
                          img_loc='data/TNBC_shareCellData/p23_labeledcellData.tiff'):
    im = Image.open(img_loc)
    color_im = im.convert("RGB")

    np_im = np.array(im)
    np_color_im = np.array(color_im)
    np_color_im = np_color_im.reshape(2048, 2048, 3)
    np_im = np_im.reshape(2048, 2048, 1)

    c = Counter(np_im.flatten())
    keys = c.keys()
    vector_size = max(keys) + 1

    images = split_image(np_im, n=square_side_length)

    bag_of_visual_words = bag_of_cells_feature_gen.generate(images, vector_size=vector_size)

    topic_gen_model = LDATopicGen(bag_of_visual_words, topics=no_topics)
    topics = topic_gen_model.fit_predict()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    label_image(np_color_im, indices, topics=no_topics, n=square_side_length)


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
    pixel_expansion_plots()

    # extract_vessel_heterogeneity()

    # show_vessel_areas(show_outliers=False)

    # counts = run_complete(use_flowsom=True, show_plots=True)
