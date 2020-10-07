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
from utils.utils_functions import mkdir_p
from utils.visualizer import mask_nonmask_heatmap, point_region_plots, vessel_region_plots, brain_region_plots

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def pixel_expansion_plots(n_expansions=1, interval=5, mask="allvessels"):
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

    expansions = [1]  # Expansions that you want to run

    assert n_expansions >= max(expansions), "More expansions selected than available!"

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data(segmentation_type=mask)

    contour_data_multiple_points = []
    contour_images_multiple_points = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    expansion_data = []
    dark_space_expansion_data = []
    vessel_space_expansion_data = []
    current_interval = interval

    for x in range(n_expansions):

        current_expansion_data = []
        current_dark_space_expansion_data = []
        current_vessel_space_expansion_data = []

        all_points_stopped_vessels = 0

        for i in range(len(contour_data_multiple_points)):
            contours = contour_data_multiple_points[i]
            marker_data = markers_data[i]
            start_expression = datetime.datetime.now()

            if x == 0:
                data = calculate_composition_marker_expression(marker_data, contours,
                                                               expression_type="area_normalized_counts",
                                                               plot=False,
                                                               vessel_id_plot=True,
                                                               vessel_id_label="Point_%s" % str(i + 1))
            else:
                data, _, stopped_vessels, dark_space_data, vessel_space_data = calculate_microenvironment_marker_expression(
                    marker_data,
                    contours,
                    expression_type="area_normalized_counts",
                    plot=False,
                    pixel_expansion_upper_bound=current_interval,
                    pixel_expansion_lower_bound=current_interval - interval,
                    n_markers=len(markers_names),
                    plot_vesselnonvessel_mask=True,
                    vesselnonvessel_label="Point_%s" % str(i + 1))

                all_points_stopped_vessels += stopped_vessels
                current_dark_space_expansion_data.append(dark_space_data)
                current_vessel_space_expansion_data.append(vessel_space_data)

            end_expression = datetime.datetime.now()

            print("Finished calculating expression %s in %s" % (str(i), end_expression - start_expression))
            print(
                "Current interval %s, previous interval %s" % (str(current_interval), str(current_interval - interval)))

            current_expansion_data.append(data)

        print("There were %s vessels which could not expand inward/outward by %s pixels" % (
            all_points_stopped_vessels, x * interval))

        expansion_data.append(current_expansion_data)
        dark_space_expansion_data.append(current_dark_space_expansion_data)
        vessel_space_expansion_data.append(current_vessel_space_expansion_data)

        if x != 0:
            current_interval += interval

    for x in expansions:
        mask_nonmask_heatmap(vessel_space_expansion_data, dark_space_expansion_data, markers_names, x + 1, interval)
        brain_region_plots(x + 1, interval, markers_names, expansion_data)
        point_region_plots(x + 1, interval, markers_names, expansion_data)
        vessel_region_plots(x + 1, interval, markers_names, expansion_data)


def extract_vessel_heterogeneity(n=56,
                                 feature_extraction_method="vesselnet"):
    """
    Extract vessel heterogeneity
    :return:
    """

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
                expression, expression_images, stopped_vessels, _ = calculate_microenvironment_marker_expression(
                    marker_data,
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


if __name__ == '__main__':
    # pixel_expansion_ring_plots()
    pixel_expansion_plots()

    # extract_vessel_heterogeneity()

    # show_vessel_areas(show_outliers=False)

    # counts = run_complete(use_flowsom=True, show_plots=True)
