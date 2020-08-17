import datetime
import random
from collections import Counter
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from marker_processing.k_means_clustering import ClusteringKMeans
from marker_processing.flowsom_clustering import ClusteringFlowSOM
from topic_generation.cnn_lda import CNNLDA
from utils.construct_microenvironments import construct_partitioned_microenvironments_from_contours
from utils.mibi_reader import read, concatenate_multiple_points
from topic_generation.lda_topic_generation import LDATopicGen
from image_segmentation.sliding_window_segmentation import split_image
from image_segmentation.watershed_segmentation import *
from image_segmentation.acwe_segmentation import *
from image_segmentation.sliding_window_segmentation import *
from feature_extraction import bag_of_cells_feature_gen
from feature_extraction.cnn_feature_gen import CNNFeatureGen
from marker_processing.markers_feature_gen import *
from topic_generation.cnnfcluster import CNNFCluster
from topic_generation.lda_topic_generation import LDATopicGen
from topic_generation.lda import LDA, lda_learning

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def run_complete(size=256,
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
        marker_segmentation_masks, markers_data, markers_names = concatenate_multiple_points()
    else:
        segmentation_mask, marker_data, marker_names = read(point_name=point)

    if point == "multiple":
        contour_images_multiple_points = []
        contour_data_multiple_points = []

        for segmentation_mask in marker_segmentation_masks:
            contour_images, contours = oversegmentation_watershed(segmentation_mask)
            contour_images_multiple_points.append(contour_images)
            contour_data_multiple_points.append(contours)

        if show_plots:
            plot_vessel_areas(contour_data_multiple_points, marker_segmentation_masks)
    else:
        contour_images, contours = oversegmentation_watershed(segmentation_mask)

    if point == "multiple":
        points_expression = None

        for i in range(len(contour_data_multiple_points)):
            contours = contour_data_multiple_points[i]
            marker_data = markers_data[i]
            start_expression = datetime.datetime.now()
            data = calculate_protein_expression_single_cell(marker_data, contours, plot=False)
            end_expression = datetime.datetime.now()

            print("Finished calculating expression %s in %s" % (str(i), end_expression - start_expression))

            if points_expression is None:
                points_expression = data
            else:
                points_expression = np.append(points_expression, data, axis=0)

        print("There are %s samples" % points_expression.shape[0])

    else:
        data = calculate_protein_expression_single_cell(marker_data, contours)

        print("There are %s samples" % len(contours))

    if not use_flowsom:
        if point == "multiple":
            marker_names = markers_names[0]

            model = ClusteringKMeans(points_expression,
                                     point,
                                     marker_names,
                                     clusters=no_phenotypes,
                                     pretrained=pretrained,
                                     show_plots=show_plots)
            model.elbow_method()
            model.fit_model()
            indices, cell_counts = model.generate_embeddings()
        else:
            model = ClusteringKMeans(data,
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


if __name__ == '__main__':
    counts = run_complete(use_flowsom=True, show_plots=False)
