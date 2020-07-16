import datetime
import random
from collections import Counter
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from marker_processing.k_means_clustering import ClusteringKMeans
from marker_processing.flowsom_clustering import ClusteringFlowSOM
from marker_processing.stitch_markers import image_stitching
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

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def run_complete(size=256,
                 no_environments=16,
                 point="Point13",
                 no_phenotypes=25,
                 use_flowsom=True,
                 use_watershed=True,
                 use_test_data=False,
                 use_cnnfcluster=True,
                 pretrained=False,
                 show_plots=True):
    """
    Run execution to get segmented image with cell labels

    :param use_flowsom: Use FlowSOM for marker clustering
    :param size: (If using sliding window) Window side length
    :param point: Point number to get data
    :param no_phenotypes: Number of clusters for K-Means
    :param use_watershed: Should use watershed segmentation?
    :param use_test_data: Should use online data?
    :param pretrained: Is K-Means model pre-trained?
    :param show_plots: Should show plots?
    :return:
    """

    begin_time = datetime.datetime.now()

    if use_test_data:
        image = cv.imread('data/Images/michael-angelo-breast-cancer-cells.jpg')
    else:
        image, marker_data, marker_names = image_stitching(point_name=point)

    if use_watershed:
        images, contours = oversegmentation_watershed(image)
    else:
        phi0 = initialize(1024, 1024, x_center=512, y_center=512, radius=100)
        acwe(np.array(image), phi0, max_iter=100, time_step=0.1, mu=0, v=0, lambda1=1, lambda2=1, epsilon=1)
        # images = split_image(image, n=size)

    data = mean_normalized_expression(marker_data, contours)

    if not use_flowsom:
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
        model = ClusteringFlowSOM(data,
                                  point,
                                  marker_names,
                                  clusters=no_phenotypes,
                                  pretrained=pretrained,
                                  show_plots=show_plots)
        model.fit_model()
        indices, cell_counts = model.predict()

    if use_watershed:
        segmented_image, data = label_image_watershed(image, contours, indices, show_plot=show_plots, no_topics=no_phenotypes)
    else:
        label_image(image, indices, topics=no_phenotypes, n=size)

    end_time = datetime.datetime.now()

    print("Segmentation finished. Time taken:", end_time - begin_time)

    segmented_images = split_image(segmented_image, n=size)
    split_data = split_image(data, n=size)

    vec_map = {}
    bag_of_words_data = bag_of_cells_feature_gen.generate(split_data, vector_size=no_phenotypes)

    if use_cnnfcluster:
        cnn = CNNFeatureGen(n=size)

        for i in range(len(segmented_images)):
            img = segmented_images[i]
            print("Data", bag_of_words_data[i])
            cv.imshow("Image", img)
            cv.waitKey(0)
            v = cnn.generate(img)
            vec_map[str(bag_of_words_data[i].tolist())] = v

        cnnfcluster = CNNFCluster()
        label_list, clusters = cnnfcluster.fit_predict(bag_of_words_data, vec_map)
        label_image(segmented_image, label_list, n=size, topics=clusters)

    else:
        lda = LDATopicGen(bag_of_words_data, topics=no_environments)
        topics = lda.fit_predict()

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
    print(c)

    images = split_image(np_im, n=square_side_length)

    bag_of_visual_words = bag_of_cells_feature_gen.generate(images, vector_size=vector_size)

    topic_gen_model = LDATopicGen(bag_of_visual_words, topics=no_topics)
    topics = topic_gen_model.fit_predict()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    label_image(np_color_im, indices, topics=no_topics, n=square_side_length)


if __name__ == '__main__':
    counts = run_complete(use_flowsom=True, show_plots=True)
