import datetime
import random
from collections import Counter
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from marker_processing.k_means_clustering import ClusteringKMeans
from marker_processing.stitch_markers import image_stitching
from topic_generation.lda_topic_generation import LDATopicGen
from image_segmentation.watershed_segmentation import *
from image_segmentation.sliding_window_segmentation import *
from feature_extraction import bag_of_cells_feature_gen
from marker_processing.markers_feature_gen import *

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def run_complete(size=4,
                 point="Point13",
                 no_topics=4,
                 use_watershed=True,
                 use_test_data=False,
                 pretrained=True,
                 show_plots=True):

    '''
    Run execution to get segmented image with cell labels

    :param size: (If using sliding window) Window side length
    :param point: Point number to get data
    :param no_topics: Number of clusters for K-Means
    :param use_watershed: Should use watershed segmentation?
    :param use_test_data: Should use online data?
    :param pretrained: Is K-Means model pre-trained?
    :param show_plots: Should show plots?
    :return:
    '''

    begin_time = datetime.datetime.now()

    if use_test_data:
        image = cv.imread('data/Images/michael-angelo-breast-cancer-cells.jpg')
    else:
        image, marker_data, marker_names = image_stitching(point_name=point)

    if use_watershed:
        images, contours = oversegmentation_watershed(image)
    else:
        images = split_image(image, n=size)

    data = mean_normalized_expression(marker_data, contours)

    model = ClusteringKMeans(data,
                             point,
                             marker_names,
                             clusters=no_topics,
                             pretrained=pretrained,
                             show_plots=show_plots)
    model.elbow_method()
    model.fit_model()
    indices = model.generate_embeddings()

    if use_watershed:
        label_image_watershed(image, contours, indices, show_plot=show_plots)
    else:
        label_image(image, indices, topics=no_topics, n=size)

    end_time = datetime.datetime.now()

    print("Segmentation finished. Time taken:", end_time-begin_time)



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
    topics = topic_gen_model.fit_model()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    label_image(np_color_im, indices, topics=no_topics, n=square_side_length)


if __name__ == '__main__':
    run_complete()
