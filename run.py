import random
from collections import Counter
import os

from feature_extraction import sift_feature_gen
from feature_clustering.k_means_clustering import ClusteringKMeans
from topic_generation.lda_topic_generation import LDATopicGen
from image_preprocessing import sliding_window
from image_preprocessing.core import marker_stitching
from feature_extraction import bag_of_cells_feature_gen
from feature_extraction.cnn_feature_gen import CNNFeatureGen
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


def run_sift_test(image=cv.imread('Images/michael-angelo-breast-cancer-cells.jpg'),
                  size=32,
                  no_topics=5,
                  vec_size=32):

    images, contours = sliding_window.oversegmentation_watershed(image)
    # images = sliding_window.split_image(image, n=size)

    model = CNNFeatureGen(n=size)

    data = []
    for img in images:
        # data_points = sift_feature_gen.generate(img, show_keypoints=False)
        data_points = model.generate(img)

        if data_points is not None:
            data.append(data_points)

    data = np.array(data).reshape(len(data), 2048)
    pca = PCA(n_components=vec_size)
    pca.fit(data)
    data = pca.transform(data)
    print(data[0:1])

    model = ClusteringKMeans(data)
    model.fit_model()
    bag_of_visual_words = model.generate_embeddings()
    print(bag_of_visual_words)

    topic_gen_model = LDATopicGen(bag_of_visual_words, topics=no_topics)
    topics = topic_gen_model.fit_model()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    print(indices)

    sliding_window.label_image_watershed(image, contours, indices, topics=no_topics)
    # sliding_window.label_image(image, indices, topics=50, n=size)


def run_TNBC_dataset_test(square_side_length=50,
                          no_topics=10,
                          img_loc='TNBC_shareCellData/p23_labeledcellData.tiff'):
    im = Image.open(img_loc)
    color_im = im.convert("RGB")
    # im.show()

    np_im = np.array(im)
    np_color_im = np.array(color_im)
    np_color_im = np_color_im.reshape(2048, 2048, 3)
    np_im = np_im.reshape(2048, 2048, 1)

    c = Counter(np_im.flatten())
    keys = c.keys()
    vector_size = max(keys) + 1
    print(c)

    images = sliding_window.split_image(np_im, n=square_side_length)

    bag_of_visual_words = bag_of_cells_feature_gen.generate(images, vector_size=vector_size)

    topic_gen_model = LDATopicGen(bag_of_visual_words, topics=no_topics)
    topics = topic_gen_model.fit_model()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    sliding_window.label_image(np_color_im, indices, topics=no_topics, n=square_side_length)


if __name__ == '__main__':
    # run_sift_test()
    # run_TNBC_dataset_test()
    img = marker_stitching()
    run_sift_test(image=img)
