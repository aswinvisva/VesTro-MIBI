from collections import Counter

from feature_extraction import sift_feature_gen
from feature_clustering.k_means_clustering import ClusteringKMeans
from topic_generation.lda_topic_generation import LDATopicGen
from image_preprocessing import sliding_window
from feature_extraction import bag_of_cells_feature_gen
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_sift_test():
    image = cv.imread('Images/michael-angelo-breast-cancer-cells.jpg')
    images = sliding_window.split_image(image, n=50)

    print(images)

    data = []
    for img in images:
        data_points = sift_feature_gen.generate(img, show_keypoints=False)

        if data_points is not None:
            data.append(data_points)

    model = ClusteringKMeans(data)
    model.fit_model()
    bag_of_visual_words = model.generate_embeddings()

    topic_gen_model = LDATopicGen(bag_of_visual_words, topics=5)
    topics = topic_gen_model.fit_model()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    print(indices)

    sliding_window.label_image(image, indices, topics=5, n=50)

def run_TNBC_dataset_test():
    im = Image.open('TNBC_shareCellData/p40_labeledcellData.tiff')
    color_im = im.convert("RGB")
    # im.show()

    np_im = np.array(im)
    np_color_im = np.array(color_im)
    np_color_im = np_color_im.reshape(2048, 2048, 3)
    np_im = np_im.reshape(2048, 2048, 1)

    c = Counter(np_im.flatten())
    print(c)

    images = sliding_window.split_image(np_im, n=25)

    bag_of_visual_words = bag_of_cells_feature_gen.generate(images)

    topic_gen_model = LDATopicGen(bag_of_visual_words, topics=10)
    topics = topic_gen_model.fit_model()

    indices = []

    for idx, topic in enumerate(topics):
        indices.append(np.where(topic == topic.max())[0][0])

    sliding_window.label_image(np_color_im, indices, topics=10, n=25)

if __name__ == '__main__':
    run_TNBC_dataset_test()




