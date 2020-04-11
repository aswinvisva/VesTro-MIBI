from feature_extraction import sift_feature_gen
from feature_clustering.k_means_clustering import ClusteringKMeans
from topic_generation.lda_topic_generation import LDATopicGen
from image_preprocessing import sliding_window
import cv2 as cv

if __name__ == '__main__':
    image = cv.imread('Images/michael-angelo-breast-cancer-cells.jpg')
    images = sliding_window.split_image(image, n=50)

    data = []
    for img in images:
        data_points = sift_feature_gen.generate(img, show_keypoints=False)

        if data_points is not None:
            data.append(data_points)

    model = ClusteringKMeans(data)
    model.fit_model()
    bag_of_visual_words = model.generate_embeddings()
    print(len(bag_of_visual_words))

    topic_gen_model = LDATopicGen(bag_of_visual_words)
    topics = topic_gen_model.fit_model()

    print(topics)





