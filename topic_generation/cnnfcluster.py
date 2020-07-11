import numpy as np
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.utils.metric import type_metric, distance_metric
from scipy.spatial.distance import *
from scipy.cluster.hierarchy import fclusterdata

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


class CNNFCluster:

    def __init__(self):
        self._cluster = None
        self.vec_map = None

    def cnn_distance(self, point1, point2):
        distance = (cosine(self.vec_map[str(point1.tolist())], self.vec_map[str(point2.tolist())]) + hamming(point1, point2)) ** 0.5
        print(cosine(self.vec_map[str(point1.tolist())], self.vec_map[str(point2.tolist())]), hamming(point1, point2))
        return distance

    def fit_predict(self, bag_of_cells, vec_map):
        self.vec_map = vec_map
        fclust1 = fclusterdata(bag_of_cells, 1.0, metric=self.cnn_distance)
        fclust2 = fclusterdata(bag_of_cells, 1.0, metric='euclidean')

        clusters = max(fclust1) + 1

        print(clusters)

        print(fclust1)
        print(fclust2)

        return fclust1, clusters

