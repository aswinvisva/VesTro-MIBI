from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

class ClusteringKMeans:

    def __init__(self, data):
        self.data = data
        self.clusters = 25
        self.model = None

    def fit_model(self):
        '''
        TODO: Fit the K-Means model to the encoded features
        '''

        self.model = KMeans(n_clusters=self.clusters)
        self.model.fit(self.data)

    def generate_embeddings(self):
        '''
        TODO: Get counts of features assigned to clusters to be used as bag-of-words feature embeddings
        '''

        bag_of_words = []

        for data_point in self.data:
            clusters = self.model.predict(data_point.reshape(1, -1))
            embedding = [0] * self.clusters

            c = Counter(clusters)
            keys = c.keys()

            for key in keys:
                embedding[key] = c[key]

            bag_of_words.append(embedding)

        return bag_of_words
