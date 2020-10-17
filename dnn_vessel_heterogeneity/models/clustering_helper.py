import os
import pickle
from collections import Counter

import matplotlib
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


class ClusteringHelper:

    def __init__(self,
                 data,
                 method="optics",
                 n_clusters=10,
                 pretrained=False,
                 show_plots=True,
                 metric="cosine",
                 save=False):

        '''
        Clustering for general purposes

        :param data: Input data
        :param clusters: Number of clusters
        :param pretrained: Is the model pretrained
        :param show_plots: Show each of the plots?
        '''

        self.data = data
        self.method = method
        self.n_clusters = n_clusters
        self.pretrained = pretrained
        self.model = None
        self.show_plots = show_plots
        self.metric = metric
        self.save = save

    def fit_predict(self):
        """
        Fit model and return embeddings

        :return:
        """

        self.fit_model()

        return self.generate_embeddings()

    def fit_model(self):
        '''
        Fit model and save if not pretrained

        :return: None
        '''

        if self.show_plots:
            self.elbow_method()

        if not self.pretrained:
            if self.method == "kmeans":
                self.model = KMeans(n_clusters=self.n_clusters)
                self.model.fit(self.data)
            elif self.method == "dbscan":
                self.model = DBSCAN(metric=self.metric, eps=0.15)
                self.model.fit(self.data)
            elif self.method == "optics":
                self.model = OPTICS(metric=self.metric)
                self.model.fit(self.data)
            elif self.method == "hierarichal":
                self.model = linkage(self.data, metric=self.metric)

            if not self.save:
                pickle.dump(self.model, open("trained_models/%s_model.pkl" % self.method, "wb"))

        else:
            self.model = pickle.load(open("trained_models/%s_model.pkl" % self.method, "rb"))

    def generate_embeddings(self):
        '''
        Get labels for each of the cells

        :return: None
        '''

        if not self.method == "hierarichal":
            labels = self.model.labels_.tolist()
        else:
            labels = leaves_list(self.model).tolist()

        print(labels)

        c = Counter(labels)

        cluster_counts = []

        for i in range(self.n_clusters):
            if i in c.keys():
                cluster_counts.append(c[i])
            else:
                cluster_counts.append(0)

        plt.bar(range(self.n_clusters), cluster_counts)
        plt.xticks(range(self.n_clusters), range(self.n_clusters))
        plt.xlabel('Cluster')
        plt.ylabel('Frequency')

        plt.show()

        return labels, cluster_counts

    def elbow_method(self):
        '''
        Elbow method for determining optimal 'k' for K-Means

        :return: None
        '''

        # k means determine k
        distortions = []
        K = range(1, 10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(self.data)
            distortions.append(km.inertia_)

        # Plot the elbow
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')

        plt.show()

    def plot(self, x=None, labels=None):

        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1.0), "midnightblue"],
                  [norm(-0.5), "seagreen"],
                  [norm(0.5), "mediumspringgreen"],
                  [norm(1.0), "yellow"]]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        if x is None:
            ax = sns.clustermap(self.model.cluster_centers_,
                                linewidth=0.5,
                                cmap=cmap
                                )
        else:
            ax = sns.clustermap(x,
                                linewidth=0.5,
                                cmap=cmap,
                                xticklabels=labels
                                )
        plt.show()
