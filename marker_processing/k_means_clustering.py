import os
import pickle
from collections import Counter

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


class ClusteringKMeans:

    def __init__(self,
                 data,
                 point_name,
                 x_labels,
                 clusters=10,
                 pretrained=False,
                 show_plots=False):

        '''
        K-Means algorithm for clustering marker distributions

        :param data: Marker data for each cell
        :param point_name: The name of the point being used
        :param x_labels: Marker labels for plots
        :param clusters: Number of clusters (Cells) to be used
        :param pretrained: Is the model pretrained
        :param show_plots: Show each of the plots?
        '''

        self.data = data
        self.clusters = clusters
        self.pretrained = pretrained
        self.model = None
        self.x_labels = x_labels
        self.point_name = point_name
        self.show_plots = show_plots

    def fit_model(self):
        '''
        Fit model and save if not pretrained

        :return: None
        '''

        if not self.pretrained:
            self.model = KMeans(n_clusters=self.clusters)
            self.model.fit(self.data)
            pickle.dump(self.model, open("models/kmeans_model.pkl", "wb"))

        else:
            self.model = pickle.load(open("models/kmeans_model.pkl", "rb"))

        for i in range(len(self.model.cluster_centers_)):
            figure(num=None, figsize=(25, 12), facecolor='w', edgecolor='k')
            plt.bar(range(len(self.model.cluster_centers_[i])), self.model.cluster_centers_[i])
            plt.xticks(range(len(self.model.cluster_centers_[i])), self.x_labels, rotation='vertical')
            plt.xlabel('Markers')
            plt.ylabel('Mean Normalized Expression')

            if self.show_plots:
                plt.savefig(os.path.join('annotated_data', 'cell_type_' + str(i) + '.png'))
                plt.show()

    def generate_embeddings(self):
        '''
        Get labels for each of the cells

        :return: None
        '''

        labels = []

        for data_point in self.data:
            clusters = self.model.predict(data_point.reshape(1, -1))

            labels.append(clusters[0])

        c = Counter(labels)

        cell_counts = []

        for i in range(self.clusters):
            if i in c.keys():
                cell_counts.append(c[i])
            else:
                cell_counts.append(0)

        plt.bar(range(self.clusters), cell_counts)
        plt.xticks(range(self.clusters), range(self.clusters))
        plt.xlabel('Cell Type')
        plt.ylabel('Frequency')

        if self.show_plots:
            plt.savefig(os.path.join('annotated_data', self.point_name, 'cell_counts.png'))
            plt.show()

        return labels, cell_counts

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
        plt.savefig(os.path.join('annotated_data', self.point_name, 'elbow_method.png'))

        if self.show_plots:
            plt.show()
