import os
import pickle
from collections import Counter

from flowsom.cluster import ConsensusCluster
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from flowsom import *
from minisom import MiniSom
from marker_processing.k_means_clustering import ClusteringKMeans


class ClusteringFlowSOM:

    def __init__(self,
                 data,
                 point_name,
                 x_labels,
                 clusters=10,
                 pretrained=False,
                 show_plots=False,
                 x_n=10,
                 y_n=10,
                 d=34):
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
        self.x_n = x_n
        self.y_n = y_n
        self.d = d

    def fit_model(self):
        '''
        Fit model and save if not pretrained

        :return: None
        '''

        self.som_mapping(self.x_n, self.y_n, self.d, sigma=2.5, lr=0.1)

    def generate_embeddings(self):
        flatten_weights = self.model.get_weights().reshape(self.x_n * self.y_n, self.d)
        print(flatten_weights.shape)

        # initialize cluster
        cluster_ = ConsensusCluster(KMeans,
                                    self.clusters, self.clusters + 10, 3)

        cluster_.fit(flatten_weights, verbose=True)  # fitting SOM weights into clustering algorithm

        # get the prediction of each weight vector on meta clusters (on bestK)
        flatten_class = cluster_.predict_data(flatten_weights)
        map_class = flatten_class.reshape(self.x_n, self.y_n)

        label_list = []
        for i in range(len(self.data)):
            # print the milestone
            if i % 10000 == 0:
                print('%d samples done...' % i)

            xx = self.data[i, :]  # fetch the sample data
            winner = self.model.winner(xx)  # make prediction, prediction = the closest entry location in the SOM
            c = map_class[winner]  # from the location info get cluster info
            label_list.append(c)

        c = Counter(label_list)

        cell_counts = []

        for i in range(self.clusters):
            if i in c.keys():
                cell_counts.append(c[i])
            else:
                cell_counts.append(0)

        return label_list, cell_counts

    def som_mapping(self, x_n, y_n, d, sigma, lr,
                    neighborhood='gaussian',
                    seed=10):
        """
        Perform SOM on transform data

        Parameters
        ----------
        x_n : int
              the dimension of expected map
        y_n : int
              the dimension of expected map
        d : int
            vector length of input df
        sigma : float
               the standard deviation of initialized weights
        lr : float
            learning rate
        batch_size : int
                     iteration times
        neighborhood : string
                       e.g. 'gaussian', the initialized weights' distribution
        tf_str : string
                 tranform parameters, go check self.tf()
                 e.g. None, 'hlog' - the transform algorithm
        if_fcs : bool
                 tranform parameters, go check self.tf()
                 whethe the imput file is fcs file. If not, it should be a csv file
                 only the fcs file could be transformed
                 if it is a csv file, you have to make your own transform function
        seed : int
               for reproducing
        """

        som = MiniSom(x_n, y_n, d, sigma, lr, neighborhood_function=neighborhood, random_seed=seed) # initialize the map
        som.pca_weights_init(self.data) # initialize the weights
        print("Training...")
        som.train(self.data, 1000)  # random training
        print("\n...ready!")
        self.model = som

