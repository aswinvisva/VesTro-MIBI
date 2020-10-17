import pickle
from collections import Counter

import matplotlib
import numpy as np
import seaborn as sns

from dnn_vessel_heterogeneity.models.consensus_clustering import ConsensusCluster
from sklearn.cluster import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from minisom import MiniSom


class ClusteringFlowSOM:

    def __init__(self,
                 data,
                 point_name,
                 x_labels,
                 clusters=10,
                 explore_clusters=0,
                 pretrained=False,
                 show_plots=False,
                 x_n=30,
                 y_n=30,
                 d=34,
                 save=True):
        '''
        FlowSOM algorithm for clustering marker distributions

        :param data: Marker data for each cell
        :param point_name: The name of the point being used
        :param x_labels: Marker labels for plots
        :param clusters: Number of clusters (Cells) to be used
        :param pretrained: Is the model pretrained
        :param show_plots: Show each of the plots?
        '''

        assert explore_clusters < clusters, "Exploration must be less than number of clusters"

        self.data = data
        self.clusters = clusters
        self.explore_clusters = explore_clusters
        self.pretrained = pretrained
        self.model = None
        self.x_labels = x_labels
        self.point_name = point_name
        self.show_plots = show_plots
        self.x_n = x_n
        self.y_n = y_n
        self.d = d
        self.save = save

    def fit_model(self):
        '''
        Fit model and save if not pretrained

        :return: None
        '''

        if not self.pretrained:
            self.som_mapping(self.x_n, self.y_n, self.d, sigma=2.5, lr=0.1)

            if self.save:
                # saving the som in the file som.p
                with open('models/som.p', 'wb') as outfile:
                    pickle.dump(self.model, outfile)
        else:
            with open('models/som.p', 'rb') as infile:
                self.model = pickle.load(infile)

            with open('models/som_clustering.p', 'rb') as infile:
                self.cluster = pickle.load(infile)

            self.flatten_weights = self.model.get_weights().reshape(self.x_n * self.y_n, self.d)

    def predict_data(self, data):
        # get the prediction of each weight vector on meta clusters (on bestK)
        flatten_class = self.cluster.fit_predict(self.flatten_weights)

        map_class = flatten_class.reshape(self.x_n, self.y_n)

        label_list = []
        for i in range(len(data)):
            # print the milestone
            if i % 10000 == 0:
                print('%d samples done...' % i)

            xx = data[i, :]  # fetch the sample data
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

        if self.show_plots:
            # get discrete colormap
            cmap = plt.get_cmap('RdBu', np.max(map_class) - np.min(map_class) + 1)
            # set limits .5 outside true range
            mat = plt.matshow(map_class, cmap=cmap, vmin=np.min(map_class) - .5, vmax=np.max(map_class) + .5)
            # tell the colorbar to tick at integers
            plt.colorbar(mat, ticks=np.arange(np.min(map_class), np.max(map_class) + 1))
            plt.show()

        return label_list, cell_counts

    def predict(self):

        # get the prediction of each weight vector on meta clusters (on bestK)
        flatten_class = self.cluster.fit_predict(self.flatten_weights)

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

        df = pd.DataFrame(self.data, columns=self.x_labels)
        df['metacluster'] = label_list

        mmm = df.groupby(['metacluster']).mean()

        if self.show_plots:
            norm = matplotlib.colors.Normalize(-1, 1)
            colors = [[norm(-1.0), "midnightblue"],
                      [norm(-0.5), "seagreen"],
                      [norm(0.5), "mediumspringgreen"],
                      [norm(1.0), "yellow"]]

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

            ax = sns.clustermap(mmm, linewidth=0.5, xticklabels=self.x_labels, cmap=cmap)
            # plt.savefig(os.path.join('results', 'quantile_scale_85', 'heatmap.png'))
            plt.show()

        for i in range(len(mmm.values)):
            figure(num=None, figsize=(25, 12), facecolor='w', edgecolor='k')
            plt.bar(range(len(self.x_labels)), mmm.to_numpy()[i])
            plt.xticks(range(len(self.x_labels)), self.x_labels, rotation='vertical')
            plt.title("Metacluster %s" % str(i))
            plt.xlabel('Markers')
            plt.ylabel('Mean Normalized Expression')
            # plt.savefig(os.path.join('results', 'quantile_scale_85', 'metacluster_%s_histogram.png' % str(i)))
            plt.draw()
            # plt.pause(0.001)
            plt.close()

        c = Counter(label_list)

        cell_counts = []

        for i in range(self.clusters):
            if i in c.keys():
                cell_counts.append(c[i])
            else:
                cell_counts.append(0)

        if self.show_plots:
            # get discrete colormap
            cmap = plt.get_cmap('RdBu', np.max(map_class) - np.min(map_class) + 1)
            # set limits .5 outside true range
            mat = plt.matshow(map_class, cmap=cmap, vmin=np.min(map_class) - .5, vmax=np.max(map_class) + .5)
            # tell the colorbar to tick at integers
            plt.colorbar(mat, ticks=np.arange(np.min(map_class), np.max(map_class) + 1))
            plt.show()

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
        neighborhood : string
                       e.g. 'gaussian', the initialized weights' distribution
        seed : int
               for reproducing
        """
        som = MiniSom(x_n, y_n, d, sigma, lr, neighborhood_function=neighborhood, random_seed=seed)  # initialize the map
        som.pca_weights_init(self.data)  # initialize the weights
        print("Training...")
        som.train(self.data, 10000)  # random training
        print("\n...ready!")
        self.model = som

        flatten_weights = self.model.get_weights().reshape(self.x_n * self.y_n, self.d)

        if not self.pretrained:
            # initialize cluster
            cluster_ = ConsensusCluster(AgglomerativeClustering,
                                        self.clusters - self.explore_clusters, self.clusters + self.explore_clusters, 3)

            k = cluster_.get_optimal_number_of_clusters(flatten_weights, verbose=True)
            # fitting SOM weights into clustering algorithm
            self.cluster = cluster_.cluster_(n_clusters=k).fit(flatten_weights)

            if self.save:
                pickle.dump(self.cluster, open("models/som_clustering.p", "wb"))

        else:
            with open('models/som_clustering.p', 'rb') as infile:
                self.cluster = pickle.load(infile)

        self.flatten_weights = flatten_weights
