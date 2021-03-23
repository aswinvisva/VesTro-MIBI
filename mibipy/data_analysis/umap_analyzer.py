import datetime
from abc import ABC
from collections import Counter
from multiprocessing import Pool
import random

import hdbscan
import matplotlib
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import cv2 as cv
import umap
from sklearn.cluster import DBSCAN
import seaborn as sns

from mibipy.data_analysis.base_analyzer import BaseAnalyzer
from mibipy.data_loading.mibi_data_feed import MIBIDataFeed
from mibipy.data_loading.mibi_loader import MIBILoader
from mibipy.data_loading.mibi_point_contours import MIBIPointContours
from mibipy.data_preprocessing.markers_feature_gen import *
from mibipy.plotting.visualizer import Visualizer
from config.config_settings import Config


def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)


def min_dist_to_exemplar(point, cluster_exemplars, data):
    dists = cdist([data[point]], data[cluster_exemplars.astype(np.int32)])
    return dists.min()


def dist_vector(point, exemplar_dict, data):
    result = {}
    for cluster in exemplar_dict:
        result[cluster] = min_dist_to_exemplar(point, exemplar_dict[cluster], data)
    return np.array(list(result.values()))


def dist_membership_vector(point, exemplar_dict, data, softmax=False):
    if softmax:
        result = np.exp(1. / dist_vector(point, exemplar_dict, data))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = 1. / dist_vector(point, exemplar_dict, data)
        result[~np.isfinite(result)] = np.finfo(np.double).max
    result /= result.sum()
    return result


class UMAPAnalyzer(BaseAnalyzer, ABC):
    """
    Class for the analysis of MIBI data
    """

    def __init__(self,
                 config: Config,
                 all_samples_features: pd.DataFrame,
                 markers_names: list,
                 all_feeds_contour_data: pd.DataFrame,
                 all_feeds_metadata: pd.DataFrame,
                 all_points_marker_data: np.array,
                 ):
        """
        Create kept vs. removed vessel expression comparison using Box Plots

        :param markers_names: array_like, [n_points, n_markers] -> Names of markers
        :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]] ->
        list of marker data for each point
        """

        super(UMAPAnalyzer, self).__init__(config,
                                           all_samples_features,
                                           markers_names,
                                           all_feeds_contour_data,
                                           all_feeds_metadata,
                                           all_points_marker_data)

        self.config = config
        self.all_samples_features = all_samples_features
        self.markers_names = markers_names
        self.all_feeds_contour_data = all_feeds_contour_data
        self.all_feeds_metadata = all_feeds_metadata
        self.all_feeds_data = all_points_marker_data

    def analyze(self, **kwargs):
        """
        Vessel Contiguity Analysis
        :return:
        """

        mask_type = kwargs.get("mask_type", "mask_only")
        min_samples = kwargs.get("min_samples", 10)
        eps = kwargs.get("eps", 0.5)

        assert mask_type in ["mask_only", "mask_and_expansion", "mask_and_expansion_weighted"], "Unknown Mask Type!"

        parent_dir = "%s/UMAP" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        if mask_type == "mask_only":
            marker_features = self.all_samples_features.loc[pd.IndexSlice[:,
                                                            :,
                                                            :-1,
                                                            "Data"], :]
        elif mask_type == "mask_and_expansion":
            marker_features = self.all_samples_features.loc[pd.IndexSlice[:,
                                                            :,
                                                            :,
                                                            "Data"], :]
        elif mask_type == "mask_and_expansion_weighted":
            marker_features = self.all_samples_features.loc[pd.IndexSlice[:,
                                                            :,
                                                            :,
                                                            "Data"], :]

        marker_features = marker_features.drop(self.config.non_marker_vars, axis=1, errors='ignore')

        marker_features.reset_index(level=['Point', 'Vessel'], inplace=True)

        average_marker_features = marker_features.groupby(['Point', 'Vessel']).mean()

        reducer = umap.UMAP(random_state=123)
        embedding = reducer.fit_transform(average_marker_features)

        clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=eps, min_cluster_size=min_samples)
        clusterer.fit(embedding)

        tree = clusterer.condensed_tree_
        exemplar_dict = {c: exemplars(c, tree) for c in tree._select_clusters()}

        cluster_labels_argmax = []

        for x in range(embedding.shape[0]):
            membership_vector = dist_membership_vector(x, exemplar_dict, embedding)
            cluster_labels_argmax.append(np.argmax(membership_vector))

        for idx, x in enumerate(average_marker_features.index):
            point_idx = x[0]
            vessel_idx = x[1]

            embedding_0 = embedding[:, 0][idx]
            embedding_1 = embedding[:, 1][idx]
            cluster = cluster_labels_argmax[idx]

            self.all_samples_features.loc[pd.IndexSlice[point_idx,
                                          vessel_idx,
                                          :,
                                          :], "UMAP0"] = embedding_0

            self.all_samples_features.loc[pd.IndexSlice[point_idx,
                                          vessel_idx,
                                          :,
                                          :], "UMAP1"] = embedding_1

            self.all_samples_features.loc[pd.IndexSlice[point_idx,
                                          vessel_idx,
                                          :,
                                          :], "HDBSCAN Cluster"] = cluster

        marker_features = self.all_samples_features.drop([x for x in self.config.non_marker_vars if x != "HDBSCAN "
                                                                                                         "Cluster"],
                                                         axis=1, errors='ignore')

        average_marker_features = marker_features.groupby(["HDBSCAN Cluster"]).mean()

        ax = sns.scatterplot(x="UMAP0",
                             y="UMAP1",
                             data=self.all_samples_features,
                             hue="HDBSCAN Cluster",
                             palette="tab20")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="DBSCAN Cluster")

        plt.title('UMAP projection of the MIBI Dataset', fontsize=18)
        plt.savefig(parent_dir + '/umap.png', bbox_inches='tight')

        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1.0), "black"],
                  [norm(-0.5), "indigo"],
                  [norm(0), "firebrick"],
                  [norm(0.5), "orange"],
                  [norm(1.0), "khaki"]]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        plt.figure(figsize=(22, 10))

        ax = sns.clustermap(average_marker_features,
                            cmap=cmap,
                            linewidths=0,
                            row_cluster=True,
                            col_cluster=True,
                            figsize=(20, 10)
                            )
        plt.savefig(parent_dir + '/hdbscan_heatmap.png', bbox_inches='tight')
        plt.clf()
