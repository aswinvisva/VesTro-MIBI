import datetime
from abc import ABC
from collections import Counter
from multiprocessing import Pool
import random

import matplotlib
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import cv2 as cv
import umap
from sklearn.cluster import DBSCAN
import seaborn as sns

from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config
from src.data_analysis._cluster import k_means, dbscan, agglomerative, hdbscan_method


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
        umap_marker_settings = kwargs.get("umap_marker_settings", "vessel_mask_markers_removed")

        assert mask_type in ["mask_only",
                             "mask_and_expansion",
                             "mask_and_expansion_weighted",
                             "expansion_only"], "Unknown Mask Type!"

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
        elif mask_type == "expansion_only":
            marker_features = self.all_samples_features.loc[pd.IndexSlice[:,
                                                            :,
                                                            0:,
                                                            "Data"], :]

        marker_features = marker_features.drop(self.config.non_marker_vars, axis=1, errors='ignore')

        marker_cluster_features = marker_features.copy()
        marker_vis_features = marker_features.copy()

        marker_cluster_features.reset_index(level=['Point', 'Vessel'], inplace=True)
        marker_vis_features.reset_index(level=['Point', 'Vessel'], inplace=True)

        cluster_features = marker_cluster_features.groupby(['Point', 'Vessel']).mean()

        if umap_marker_settings == "vessel_mask_markers_removed":
            marker_vis_features = marker_vis_features.drop(self.config.mask_marker_clusters["Vessels"],
                                                           axis=1, errors='ignore')

        elif umap_marker_settings == "vessel_markers_only":
            for cluster in self.config.marker_clusters.keys():
                if cluster != "Vessels":
                    marker_vis_features = marker_vis_features.drop(self.config.marker_clusters[cluster],
                                                                   axis=1, errors='ignore')

        elif umap_marker_settings == "vessel_and_astrocyte_markers":
            for cluster in self.config.marker_clusters.keys():
                if cluster != "Vessels" and cluster != "Astrocytes":
                    marker_vis_features = marker_vis_features.drop(self.config.marker_clusters[cluster],
                                                                   axis=1, errors='ignore')

        visualization_features = marker_vis_features.groupby(['Point', 'Vessel']).mean()

        reducer = umap.UMAP(random_state=42, n_neighbors=4)
        embedding = reducer.fit_transform(visualization_features)

        clustering_models = [{"title": "K-Means", "cluster": k_means},
                             {"title": "Hierarchical", "cluster": agglomerative}]

        features = [{"title": "UMAP Space", "features": embedding},
                    {"title": "Original Space", "features": cluster_features.to_numpy()}]
        n_clusters_trials = [5, 10, 15, 20]

        for n_clusters in n_clusters_trials:

            cluster_dir = "%s/%s Clusters" % (parent_dir, str(n_clusters))
            mkdir_p(cluster_dir)

            for feature in features:
                X = feature["features"]

                for cluster_model in clustering_models:

                    y = cluster_model["cluster"](X, n_clusters=n_clusters)

                    title = "%s Clustering in %s" % (cluster_model["title"], feature["title"])

                    for idx, x in enumerate(visualization_features.index):
                        point_idx = x[0]
                        vessel_idx = x[1]

                        embedding_0 = embedding[:, 0][idx]
                        embedding_1 = embedding[:, 1][idx]
                        cluster = y[idx]

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
                                                      :], cluster_model["title"]] = cluster

                    marker_features = self.all_samples_features.drop(
                        [x for x in self.config.non_marker_vars if x != cluster_model["title"]],
                        axis=1, errors='ignore')

                    average_marker_expression = marker_features.groupby([cluster_model["title"]]).mean()

                    plt.figure(figsize=(12, 10))

                    ax = sns.scatterplot(x="UMAP0",
                                         y="UMAP1",
                                         data=self.all_samples_features,
                                         hue=cluster_model["title"],
                                         palette="tab20")

                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Cluster")

                    plt.title('UMAP projection of the MIBI Dataset', fontsize=18)
                    plt.savefig(cluster_dir + '/%s.png' % title, bbox_inches='tight')
                    plt.clf()

                    norm = matplotlib.colors.Normalize(-1, 1)
                    colors = [[norm(-1.0), "black"],
                              [norm(-0.5), "indigo"],
                              [norm(0), "firebrick"],
                              [norm(0.5), "orange"],
                              [norm(1.0), "khaki"]]

                    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

                    clrs = sns.color_palette('tab20', n_colors=n_clusters)

                    ax = sns.clustermap(average_marker_expression,
                                        cmap=cmap,
                                        linewidths=0,
                                        row_cluster=True,
                                        col_cluster=True,
                                        row_colors=clrs,
                                        figsize=(20, 10)
                                        )
                    plt.savefig(cluster_dir + '/%s_heatmap.png' % title, bbox_inches='tight')
                    plt.clf()
