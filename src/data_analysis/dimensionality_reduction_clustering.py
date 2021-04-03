from abc import ABC

import matplotlib
import seaborn as sns

from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_preprocessing.markers_feature_gen import *
from config.config_settings import Config
from src.data_analysis._cluster import k_means, dbscan, agglomerative, hdbscan_method
from src.data_analysis._decomposition import umap, tsne, pca, svd


class DimensionalityReductionClusteringAnalyzer(BaseAnalyzer, ABC):
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

        super(DimensionalityReductionClusteringAnalyzer, self).__init__(config,
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
        marker_settings = kwargs.get("marker_settings", "vessel_mask_markers_removed")

        assert mask_type in ["mask_only",
                             "mask_and_expansion",
                             "mask_and_expansion_weighted",
                             "expansion_only"], "Unknown Mask Type!"

        parent_dir = "%s/Clustering" % self.config.visualization_results_dir
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

        if marker_settings == "vessel_mask_markers_removed":
            marker_vis_features = marker_vis_features.drop(self.config.mask_marker_clusters["Vessels"],
                                                           axis=1, errors='ignore')

        elif marker_settings == "vessel_markers_only":
            for cluster in self.config.marker_clusters.keys():
                if cluster != "Vessels":
                    marker_vis_features = marker_vis_features.drop(self.config.marker_clusters[cluster],
                                                                   axis=1, errors='ignore')

        elif marker_settings == "vessel_and_astrocyte_markers":
            for cluster in self.config.marker_clusters.keys():
                if cluster != "Vessels" and cluster != "Astrocytes":
                    marker_vis_features = marker_vis_features.drop(self.config.marker_clusters[cluster],
                                                                   axis=1, errors='ignore')

        visualization_features = marker_vis_features.groupby(['Point', 'Vessel']).mean()

        umap_embedding = umap(visualization_features, n_neighbors=4)
        tsne_embedding = tsne(visualization_features)
        pca_embedding = pca(visualization_features)
        svd_embedding = svd(visualization_features)

        clustering_models = [{"title": "K-Means", "cluster": k_means},
                             {"title": "Hierarchical", "cluster": agglomerative}]

        cluster_features_trials = [{"title": "UMAP Space", "features": umap_embedding},
                                   {"title": "TSNE Space", "features": tsne_embedding},
                                   {"title": "PCA Space", "features": pca_embedding},
                                   {"title": "SVD", "features": svd_embedding},
                                   {"title": "Original Space", "features": cluster_features.to_numpy()}, ]

        vis_features = [{"title": "UMAP", "features": umap_embedding},
                        {"title": "TSNE", "features": tsne_embedding},
                        {"title": "PCA", "features": pca_embedding},
                        {"title": "SVD", "features": svd_embedding}]

        n_clusters_trials = [5, 10, 15, 20]

        for n_clusters in n_clusters_trials:

            cluster_dir = "%s/%s Clusters" % (parent_dir, str(n_clusters))
            mkdir_p(cluster_dir)

            for vis_feature in vis_features:
                vis_dir = "%s/%s Visualization" % (cluster_dir, vis_feature["title"])
                mkdir_p(vis_dir)

                X_vis = vis_feature["features"]

                for cluster_feature in cluster_features_trials:
                    cluster_feature_dir = "%s/%s" % (vis_dir, cluster_feature["title"])
                    mkdir_p(cluster_feature_dir)

                    X = cluster_feature["features"]

                    for cluster_model in clustering_models:

                        cluster_model_dir = "%s/%s Model" % (cluster_feature_dir, cluster_model["title"])
                        mkdir_p(cluster_model_dir)

                        y = cluster_model["cluster"](X, n_clusters=n_clusters)

                        for idx, x in enumerate(visualization_features.index):
                            point_idx = x[0]
                            vessel_idx = x[1]

                            embedding_0 = X_vis[:, 0][idx]
                            embedding_1 = X_vis[:, 1][idx]
                            cluster = y[idx]

                            self.all_samples_features.loc[pd.IndexSlice[point_idx,
                                                          vessel_idx,
                                                          :,
                                                          :], vis_feature["title"] + "0"] = embedding_0

                            self.all_samples_features.loc[pd.IndexSlice[point_idx,
                                                          vessel_idx,
                                                          :,
                                                          :], vis_feature["title"] + "1"] = embedding_1

                            self.all_samples_features.loc[pd.IndexSlice[point_idx,
                                                          vessel_idx,
                                                          :,
                                                          :], cluster_model["title"]] = cluster

                        marker_features = self.all_samples_features.drop(
                            [x for x in self.config.non_marker_vars if x != cluster_model["title"]],
                            axis=1, errors='ignore')

                        marker_features = marker_features.groupby(['Point', 'Vessel']).mean()

                        average_marker_expression = marker_features.groupby([cluster_model["title"]]).mean()

                        plt.figure(figsize=(12, 10))

                        ax = sns.scatterplot(x=vis_feature["title"] + "0",
                                             y=vis_feature["title"] + "1",
                                             data=self.all_samples_features,
                                             hue=cluster_model["title"],
                                             palette="tab20")

                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Cluster")

                        plt.title('%s projection of the MIBI Dataset' % vis_feature["title"], fontsize=18)
                        plt.savefig(cluster_model_dir + '/reduced_clustered.png', bbox_inches='tight')
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
                        plt.savefig(cluster_model_dir + '/heatmap.png', bbox_inches='tight')
                        plt.clf()
