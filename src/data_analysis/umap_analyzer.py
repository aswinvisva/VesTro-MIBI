import datetime
from abc import ABC
from collections import Counter
from multiprocessing import Pool

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

        parent_dir = "%s/UMAP" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        marker_features = self.all_samples_features.drop(self.config.non_marker_vars, axis=1, errors='ignore')

        marker_features.reset_index(level=['Vessel', 'Point'], inplace=True)

        average_marker_features = marker_features.groupby(['Vessel', 'Point']).mean()

        print(average_marker_features)

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(average_marker_features)

        clusterer = DBSCAN(min_samples=10)
        cluster_labels = clusterer.fit_predict(embedding)

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in cluster_labels])

        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the MIBI Dataset', fontsize=18)
        plt.savefig(parent_dir + '/umap.png', bbox_inches='tight')

        print(average_marker_features.index)

        self.all_samples_features.reset_index(level=['Expansion'], inplace=True)
        self.all_samples_features.loc[average_marker_features.index, "UMAP0"] = embedding[:, 0]
        self.all_samples_features.loc[average_marker_features.index, "UMAP1"] = embedding[:, 1]
        self.all_samples_features.loc[average_marker_features.index, "DBSCAN Cluster"] = cluster_labels
        self.all_samples_features.set_index(['Point', 'Vessel', 'Expansion', 'Data Type'])

        print(self.all_samples_features)
