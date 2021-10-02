from abc import ABC

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib
import seaborn as sns

from src.data_analysis._shape_quantification_metrics import *
from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config
from src.data_preprocessing.markers_feature_gen import arcsinh
from src.data_preprocessing.transforms import melt_markers, loc_by_expansion
from src.utils.utils_functions import mkdir_p, round_to_nearest_half, save_fig_or_show
from src.utils.iterators import feed_features_iterator


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1


class LinearRegressionAnalyzer(BaseAnalyzer, ABC):
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

        super(LinearRegressionAnalyzer, self).__init__(config,
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

    def analyze(self, results_dir, **kwargs):
        """
        Linear Regression Analysis
        :return:
        """

        save_fig = kwargs.get("save_fig", True)

        n_markers = len(self.markers_names)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=results_dir):

            feed_features.reset_index(level=['Point', 'Vessel', 'Expansion', 'Expansion Type'], inplace=True)

            bins = [brain_region_point_ranges[i][0] - 1 for i in range(len(brain_region_point_ranges))]
            bins.append(float('Inf'))

            feed_features['Region'] = pd.cut(feed_features['Point'],
                                             bins=bins,
                                             labels=brain_region_names)

            feed_features.set_index(['Point', 'Vessel', 'Expansion', 'Expansion Type'], inplace=True)

            for region in feed_features['Region'].unique():
                region_dir = "%s/%s" % (feed_dir, region)
                mkdir_p(region_dir)

                region_features = feed_features[feed_features['Region'] == region]

                reg_features = loc_by_expansion(region_features,
                                                expansion_type="mask_and_expansion",
                                                average=False)

                small_features = reg_features[reg_features["Contour Area"].isin(["25%", "50%"]).to_numpy()]
                large_features = reg_features[reg_features["Contour Area"].isin(["75%", "100%"]).to_numpy()]

                heatmap_data = np.zeros((n_markers, n_markers))

                for i, marker_a in enumerate(self.markers_names):
                    for j, marker_b in enumerate(self.markers_names):
                        if i <= j:
                            x = large_features[marker_a].values
                            y = large_features[marker_b].values
                        else:
                            x = small_features[marker_a].values
                            y = small_features[marker_b].values

                        x2 = sm.add_constant(x)

                        est = sm.OLS(y, x2)
                        est2 = est.fit()

                        coef = est2.params[1]

                        heatmap_data[i, j] = est2.rsquared * sign(coef)

                norm = matplotlib.colors.Normalize(-1, 1)
                colors = [[norm(-1.0), "darkblue"],
                          [norm(-0.5), "blue"],
                          [norm(0), "white"],
                          [norm(0.5), "red"],
                          [norm(1.0), "darkred"]]

                cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

                plt.figure(figsize=(22, 10))

                ax = sns.heatmap(heatmap_data,
                                 cmap=cmap,
                                 xticklabels=self.markers_names,
                                 yticklabels=self.markers_names,
                                 linewidths=0,
                                 vmin=-1.0,
                                 vmax=1.0,
                                 )

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=region_dir + '/marker_correlation_heatmap.png')
