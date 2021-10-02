import datetime
from abc import ABC
from collections import Counter
from multiprocessing import Pool

from scipy.stats import f_oneway

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
from src.utils.iterators import feed_features_iterator


class ANOVAMarkerExpressionAnalyzer(BaseAnalyzer, ABC):
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

        super(ANOVAMarkerExpressionAnalyzer, self).__init__(config,
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
        ANOVA Brain Region Marker Expression Mean Analysis
        :return:
        """

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

            anova_test_dict = {
                "Marker": [],
                "F-Statistic": [],
                "P-Value": []
            }

            for i, marker in enumerate(self.markers_names):

                region_marker_features = []

                for region in feed_features['Region'].unique():
                    region_features = feed_features[feed_features['Region'] == region]

                    reg_features = loc_by_expansion(region_features,
                                                    columns_to_keep=self.markers_names,
                                                    expansion_type="mask_and_expansion",
                                                    average=True)

                    x = reg_features[marker].values

                    region_marker_features.append(x)

                f_statistic, p_value = f_oneway(*region_marker_features)

                anova_test_dict["Marker"].append(marker)
                anova_test_dict["F-Statistic"].append(f_statistic)
                anova_test_dict["P-Value"].append(p_value)

            t_test_df = pd.DataFrame.from_dict(anova_test_dict)
            t_test_df.to_csv(os.path.join(feed_dir, "anova_test.csv"))
