from abc import ABC

import matplotlib
import seaborn as sns

from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_preprocessing.markers_feature_gen import *
from config.config_settings import Config
from src.data_analysis._cluster import k_means, dbscan, agglomerative, hdbscan_method
from src.data_analysis._decomposition import umap, tsne, pca, svd
from src.data_preprocessing.transforms import loc_by_expansion


class PositiveVesselSummaryAnalyzer(BaseAnalyzer, ABC):
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
        super(PositiveVesselSummaryAnalyzer, self).__init__(config,
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
        Positive vessel summary Analysis
        :return:
        """

        mask_type = kwargs.get("mask_type", "mask_only")
        marker_expression_threshold = kwargs.get("marker_expression_threshold", 0.25)

        assert mask_type in ["mask_only",
                             "mask_and_expansion",
                             "mask_and_expansion_weighted",
                             "expansion_only"], "Unknown Mask Type!"

        marker_features = loc_by_expansion(self.all_samples_features.copy(),
                                           columns_to_keep=self.markers_names,
                                           expansion_type=mask_type,
                                           average=True)

        n_vessels = marker_features.shape[0]

        vessel_proportion_dict = {"Marker": [], "% Positive Vessels": [], "% Negative Vessels": [],
                                  "# Positive Vessels": [], "# Negative Vessels": []}

        for marker in self.markers_names:
            n_positive_vessels = marker_features.loc[marker_features[marker] >= marker_expression_threshold].shape[0]
            n_negative_vessels = n_vessels - n_positive_vessels

            positive_vessels_proportion = 100.0 * (float(n_positive_vessels) / float(n_vessels))
            negative_vessels_proportion = 100.0 * (float(n_negative_vessels) / float(n_vessels))

            vessel_proportion_dict["Marker"].append(marker)
            vessel_proportion_dict["% Positive Vessels"].append(positive_vessels_proportion)
            vessel_proportion_dict["% Negative Vessels"].append(negative_vessels_proportion)
            vessel_proportion_dict["# Positive Vessels"].append(n_positive_vessels)
            vessel_proportion_dict["# Negative Vessels"].append(n_negative_vessels)

        vessel_proportion_df = pd.DataFrame.from_dict(vessel_proportion_dict)

        vessel_proportion_df.to_csv(
            os.path.join(self.config.visualization_results_dir, "vessel_positive_proportion.csv"))

        logging.info("\n" + vessel_proportion_df.to_markdown())
