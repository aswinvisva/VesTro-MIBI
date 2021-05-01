import datetime
from abc import ABC
from collections import Counter
from multiprocessing import Pool

from src.data_analysis._shape_quantification_metrics import *
from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config
from src.data_preprocessing.markers_feature_gen import arcsinh


class ShapeQuantificationAnalyzer(BaseAnalyzer, ABC):
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

        super(ShapeQuantificationAnalyzer, self).__init__(config,
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
        Vessel Contiguity Analysis
        :return:
        """

        vessel_mask_marker_threshold = kwargs.get("vessel_mask_marker_threshold", 0.25)

        shape_quantification_method = kwargs.get("shape_quantification_method", {
            "Name": "Circularity",
            "Metric": circularity
        })

        img_shape = kwargs.get("img_shape", self.config.segmentation_mask_size)

        shape_quantification_name = shape_quantification_method["Name"]
        shape_quantification_metric = shape_quantification_method["Metric"]

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
            feed_data = self.all_feeds_contour_data.loc[feed_idx]
            idx = pd.IndexSlice

            for point_idx in feed_data.index.get_level_values('Point Index').unique():
                point_data = feed_data.loc[point_idx, "Contours"]

                for cnt_idx, cnt in enumerate(point_data.contours):

                    self.all_samples_features.loc[idx[point_idx + 1,
                                                  cnt_idx,
                                                  :,
                                                  :], shape_quantification_name] = "NA"

                    self.all_samples_features.loc[idx[point_idx + 1,
                                                  cnt_idx,
                                                  :,
                                                  :], "%s Score" % shape_quantification_name] = float("NaN")

                    cnt_area = cv.contourArea(cnt)

                    if cnt_area > self.config.small_vessel_threshold:
                        asymmetry_score = shape_quantification_metric(cnt, img_shape=img_shape)

                        self.all_samples_features.loc[idx[point_idx + 1,
                                                      cnt_idx,
                                                      :,
                                                      :], "%s Score" % shape_quantification_name] = asymmetry_score

            self.all_samples_features["%s Score" % shape_quantification_name] = \
                self.all_samples_features[
                    "%s Score" % shape_quantification_name] / np.percentile(
                    self.all_samples_features["%s Score" % shape_quantification_name],
                    self.config.percentile_to_normalize,
                    axis=0)

            self.all_samples_features[shape_quantification_name] = "No"

            quartile_25 = np.percentile(self.all_samples_features["%s Score" % shape_quantification_name], 25)
            quartile_50 = np.percentile(self.all_samples_features["%s Score" % shape_quantification_name], 50)
            quartile_75 = np.percentile(self.all_samples_features["%s Score" % shape_quantification_name], 75)

            for point_idx in feed_data.index.get_level_values('Point Index').unique():
                point_data = feed_data.loc[point_idx, "Contours"]

                for cnt_idx, cnt in enumerate(point_data.contours):
                    vessel_features = self.all_samples_features.loc[idx[point_idx + 1,
                                                                    cnt_idx,
                                                                    :,
                                                                    :], :]

                    if (vessel_features["GLUT1"] >= vessel_mask_marker_threshold).any() or (
                            vessel_features["vWF"] >= vessel_mask_marker_threshold).any() or (
                            vessel_features["CD31"] >= vessel_mask_marker_threshold).any() or (
                            vessel_features["SMA"] >= vessel_mask_marker_threshold).any():

                        if (vessel_features["%s Score" % shape_quantification_name] >= quartile_75).any():
                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], shape_quantification_name] = "100%"
                        elif (vessel_features["%s Score" % shape_quantification_name] >= quartile_50).any() and (
                                vessel_features["%s Score" % shape_quantification_name] < quartile_75).any():
                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], shape_quantification_name] = "75%"
                        elif (vessel_features["%s Score" % shape_quantification_name] >= quartile_25).any() and (
                                vessel_features["%s Score" % shape_quantification_name] < quartile_50).any():
                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], shape_quantification_name] = "50%"
                        elif (vessel_features["%s Score" % shape_quantification_name] <= quartile_25).any():
                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], shape_quantification_name] = "25%"
