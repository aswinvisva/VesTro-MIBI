import datetime
from abc import ABC
from collections import Counter
from multiprocessing import Pool

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import cv2 as cv

from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config


class VesselAsymmetryAnalyzer(BaseAnalyzer, ABC):
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

        super(VesselAsymmetryAnalyzer, self).__init__(config,
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

        asymmetry_threshold = kwargs.get("asymmetry_threshold", 0.15)

        img_shape = self.config.segmentation_mask_size
        parent_dir = "%s/Vessel Asymmetry" % self.config.visualization_results_dir

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
            feed_data = self.all_feeds_contour_data.loc[feed_idx]
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            asymmetry_dir = "%s/%s" % (output_dir, "Asymmetry")
            mkdir_p(asymmetry_dir)

            non_asymmetry_dir = "%s/%s" % (output_dir, "Non-Asymmetry")
            mkdir_p(non_asymmetry_dir)

            na_dir = "%s/%s" % (output_dir, "Excluded")
            mkdir_p(na_dir)

            for point_idx in feed_data.index.get_level_values('Point Index').unique():
                point_data = feed_data.loc[point_idx, "Contours"]

                for cnt_idx, cnt in enumerate(point_data.contours):

                    self.all_samples_features.loc[idx[point_idx + 1,
                                                  cnt_idx,
                                                  :,
                                                  :], "Asymmetry"] = "NA"

                    self.all_samples_features.loc[idx[point_idx + 1,
                                                  cnt_idx,
                                                  :,
                                                  :], "Asymmetry Score"] = float("NaN")

                    cnt_area = cv.contourArea(cnt)

                    mask = np.zeros((img_shape[0], img_shape[1], 1), np.uint8)

                    cv.drawContours(mask, [cnt], -1, 1, cv.FILLED)

                    if cnt_area > self.config.small_vessel_threshold:
                        # hull = cv.convexHull(cnt, returnPoints=False)
                        point_hull = cv.convexHull(cnt)
                        hull_mask = np.zeros((img_shape[0], img_shape[1], 1), np.uint8)

                        cv.drawContours(hull_mask, [point_hull], -1, 1, cv.FILLED)

                        hull_non_zero_pixels = cv.countNonZero(hull_mask)
                        mask_non_zero_pixels = cv.countNonZero(mask)

                        asymmetry_score = 1.0 - (float(mask_non_zero_pixels) / float(hull_non_zero_pixels))

                        self.all_samples_features.loc[idx[point_idx + 1,
                                                      cnt_idx,
                                                      :,
                                                      :], "Asymmetry Score"] = asymmetry_score

                        if cnt_area > self.config.large_vessel_threshold:
                            size = "Large"
                        elif cnt_area > self.config.medium_vessel_threshold:
                            size = "Medium"
                        elif cnt_area > self.config.small_vessel_threshold:
                            size = "Small"

                        if asymmetry_score > asymmetry_threshold:
                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], "Asymmetry"] = "Yes"

                            example_mask = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
                            cv.drawContours(example_mask, [cnt], -1, (255, 255, 255), cv.FILLED)
                            cv.drawContours(example_mask, [point_hull], -1, (0, 0, 255), 2)

                            M = cv.moments(cnt)
                            cX = int(M["m10"] / M["m00"]) + 10
                            cY = int(M["m01"] / M["m00"]) - 10

                            cv.putText(example_mask,
                                       "Proportion: %s" % round(1.0 - asymmetry_score, 2),
                                       (cX, cY),
                                       cv.FONT_HERSHEY_SIMPLEX,
                                       0.5,
                                       (0, 255, 0),
                                       2,
                                       cv.LINE_AA)

                            cv.imwrite(os.path.join(asymmetry_dir,
                                                    "Point_Num_%s_Vessel_ID_%s_%s.png" % (str(point_idx + 1),
                                                                                          str(cnt_idx + 1), size)),
                                       example_mask)
                        else:
                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], "Asymmetry"] = "No"

                            example_mask = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
                            cv.drawContours(example_mask, [cnt], -1, (255, 255, 255), cv.FILLED)
                            cv.drawContours(example_mask, [point_hull], -1, (0, 0, 255), 2)

                            M = cv.moments(cnt)
                            cX = int(M["m10"] / M["m00"]) + 5
                            cY = int(M["m01"] / M["m00"]) + 5

                            cv.putText(example_mask,
                                       "Proportion: %s" % round(1.0 - asymmetry_score, 2),
                                       (cX, cY),
                                       cv.FONT_HERSHEY_SIMPLEX,
                                       0.5,
                                       (0, 255, 0),
                                       2,
                                       cv.LINE_AA)

                            cv.imwrite(os.path.join(non_asymmetry_dir,
                                                    "Point_Num_%s_Vessel_ID_%s_%s.png" % (str(point_idx + 1),
                                                                                          str(cnt_idx + 1), size)),
                                       example_mask)

            max_value \
                = self.all_samples_features["Asymmetry Score"].max()
            min_value \
                = self.all_samples_features["Asymmetry Score"].min()

            self.all_samples_features["Asymmetry Score"] = (self.all_samples_features["Asymmetry Score"]
                                                            - min_value) / (max_value - min_value)
