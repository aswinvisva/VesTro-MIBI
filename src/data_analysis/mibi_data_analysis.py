import datetime
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm
import cv2 as cv

from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config


class MIBIAnalyzer:
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

        self.config = config
        self.all_samples_features = all_samples_features
        self.markers_names = markers_names
        self.all_feeds_contour_data = all_feeds_contour_data
        self.all_feeds_metadata = all_feeds_metadata
        self.all_feeds_data = all_points_marker_data

    def vessel_contiguity_analysis(self, area_threshold=50, defect_distance_threshold=3, n_defect_points_threshold=3):
        """
        Vessel Contiguity Analysis
        :return:
        """

        img_shape = self.config.segmentation_mask_size
        parent_dir = "%s/Vessel Breakage" % self.config.visualization_results_dir

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
            feed_data = self.all_feeds_contour_data.loc[feed_idx]
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            breakage_dir = "%s/%s" % (output_dir, "Breakage")
            mkdir_p(breakage_dir)

            non_breakage_dir = "%s/%s" % (output_dir, "Non-Breakage")
            mkdir_p(non_breakage_dir)

            for point_idx in feed_data.index.get_level_values('Point Index').unique():
                point_data = feed_data.loc[point_idx, "Contours"]

                for cnt_idx, cnt in enumerate(point_data.contours):

                    self.all_samples_features.loc[idx[point_idx + 1,
                                                  cnt_idx,
                                                  :,
                                                  :], "Breakage"] = False

                    if cv.contourArea(cnt) > area_threshold:
                        hull = cv.convexHull(cnt, returnPoints=False)
                        point_hull = cv.convexHull(cnt)

                        defects = cv.convexityDefects(cnt, hull)

                        if defects is not None:
                            defect_distances = [d[0][3] / 256.0 for d in defects]
                            furthest_points = [d[0][2] for d in defects]
                            furthest_points = [cnt[i] for i in furthest_points]
                            mask = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)

                            cv.drawContours(mask, [cnt], -1, (255, 255, 255), cv.FILLED)
                            cv.drawContours(mask, [point_hull], -1, (0, 255, 0), 3)

                            n_defective = 0

                            for i, point in enumerate(furthest_points):
                                if defect_distances[i] > defect_distance_threshold:
                                    x = point[0][0]
                                    y = point[0][1]

                                    cv.circle(mask, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

                                    n_defective += 1

                            breakage = n_defective > n_defect_points_threshold

                            self.all_samples_features.loc[idx[point_idx + 1,
                                                          cnt_idx,
                                                          :,
                                                          :], "Breakage"] = breakage

                            if breakage:
                                cv.imwrite(os.path.join(breakage_dir,
                                                        "Point_Num_%s_Vessel_ID_%s.png" % (str(point_idx + 1),
                                                                                           str(cnt_idx))),
                                           mask)
                            else:
                                cv.imwrite(os.path.join(non_breakage_dir,
                                                        "Point_Num_%s_Vessel_ID_%s.png" % (str(point_idx + 1),
                                                                                           str(cnt_idx))),
                                           mask)
