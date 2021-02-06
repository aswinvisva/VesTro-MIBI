import datetime
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

    def vessel_contiguity_analysis(self, area_threshold=50, kernel_size=10):
        """
        Vessel Contiguity Analysis
        :return:
        """
        img_shape = self.config.segmentation_mask_size
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            for point_idx in feed_data.index.get_level_values('Point Index').unique():
                point_data = feed_data.loc[point_idx, "Contours"]

                for cnt in point_data.contours:
                    if cv.contourArea(cnt) > area_threshold:

                        mask = np.zeros(img_shape, np.uint8)
                        cv.drawContours(mask, [cnt], -1, (1, 1, 1), cv.FILLED)

                        processed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

                        contiguity_score = float(cv.countNonZero(mask)) / float(cv.countNonZero(processed_mask))

                        print("%s:%s - Score: %s" % (str(feed_idx), str(point_idx), str(contiguity_score)))

                        if contiguity_score < 0.8:
                            cv.imshow("mask", mask * 255)
                            cv.imshow("processed_mask", processed_mask * 255)
                            cv.waitKey(0)
