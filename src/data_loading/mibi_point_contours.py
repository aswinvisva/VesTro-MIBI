import numpy as np

from config.config_settings import Config
from src.data_preprocessing.object_extractor import ObjectExtractor
from src.utils.utils_functions import get_contour_areas_list


class MIBIPointContours:

    def __init__(self,
                 segmentation_mask: np.array,
                 point_idx: int,
                 config: Config,
                 object_extractor: ObjectExtractor):
        """
        Data structure for storing MIBI contours

        :param segmentation_mask: array_like, Segmentation mask
        :param point_idx: int, Point index
        """
        self.config = config
        self.object_extractor = object_extractor

        self._rois, self._contours, self._removed_contours = self.object_extractor.extract(
            segmentation_mask,
            point_name=str(
                point_idx + 1))

    @property
    def contours(self):
        """
        Get contours list
        :return: list, Contours
        """
        return self._contours

    @property
    def areas(self):
        """
        Get areas list
        :return: list, Areas of contours
        """
        return get_contour_areas_list(self._contours)

    @property
    def rois(self):
        """
        Get ROI's list
        :return: list, ROI's of contours
        """
        return self._rois

    @property
    def removed_contours(self):
        """
        Get removed contours
        :return: list, Contours which were filtered out
        """
        return self._removed_contours

    @property
    def removed_areas(self):
        """
        Get areas list
        :return: list, Areas of contours which were filtered out
        """
        return get_contour_areas_list(self._removed_contours)
