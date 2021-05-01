import math
import os
from typing import Union

from config.config_settings import Config


class MIBIDataFeed:
    def __init__(self,
                 feed_data_loc: Union[str, list],
                 feed_mask_loc: Union[str, list],
                 feed_name: str,
                 n_points: int,
                 points_per_dir: int = 48,
                 brain_region_point_ranges: list = [(1, 16), (17, 32), (33, 48)],
                 brain_region_names: list = ["MFG", "HIP", "CAUD"],
                 point_dir_name: str = "Point",
                 tif_dir_name: str = "TIFs",
                 segmentation_mask_type: str = "allvessels"):
        """
        MIBI Data Feed

        :param segmentation_mask_type: str, Segmentation Mask Type
        :param feed_data_loc: Union[str, list], Either string or list of strings - paths to data directories
        :param feed_mask_loc: Union[str, list], Either string or list of strings - paths to mask directories
        :param feed_name: str, Name of data feed
        :param n_points: int, Total number of points in data feed
        :param points_per_dir: int, Number of points per directory in data feed
        """

        self.segmentation_mask_type = segmentation_mask_type
        self.name = feed_name
        self.is_multi_dir = isinstance(feed_data_loc, list) and isinstance(feed_mask_loc, list)
        self.data_loc = feed_data_loc
        self.mask_loc = feed_mask_loc
        self.total_points = n_points
        self.brain_region_point_ranges = brain_region_point_ranges
        self.brain_region_names = brain_region_names
        self.points_per_dir = points_per_dir
        self.points_dir_name = point_dir_name
        self.tifs_dir_name = tif_dir_name

    def get_locs(self, point_num: int):
        """
        Get data and mask locations for a given point

        :param point_num: int, Point number
        :return:
        """

        if self.is_multi_dir:
            data_idx = math.ceil(point_num / self.points_per_dir)
            point_num_in_dir = point_num % self.points_per_dir

            data_loc = os.path.join(self.data_loc[data_idx],
                                    self.points_dir_name + str(point_num_in_dir),
                                    self.tifs_dir_name)

            mask_loc = os.path.join(self.mask_loc[data_idx],
                                    self.points_dir_name + str(point_num),
                                    self.segmentation_mask_type + '.tif')

            return data_loc, mask_loc
        else:
            data_loc = os.path.join(self.data_loc,
                                    self.points_dir_name + str(point_num),
                                    self.tifs_dir_name)

            mask_loc = os.path.join(self.mask_loc,
                                    self.points_dir_name + str(point_num),
                                    self.segmentation_mask_type + '.tif')

            return data_loc, mask_loc
