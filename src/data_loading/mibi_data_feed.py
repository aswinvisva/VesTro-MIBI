import math
import os
from typing import Union
import json

from config.config_settings import Config


class MIBIDataFeed:
    def __init__(self,
                 json_file_path: str = None,
                 feed_data_loc: Union[str, list] = "/data_dir/data",
                 feed_mask_loc: Union[str, list] = "/data_dir/masks",
                 feed_name: str = "MIBI Feed",
                 n_points: int = 48,
                 points_per_dir: int = 48,
                 brain_region_point_ranges: list = [(1, 16), (17, 32), (33, 48)],
                 brain_region_names: list = ["MFG", "HIP", "CAUD"],
                 point_dir_name: str = "Point",
                 tif_dir_name: str = "TIFs",
                 segmentation_mask_type: str = "allvessels",
                 segmentation_mask_size: tuple = (1024, 1024),
                 data_resolution_size: tuple = (500, 500),
                 data_resolution_units: str = "μm",
                 config: Config = None):
        """
        MIBI Data Feed

        :param segmentation_mask_type: str, Segmentation Mask Type
        :param feed_data_loc: Union[str, list], Either string or list of strings - paths to data directories
        :param feed_mask_loc: Union[str, list], Either string or list of strings - paths to mask directories
        :param feed_name: str, Name of data feed
        :param n_points: int, Total number of points in data feed
        :param points_per_dir: int, Number of points per directory in data feed
        """

        if json_file_path is not None:
            self._parse_json(json_file_path)
        else:
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
            self.segmentation_mask_size = segmentation_mask_size
            self.data_resolution_size = data_resolution_size
            self.data_resolution_units = data_resolution_units
            self.markers_to_ignore = []
            self.all_masks = []
            self.n_markers = 34

        self.pixels_to_distance = float(self.data_resolution_size[0]) / float(self.segmentation_mask_size[0])

        if config is not None:
            self.distance_interval = config.pixel_interval * self.pixels_to_distance
        else:
            self.distance_interval = None

    def _parse_json(self, json_file_path):
        """
        Initialize data feed with input from JSON file
        """

        with open(json_file_path) as json_file:
            feed_dict = json.load(json_file)

            self.data_loc = feed_dict["feed_data_loc"]
            self.mask_loc = feed_dict["feed_mask_loc"]
            self.segmentation_mask_type = feed_dict["segmentation_mask_type"]
            self.name = feed_dict["feed_name"]
            self.is_multi_dir = isinstance(feed_dict["feed_data_loc"], list) and isinstance(feed_dict["feed_mask_loc"],
                                                                                            list)

            self.total_points = feed_dict["n_points"]
            self.brain_region_point_ranges = [feed_dict["brain_region_point_ranges"][0][key] for key in
                                              feed_dict["brain_region_point_ranges"][0].keys()]
            self.brain_region_names = feed_dict["brain_region_names"]
            self.points_per_dir = feed_dict["points_per_dir"]
            self.points_dir_name = feed_dict["point_dir_name"]
            self.tifs_dir_name = feed_dict["tif_dir_name"]
            self.segmentation_mask_size = tuple(feed_dict["segmentation_mask_size"])
            self.data_resolution_size = tuple(feed_dict["data_resolution_size"])
            self.data_resolution_units = feed_dict["data_resolution_units"]

            self.markers_to_ignore = feed_dict["markers_to_ignore"]
            self.all_masks = feed_dict["all_masks"]
            self.n_markers = feed_dict["n_markers"]

    def display(self):
        """Display Configurations."""

        print("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()

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
