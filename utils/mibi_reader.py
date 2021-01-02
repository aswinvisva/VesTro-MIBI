import datetime
import json
import os
import random
from collections import Counter
import logging
from tqdm import tqdm

import numpy as np
import cv2 as cv
from PIL import Image

from utils import tiff_reader
import config.config_settings as config

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


def read(data_loc: str, mask_loc: str) -> (np.ndarray, np.ndarray, list):
    """
    Read the MIBI data from a single point

    :param mask_loc: str -> Directory pointing to the segmentation masks
    :param data_loc: str -> Directory pointing to the marker data
    :return: array_like, [n_markers, point_size[0], point_size[1]] -> Marker data,
    array_like, [point_size[0], point_size[1]] -> Segmentation mask
    """

    # Get marker clusters, markers to ignore etc.
    markers_to_ignore = config.markers_to_ignore
    marker_clusters = config.marker_clusters
    plot = config.show_segmentation_masks_when_reading
    plot_markers = config.describe_markers_when_reading

    marker_names = []
    marker_images = []

    for key in marker_clusters.keys():
        for marker_name in marker_clusters[key]:

            # Path to marker tif file
            path = os.path.join(data_loc, "%s.tif" % marker_name)
            img = tiff_reader.read(path, describe=plot_markers)

            if marker_name not in markers_to_ignore:
                marker_images.append(img.copy())
                marker_names.append(marker_name)

    markers_img = np.array(marker_images)

    try:
        segmentation_mask = np.array(Image.open(mask_loc).convert("RGB"))
    except FileNotFoundError:
        # If there is no segmentation mask, return a blank image
        segmentation_mask = np.zeros((config.segmentation_mask_size[0], config.segmentation_mask_size[1], 3), np.uint8)

    if plot:
        cv.imshow("Segmentation Mask", segmentation_mask)
        cv.waitKey(0)

    return segmentation_mask, markers_img, marker_names


def get_all_point_data() -> (list, list, list):
    """
    Collect all points marker data, segmentation masks and marker names

    :return: array_like, [n_points, n_markers, point_size[0], point_size[1]] -> Marker data,
    array_like, [n_points, point_size[0], point_size[1]] -> Segmentation masks,
    array_like, [n_points, n_markers] -> Names of markers
    """

    fovs = [config.point_dir + str(i + 1) for i in range(config.n_points_per_dir)]

    all_points_segmentation_masks = []
    all_points_marker_data = []
    segmentation_type = config.selected_segmentation_mask_type

    if config.caud_hip_mfg_separate_dir:

        brain_region_directories = [config.mfg_dir, config.hip_dir, config.caud_dir]

        for brain_region_directory in brain_region_directories:
            for fov in fovs:
                # Get path to data selected through configuration settings
                data_loc = os.path.join(config.data_dir,
                                        brain_region_directory,
                                        fov,
                                        config.tifs_dir)

                # Get path to mask selected through configuration settings
                mask_loc = os.path.join(config.masks_dr,
                                        brain_region_directory,
                                        fov,
                                        segmentation_type + '.tif')

                start = datetime.datetime.now()
                segmentation_mask, marker_data, marker_names = read(data_loc, mask_loc)
                end = datetime.datetime.now()

                logging.debug("Finished reading %s in %s" % (fov, str(end - start)))

                all_points_segmentation_masks.append(segmentation_mask)
                all_points_marker_data.append(marker_data)
    else:
        for fov in fovs:
            # Get path to data selected through configuration settings
            data_loc = os.path.join(config.data_dir,
                                    fov,
                                    config.tifs_dir)

            # Get path to mask selected through configuration settings
            mask_loc = os.path.join(config.masks_dr,
                                    fov,
                                    segmentation_type + '.tif')

            start = datetime.datetime.now()
            segmentation_mask, marker_data, marker_names = read(data_loc, mask_loc)
            end = datetime.datetime.now()

            logging.debug("Finished reading %s in %s" % (fov, str(end - start)))

            all_points_segmentation_masks.append(segmentation_mask)
            all_points_marker_data.append(marker_data)

    return all_points_segmentation_masks, all_points_marker_data, marker_names
