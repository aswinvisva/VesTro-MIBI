import os
import random
import tempfile

import PIL
import cv2
import numpy as np
import skimage.io as io

from config.config_settings import Config
from src.utils.utils_functions import mkdir_p


def create_test_data(test_dir,
                     n_points=48,
                     resolution=(2048, 2048)):
    config = Config()

    mask_dir = os.path.join(test_dir, "masks")
    data_dir = os.path.join(test_dir, "data")

    mkdir_p(mask_dir)
    mkdir_p(data_dir)

    for point_idx in range(n_points):
        directory_name = "Point" + str(point_idx + 1)

        point_data_dir = os.path.join(data_dir, directory_name, "TIFs")
        point_mask_dir = os.path.join(mask_dir, directory_name)

        mkdir_p(point_data_dir)
        mkdir_p(point_mask_dir)

        for cluster in config.marker_clusters.keys():
            for marker in config.marker_clusters[cluster]:
                data = generate_random_data(resolution)
                write_tif_data(data, marker, point_data_dir)

        mask = generate_random_mask(resolution)
        write_tif_data(mask, "allvessels", point_mask_dir)


def write_tif_data(data, tif_name, path):
    """
    Write TIFF data
    """
    raw_tiff = PIL.Image.fromarray(data)
    raw_tiff.save(os.path.join(path, "%s.tif" % tif_name))


def generate_random_data(resolution,
                         random_seed=42):
    """
    Generate random data
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    return np.random.uniform(0, 1, resolution)


def generate_random_mask(resolution,
                         n_pseudo_vessels=10,
                         n_pseudo_vessel_size=(10, 50),
                         random_seed=42):
    """
    Generate random mask
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    mask = np.zeros(shape=resolution, dtype=np.uint8)

    for i in range(n_pseudo_vessels):
        width = resolution[0]
        height = resolution[0]

        rad = random.randint(n_pseudo_vessel_size[0], n_pseudo_vessel_size[1])

        x = random.randint(rad, width - rad)
        y = random.randint(rad, height - rad)

        cv2.circle(mask,
                   (x, y),
                   rad,
                   (255, 255, 255),
                   -1,
                   8)

    return mask
