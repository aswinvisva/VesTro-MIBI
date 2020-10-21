import os

import numpy as np
from tifffile.tifffile import TiffFile
import cv2 as cv


def read(path: str, describe: bool = False) -> np.ndarray:
    """
    Read a Tiff file and collect a numpy array

    :param path: str, Path to file
    :param describe: bool, Provide a description of data
    :return: array_like, [point_size[0], point_size[1]] -> MIBI tiff data
    """

    with TiffFile(path) as tif:
        img = tif.asarray()

        if describe:
            print(tif)
            print("Pages:", len(tif.pages))

            cv.imshow("Marker", (img / max(img.flatten())) * 255)
            cv.waitKey(0)

    return img
