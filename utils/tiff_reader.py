import os

import numpy as np
from tifffile.tifffile import TiffFile
import cv2 as cv


def read(path, describe=False):
    with TiffFile(path) as tif:
        img = tif.asarray()

        if describe:
            print(tif)
            print("Pages:", len(tif.pages))
            cv.imshow("Marker", img)
            cv.waitKey(0)

    return img
