import datetime
import random
from collections import Counter
import os
import logging

import cv2 as cv


def mkdir_p(mypath: str):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    :param str, mypath: Path to create
    """

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def get_contour_areas_list(contours: list) -> list:
    """
    Get list of contour areas given contours

    :param contours: list, [n_contours] -> Contours to get areas from
    :return:
    """

    return [cv.contourArea(cnt) for cnt in contours]
