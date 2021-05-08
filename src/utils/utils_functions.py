import datetime
import random
from collections import Counter
import os
import logging

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def round_to_nearest_half(number):
    """
    Round float to nearest 0.5
    :param number: float, number to round
    :return: float, number rounded to nearest 0.5
    """
    return round(number * 2) / 2


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


def save_fig_or_show(save_fig: bool = True,
                     figure_path: str = None,
                     fig: plt.Figure = None):
    """
    Save figure or show figure
    """

    if save_fig:
        if fig is None:
            plt.savefig(figure_path, bbox_inches='tight')
            plt.clf()
        else:
            fig.savefig(figure_path,
                        bbox_inches='tight')
            fig.clf()
    else:
        plt.show()
