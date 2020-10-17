import json
import random
from collections import Counter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

import config.config_settings as config

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def extract(img: np.ndarray) -> (list, list):
    """
    Extract vessel contours and vessel ROI's from a segmentation mask

    :param img: np.ndarray, [point_size[0], point_size[1]] -> Segmentation mask
    :return: array_like, [n_vessels, vessel_size[0], vessel_size[1]] -> Vessel regions of interest,
    array_like, [n_vessels] -> vessel contours
    """

    show = config.show_vessel_contours_when_extracting
    min_contour_area = config.minimum_contour_area_to_remove

    # If the segmentation mask is a 3-channel image, convert it to grayscale
    if img.shape[2] == 3:
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        imgray = img

    # Perform guassian blur if setting is selected
    if config.use_guassian_blur_when_extracting_vessels:
        imgray = cv.blur(imgray, config.guassian_blur)

    # Perform vessel contour extraction using OpenCV
    im2, contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    images = []
    usable_contours = []

    # Iterate through vessel contours to filter unusable ones
    for i, cnt in enumerate(contours):

        # Create a region of interest around vessel contour
        contour_area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        roi = img[y:y + h, x:x + w]

        mean = hierarchy[0, i, 3]

        # If vessel area is lower than threshold, remove it
        if contour_area < min_contour_area:
            if show:
                cv.imshow("Removed Vessel", roi)
                cv.waitKey(0)
            continue

        # Remove contours which are inside other vessels
        if mean != -1:
            if show:
                cv.imshow("Removed Vessel", roi)
                cv.waitKey(0)
            continue

        if show:
            print("(x1: %s, x2: %s, y1: %s, y2: %s), w: %s, h: %s" % (x, x + w, y, y + h, w, h))

            cv.imshow("Vessel", roi)
            cv.waitKey(0)

        images.append(roi)
        usable_contours.append(cnt)

    if show:
        copy = img.copy()
        cv.imshow("Segmented Cells", cv.drawContours(copy, usable_contours, -1, (0, 255, 0), 3))
        cv.waitKey(0)
        del copy

    return images, usable_contours
