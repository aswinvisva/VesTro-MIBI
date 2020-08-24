import json
import random

import numpy as np
import cv2 as cv

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def construct_vessel_relative_area_microenvironments_from_contours(contours, img, r=0.1):
    '''
    Construct Microenvironments from Segmented Cell Events by taking an area relative to its size around each vessel

    :param contours: Spatial information of segmented cell events
    :param img: Original segmentation mask
    :param r: Ratio of area
    :return: Vector containing partitioned images
    '''

    microenvironments = []

    height, width, _ = img.shape

    area = height * width

    contours.sort(key=lambda x: cv.contourArea(x), reverse=True)

    included = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    i = 0

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        contour_area = cv.contourArea(cnt)
        ratio = contour_area / area

        ratio = max(ratio, r)

        n_d = max(int(ratio * w), int(ratio * height))

        M = cv.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if included[cy][cx] == 1:
            continue

        i += 1

        x1 = int(cx - w - n_d)
        x2 = int(cx + w + n_d)
        y1 = int(cy - h - n_d)
        y2 = int(cy + h + n_d)

        print(cx, cy, x1, x2, y1, y2, ratio)

        if x1 < 0:
            x1 = 0

        if y1 < 0:
            y1 = 0

        if x2 > width:
            x2 = width

        if y2 > height:
            y2 = height

        included[y1:y2, x1:x2] = 1
        ROI = img.copy()[y1:y2, x1:x2]
        # ROI = img.copy()
        # cv.drawContours(ROI, [cnt], -1, (0, 255, 0), 3)
        cv.imshow("ASD", ROI)
        cv.imwrite("microenvironment_%s.png" % (str(i)), ROI)

        # cv.imshow("ASD1", ROI1)

        cv.waitKey(0)
        microenvironments.append(ROI)

    cv.imwrite("original.png", img)
    cv.imshow("Original", img)
    cv.waitKey(0)

    return microenvironments


def construct_partitioned_microenvironments_from_contours(contours, img, r=256):
    '''
    Construct Microenvironments from Segmented Cell Events by partitioning the segmentation mask into images of size rxr

    :param contours: Spatial information of segmented cell events
    :param img: Original segmentation mask
    :param r: Dimensions of microenvironments
    :return: Vector containing partitioned images
    '''
    microenvironments = []

    height, width, _ = img.shape

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if w > r or h > r:
            if w > h:
                r = w
            else:
                r = h

        cx = (x + w) / 2
        cy = (y + h) / 2

        x1 = int(cx - r)
        x2 = int(cx + r)
        y1 = int(cy - r)
        y2 = int(cy + r)

        if x1 < 0:
            x1 = 0

        if y1 < 0:
            y1 = 0

        if x2 > width:
            x2 = width

        if y2 > height:
            y2 = height

        ROI = img[y1:y2, x1:x2]
        cv.imshow("ASD", ROI)
        cv.waitKey(0)
        microenvironments.append(ROI)

    return microenvironments
