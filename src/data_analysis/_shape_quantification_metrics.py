import cv2 as cv
import numpy as np


def solidity(cnt, **kwargs):
    """
    Take proportion of pixels inside vessel mask to the convex hull in order to generate Asymmetry score
    """

    img_shape = kwargs.get("img_shape")

    mask = np.zeros((img_shape[0], img_shape[1], 1), np.uint8)
    cv.drawContours(mask, [cnt], -1, 1, cv.FILLED)

    point_hull = cv.convexHull(cnt)
    hull_mask = np.zeros((img_shape[0], img_shape[1], 1), np.uint8)

    cv.drawContours(hull_mask, [point_hull], -1, 1, cv.FILLED)

    hull_non_zero_pixels = cv.countNonZero(hull_mask)
    mask_non_zero_pixels = cv.countNonZero(mask)

    metric = (float(mask_non_zero_pixels) / float(hull_non_zero_pixels))

    return metric


def circularity(cnt, **kwargs):
    """
    Measure circularity of contour
    """
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)

    metric = ((4.0 * np.pi * area) / np.power(perimeter, 2))

    return metric


def convex_hull_area(cnt, **kwargs):
    """
    Get area of convex hull
    """

    point_hull = cv.convexHull(cnt)
    metric = cv.contourArea(point_hull)

    return metric


def contour_area(cnt, **kwargs):
    """
    Get area of contour
    """

    return cv.contourArea(cnt)


def eccentricity(cnt, **kwargs):
    """
    Get eccentricity of contour
    """

    ellipse = cv.fitEllipse(cnt)

    return ellipse[1][1] / ellipse[1][0]


def roughness(cnt):
    """
    Get roughness of contour
    """
    perimeter = cv.arcLength(cnt, True)
    point_hull = cv.convexHull(cnt)
    hull_perimeter = cv.arcLength(point_hull, True)

    return perimeter/hull_perimeter
