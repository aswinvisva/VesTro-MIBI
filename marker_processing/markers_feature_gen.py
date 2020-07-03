from collections import Counter

import numpy as np
import cv2 as cv
from scipy.special import softmax

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def mean_normalized_expression(markers_data, contours):
    '''
    Get mean normalized expression of markers in given cells

    :param markers_data: Pixel data for each marker
    :param contours: Contours of cells in image
    :return:
    '''

    contour_mean_values = []

    for idx, cnt in enumerate(contours):
        mean_val_vec = []

        for marker in markers_data:

            x, y, w, h = cv.boundingRect(cnt)
            mean_val = cv.mean(marker[y:y + h, x:x + w])[0]/255

            mean_val_vec.append(mean_val)

        contour_mean_values.append(np.array(mean_val_vec))

    return np.array(contour_mean_values)
