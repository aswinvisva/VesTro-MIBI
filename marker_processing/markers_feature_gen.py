import math
from collections import Counter

import numpy as np
import cv2 as cv
from scipy.special import softmax
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer
import scipy.stats as stats
import matplotlib.pyplot as plt

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def sigmoid(x):
    e = np.exp(1)
    y = 1 / (1 + e ** (-x))
    return y


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
            mean_val = cv.mean(marker[y:y + h, x:x + w])[0]

            mean_val_vec.append(mean_val)

        contour_mean_values.append(np.array(mean_val_vec))

    sc = Normalizer().fit(np.array(contour_mean_values))
    contour_mean_values = sc.transform(contour_mean_values)

    # quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    # contour_mean_values = quantile_transformer.fit_transform(np.array(contour_mean_values))

    flat_list = sorted([item for sublist in contour_mean_values for item in sublist])

    plt.plot(np.array(flat_list), stats.norm.pdf(np.array(flat_list)))
    plt.show()

    return contour_mean_values
