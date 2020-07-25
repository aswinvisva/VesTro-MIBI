import math
from collections import Counter

import numpy as np
import cv2 as cv
from scipy.special import softmax
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer
import scipy.stats as stats
import matplotlib.pyplot as plt

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def arcsinh(data, cofactor=5):
    """Inverse hyperbolic sine transform

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cofactor : float or None, optional (default: 5)
        Factor by which to divide data before arcsinh transform
    """
    if cofactor <= 0:
        raise ValueError("Expected cofactor > 0 or None. " "Got {}".format(cofactor))
    if cofactor is not None:
        data = data / cofactor
    return np.arcsinh(data)


def sigmoid(x):
    e = np.exp(1)
    y = 1 / (1 + e ** (-x))
    return y


def calculate_protein_expression_single_cell(markers_data, contours,
                                             scaling_factor=15,
                                             expression_type="area_normalized_counts",
                                             transformation="log",
                                             normalization="percentile",
                                             plot=True):
    '''
    Get normalized expression of markers in given cells

    :param plot:
    :param scaling_factor: Scaling factor by which to scale the data
    :param normalization: Method to scale data
    :param transformation: Transformation of expression vector
    :param expression_type: Method of determining protein expression
    :param markers_data: Pixel data for each marker
    :param contours: Contours of cells in image
    :return:
    '''

    contour_mean_values = []

    for idx, cnt in enumerate(contours):
        data_vec = []

        for marker in markers_data:
            x, y, w, h = cv.boundingRect(cnt)

            if expression_type == "mean":
                # Get mean intensity of marker
                marker_data = cv.mean(marker[y:y + h, x:x + w])[0]
            elif expression_type == "area_normalized_counts":
                # Get cell area normalized count of marker
                marker_data = np.sum(marker[y:y + h, x:x + w]) / cv.contourArea(cnt)

            data_vec.append(marker_data)

        contour_mean_values.append(np.array(data_vec))

    if scaling_factor > 0:
        contour_mean_values = np.array(contour_mean_values) * scaling_factor

    if transformation == "quantiletransform":
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0,
                                                                 n_quantiles=100)
        contour_mean_values = quantile_transformer.fit_transform(np.array(contour_mean_values))
        # contour_mean_values += abs(np.min(contour_mean_values))
    elif transformation == "boxcox":
        contour_mean_values, _ = boxcox(contour_mean_values)
    elif transformation == "sqrt":
        contour_mean_values = np.sqrt(contour_mean_values)
    elif transformation == "log":
        contour_mean_values = np.log(contour_mean_values + 1)
    elif transformation == "arcsinh":
        # Apply arcsinh transformation
        contour_mean_values = arcsinh(np.array(contour_mean_values))

    if normalization == "percentile":
        # Scale data by the 99th percentile
        contour_mean_values = np.array(contour_mean_values) / np.percentile(np.array(contour_mean_values), 99)
    elif normalization == "normalizer":
        # Scale data by Sklearn normalizer
        sc = Normalizer().fit(np.array(contour_mean_values))
        contour_mean_values = sc.transform(contour_mean_values)

    flat_list = sorted([item for sublist in contour_mean_values for item in sublist])

    if plot:
        plt.plot(np.array(flat_list), stats.norm.pdf(np.array(flat_list)))
        plt.xlabel('Area Normalized Marker Expression')
        plt.ylabel('Probability')
        plt.title('PDF of Marker Expression')
        plt.show()

    return contour_mean_values
