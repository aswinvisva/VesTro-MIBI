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


def dilate_mask(mask, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv.dilate(mask, kernel, iterations=1)
    return dilated


def expand_vessel_region(cnt, scale=1.15):
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def calculate_microenvironment_marker_expression_single_vessel(markers_data, contours,
                                                               scaling_factor=15,
                                                               expansion_coefficient=1.1,
                                                               prev_expansion_coefficient=1,
                                                               expression_type="area_normalized_counts",
                                                               transformation="arcsinh",
                                                               normalization="percentile",
                                                               plot=True):
    '''
    Get normalized expression of markers in given cells

    :param prev_expansion_coefficient:
    :param expansion_coefficient:
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
    expression_images = []

    for idx, cnt in enumerate(contours):
        data_vec = []
        expression_image = []

        for marker in markers_data:
            x, y, w, h = cv.boundingRect(cnt)

            expanded_cnt = expand_vessel_region(cnt, scale=expansion_coefficient)

            if prev_expansion_coefficient != 1:
                cnt = expand_vessel_region(cnt, scale=prev_expansion_coefficient)

            mask = np.zeros(marker.shape, np.uint8)

            cv.drawContours(mask, [cnt], -1, (255, 255, 255), cv.FILLED)

            mask_expanded = np.zeros(marker.shape, np.uint8)

            cv.drawContours(mask_expanded, [expanded_cnt], -1, (255, 255, 255), cv.FILLED)

            result_mask = mask_expanded - mask

            result = cv.bitwise_and(marker, result_mask)

            roi_result = result[y:y + h, x:x + w]

            expression_image.append(roi_result)

            if plot:
                cv.imshow("mask", mask)
                cv.imshow("mask_expanded", mask_expanded)
                cv.imshow("result", result*255)
                cv.imshow("roi_result", roi_result*255)
                cv.waitKey(0)

            if expression_type == "mean":
                # Get mean intensity of marker
                marker_data = cv.mean(result)

            elif expression_type == "area_normalized_counts":
                # Get cell area normalized count of marker
                marker_data = cv.countNonZero(result) / cv.contourArea(cnt)

            elif expression_type == "counts":
                # Get cell area normalized count of marker
                marker_data = cv.countNonZero(result)

            data_vec.append(marker_data)

        contour_mean_values.append(np.array(data_vec))
        expression_images.append(expression_image)

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

    expression_images = np.array(expression_images)

    return contour_mean_values, expression_images


def calculate_marker_composition_single_vessel(markers_data, contours,
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

                mask = np.zeros(marker.shape, np.uint8)
                cv.drawContours(mask, [cnt], -1, (255, 255, 255), 1)
                marker_data = cv.mean(marker, mask=mask)

            elif expression_type == "area_normalized_counts":

                mask = np.zeros(marker.shape, np.uint8)
                cv.drawContours(mask, [cnt], -1, (255, 255, 255), cv.FILLED)
                result = cv.bitwise_and(marker, mask)

                # Get cell area normalized count of marker
                marker_data = cv.countNonZero(result) / cv.contourArea(cnt)

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
