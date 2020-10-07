import datetime
import math
import os
import random
from collections import Counter

import numpy as np
import cv2 as cv
import sklearn
from scipy.special import softmax
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils.utils_functions import mkdir_p

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def get_assigned_regions(contours, img_shape):
    vessel_img = np.zeros(img_shape)
    dist_mats = []
    sub_masks = []
    regions = []

    for i in range(len(contours)):
        sub_mask = np.ones(img_shape, np.uint8)
        cv.drawContours(sub_mask, contours, i, (0, 0, 0), cv.FILLED)
        dist = cv.distanceTransform(sub_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dist_mats.append(dist)
        sub_masks.append(sub_mask)
        vessel_img[np.where(sub_mask == 0)] = i

    for i in range(len(contours)):
        region = np.ones(img_shape, dtype=bool)
        for j in range(len(contours)):
            if i != j:
                region = region & (dist_mats[i] < dist_mats[j])

        regions.append(region)

    del dist_mats
    del sub_masks

    return regions


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


def contract_vessel_region(cnt, shape, pixel_expansion=5):
    pixel_expansion = abs(pixel_expansion)
    zeros = np.zeros(shape, np.uint8)
    cv.drawContours(zeros, [cnt], -1, (255, 255, 255), cv.FILLED)

    dist = cv.distanceTransform(zeros, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    ring = cv.inRange(dist, 0, pixel_expansion)  # take all pixels at distance between 0 px and pixel_expansion px
    ring = (ring / 255).astype(np.uint8)

    return ring


def expand_vessel_region(cnt, shape, pixel_expansion=5):
    inverted = np.ones(shape, np.uint8)
    cv.drawContours(inverted, [cnt], -1, (0, 0, 0), cv.FILLED)

    dist = cv.distanceTransform(inverted, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    ring = cv.inRange(dist, 0, pixel_expansion)  # take all pixels at distance between 0 px and pixel_expansion px
    ring = (ring / 255).astype(np.uint8)

    return ring


def normalize_expression_data(expression_data,
                              transformation="arcsinh",
                              normalization="percentile",
                              scaling_factor=100,
                              n_markers=34):
    """
    Normalize expression vectors

    :param expression_data:
    :param transformation:
    :param normalization:
    :param scaling_factor:
    :param n_markers:
    :return:
    """

    if scaling_factor > 0:
        expression_data = np.array(expression_data) * scaling_factor

    if transformation == "quantiletransform":
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0,
                                                                 n_quantiles=100)
        expression_data = quantile_transformer.fit_transform(np.array(expression_data))

        # expression_data += abs(np.min(expression_data))
    elif transformation == "boxcox":
        expression_data, _ = boxcox(expression_data)

    elif transformation == "sqrt":
        expression_data = np.sqrt(expression_data)

    elif transformation == "log":
        expression_data = np.log(expression_data + 1)

    elif transformation == "arcsinh":
        # Apply arcsinh transformation
        expression_data = arcsinh(np.array(expression_data))

    if normalization == "percentile":
        try:
            expression_data = np.array(expression_data) / np.percentile(np.array(expression_data),
                                                                        99,
                                                                        axis=0)
            expression_data = np.nan_to_num(expression_data)

        except IndexError:
            print("Caught Exception!", len(expression_data), n_markers)
            expression_data = np.zeros((1, n_markers))

    elif normalization == "normalizer":
        # Scale data by Sklearn normalizer
        sc = Normalizer().fit(np.array(expression_data))
        expression_data = sc.transform(expression_data)

    return expression_data


def preprocess_marker_data(data,
                           mask,
                           expression_type="area_normalized_counts"):
    """
    Take raw marker counts and return processed expression vectors

    :param data:
    :param mask:
    :param expression_type:
    :return:
    """

    assert expression_type in ["mean", "area_normalized_counts", "counts"], "Unrecognized expression type!"

    if expression_type == "mean":
        # Get mean intensity of marker
        marker_data = cv.mean(data)[0]

    elif expression_type == "area_normalized_counts":
        # Get cell area normalized count of marker
        if cv.countNonZero(data) != 0:
            marker_data = np.sum(data, axis=None) / cv.countNonZero(mask)
        else:
            marker_data = 0

    elif expression_type == "counts":
        # Get cell area normalized count of marker
        marker_data = cv.countNonZero(data)

    return marker_data


def expansion_ring_plots(contours,
                         expansion_image,
                         pixel_expansion_amount=5,
                         prev_pixel_expansion_amount=0):
    """
    Draw rings on vessel mask

    :param contours:
    :param expansion_image:
    :param pixel_expansion_amount:
    :param prev_pixel_expansion_amount:
    :return:
    """

    img_shape = expansion_image.shape
    regions = get_assigned_regions(contours, img_shape)

    pix_val = random.randint(50, 255)

    for idx, cnt in enumerate(contours):

        if pixel_expansion_amount < 0:
            mask_expanded = contract_vessel_region(cnt, img_shape, pixel_expansion=pixel_expansion_amount)
        else:
            mask_expanded = expand_vessel_region(cnt, img_shape, pixel_expansion=pixel_expansion_amount)

        if prev_pixel_expansion_amount != 0:
            if prev_pixel_expansion_amount < 0:
                mask = contract_vessel_region(cnt, img_shape, pixel_expansion=prev_pixel_expansion_amount)
            else:
                mask = expand_vessel_region(cnt, img_shape, pixel_expansion=prev_pixel_expansion_amount)
        else:
            mask = np.zeros(img_shape, np.uint8)

        result_mask = mask_expanded - mask
        result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))

        _, temp_contours, _ = cv.findContours(regions[idx].astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(expansion_image, temp_contours, 0, (255, 255, 255), 1)
        expansion_image[np.where(result_mask != 0)] = pix_val


def calculate_microenvironment_marker_expression(markers_data, contours,
                                                 scaling_factor=100,
                                                 pixel_expansion_upper_bound=5,
                                                 pixel_expansion_lower_bound=0,
                                                 expression_type="area_normalized_counts",
                                                 transformation="arcsinh",
                                                 normalization="percentile",
                                                 plot=False,
                                                 n_markers=34,
                                                 plot_vesselnonvessel_mask=True,
                                                 vesselnonvessel_label="Point1"):
    """
    Get normalized expression of markers in given cells

    :param vesselnonvessel_label:
    :param plot_vesselnonvessel_mask:
    :param n_markers:
    :param pixel_expansion_lower_bound:
    :param pixel_expansion_upper_bound:
    :param plot:
    :param scaling_factor: Scaling factor by which to scale the data
    :param normalization: Method to scale data
    :param transformation: Transformation of expression vector
    :param expression_type: Method of determining protein expression
    :param markers_data: Pixel data for each marker
    :param contours: Contours of cells in image
    :return:
    """

    contour_mean_values = []
    dark_space_values = []
    vessel_space_values = []
    expression_images = []

    img_shape = markers_data[0].shape
    regions = get_assigned_regions(contours, img_shape)

    stopped_vessels = 0

    if plot_vesselnonvessel_mask:
        example_img = np.zeros(img_shape, np.uint8)
        example_img = cv.cvtColor(example_img, cv.COLOR_GRAY2BGR)

    for idx, cnt in enumerate(contours):
        data_vec = []
        dark_space_vec = []
        vessel_space_vec = []

        expression_image = []

        if pixel_expansion_upper_bound < 0:
            mask_expanded = contract_vessel_region(cnt, img_shape, pixel_expansion=pixel_expansion_upper_bound)
        else:
            mask_expanded = expand_vessel_region(cnt, img_shape, pixel_expansion=pixel_expansion_upper_bound)

        if pixel_expansion_lower_bound != 0:
            if pixel_expansion_lower_bound < 0:
                mask = contract_vessel_region(cnt, img_shape, pixel_expansion=pixel_expansion_lower_bound)
            else:
                mask = expand_vessel_region(cnt, img_shape, pixel_expansion=pixel_expansion_lower_bound)
        else:
            mask = cv.drawContours(np.zeros(img_shape, np.uint8), [cnt], -1, (1, 1, 1), cv.FILLED)

        result_mask = mask_expanded - mask
        result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))
        dark_space_mask = regions[idx].astype(np.uint8) - mask_expanded

        if plot_vesselnonvessel_mask:
            example_img[np.where(dark_space_mask == 1)] = (0, 0, 255)
            example_img[np.where(mask_expanded == 1)] = (0, 255, 0)

        if cv.countNonZero(result_mask) == 0:
            stopped_vessels += 1

        for marker in markers_data:
            x, y, w, h = cv.boundingRect(cnt)

            result = cv.bitwise_and(marker, marker, mask=result_mask)
            dark_space_result = cv.bitwise_and(marker, marker, mask=dark_space_mask)
            vessel_space_result = cv.bitwise_and(marker, marker, mask=mask_expanded)

            roi_result = result[y:y + h, x:x + w]

            expression_image.append(roi_result)

            marker_data = preprocess_marker_data(result,
                                                 result_mask,
                                                 expression_type=expression_type)

            dark_space_data = preprocess_marker_data(dark_space_result,
                                                     dark_space_mask,
                                                     expression_type=expression_type)

            vessel_space_data = preprocess_marker_data(vessel_space_result,
                                                       mask_expanded,
                                                       expression_type=expression_type)

            data_vec.append(marker_data)
            dark_space_vec.append(dark_space_data)
            vessel_space_vec.append(vessel_space_data)

        contour_mean_values.append(np.array(data_vec))
        dark_space_values.append(np.array(dark_space_vec))
        vessel_space_values.append(np.array(vessel_space_vec))

        expression_images.append(expression_image)

    contour_mean_values = normalize_expression_data(contour_mean_values,
                                                    transformation=transformation,
                                                    normalization=normalization,
                                                    scaling_factor=scaling_factor,
                                                    n_markers=n_markers)

    vessel_non_vessel_data = []
    vessel_non_vessel_data.extend(dark_space_values)
    vessel_non_vessel_data.extend(vessel_space_values)

    vessel_non_vessel_data = normalize_expression_data(vessel_non_vessel_data,
                                                       transformation=transformation,
                                                       normalization=normalization,
                                                       scaling_factor=scaling_factor,
                                                       n_markers=n_markers)

    dark_space_values = vessel_non_vessel_data[0:len(dark_space_values)]
    vessel_space_values = vessel_non_vessel_data[len(dark_space_values):]

    flat_list = sorted([item for sublist in contour_mean_values for item in sublist])

    if plot:
        plt.plot(np.array(flat_list), stats.norm.pdf(np.array(flat_list)))
        plt.xlabel('Area Normalized Marker Expression')
        plt.ylabel('Probability')
        plt.title('PDF of Marker Expression')
        plt.show()

    if plot_vesselnonvessel_mask:
        output_dir = "results/vessel_nonvessel_masks/%s_pixel_expansion" % str(pixel_expansion_upper_bound)
        mkdir_p(output_dir)
        cv.imwrite(os.path.join(output_dir, "vessel_non_vessel_point_%s.png" % vesselnonvessel_label), example_img)

    return contour_mean_values, expression_images, stopped_vessels, dark_space_values, vessel_space_values


def calculate_composition_marker_expression(markers_data, contours,
                                            scaling_factor=100,
                                            expression_type="area_normalized_counts",
                                            transformation="arcsinh",
                                            normalization="percentile",
                                            plot=True,
                                            n_markers=34,
                                            vessel_id_plot=False,
                                            vessel_id_label="Point1"):
    """
    Get normalized expression of markers in given cells

    :param vessel_id_label:
    :param n_markers:
    :param vessel_id_plot:
    :param plot:
    :param scaling_factor: Scaling factor by which to scale the data
    :param normalization: Method to scale data
    :param transformation: Transformation of expression vector
    :param expression_type: Method of determining protein expression
    :param markers_data: Pixel data for each marker
    :param contours: Contours of cells in image
    :return:
    """

    contour_mean_values = []
    img_shape = markers_data[0].shape

    if vessel_id_plot:
        vessel_id_img = np.zeros(markers_data[0].shape)

    for idx, cnt in enumerate(contours):
        data_vec = []

        if vessel_id_plot:
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.drawContours(vessel_id_img, [cnt], -1, (255, 255, 255), 1)
            cv.putText(vessel_id_img, str(idx), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        mask = np.zeros(img_shape, np.uint8)
        cv.drawContours(mask, [cnt], -1, (1, 1, 1), cv.FILLED)

        for marker in markers_data:
            result = cv.bitwise_and(marker, marker, mask=mask)

            marker_data = preprocess_marker_data(result,
                                                 mask,
                                                 expression_type=expression_type)

            data_vec.append(marker_data)

        contour_mean_values.append(np.array(data_vec))

    contour_mean_values = normalize_expression_data(contour_mean_values,
                                                    transformation=transformation,
                                                    normalization=normalization,
                                                    scaling_factor=scaling_factor,
                                                    n_markers=n_markers)

    flat_list = sorted([item for sublist in contour_mean_values for item in sublist])

    if plot:
        plt.plot(np.array(flat_list), stats.norm.pdf(np.array(flat_list)))
        plt.xlabel('Area Normalized Marker Expression')
        plt.ylabel('Probability')
        plt.title('PDF of Marker Expression')
        plt.show()

    if vessel_id_plot:
        output_dir = "results/vessel_id_masks"
        mkdir_p(output_dir)
        cv.imwrite(os.path.join(output_dir, "vessel_id_plot_%s.png" % vessel_id_label), vessel_id_img)

    return contour_mean_values
