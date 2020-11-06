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
from PIL import Image
import pandas as pd

from utils.utils_functions import mkdir_p
import config.config_settings as config

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


def get_assigned_regions(per_point_contours: list, img_shape: (int, int)) -> list:
    """
    Get vessel boundaries beyond which a vessel cannot expand

    :param per_point_contours: list, [n_vessels] -> List of vessel contours in a given point
    :param img_shape: tuple, Point data size ex. (1024, 1024)

    :return: list, [n_vessels, point_size[0], point_size[1]] of region masks for each vessel beyond which it cannot
    expand
    """

    dist_mats = []
    regions = []

    # Iterate through all vessel contours and create distance transform matrices
    for i in range(len(per_point_contours)):
        sub_mask = np.ones(img_shape, np.uint8)
        cv.drawContours(sub_mask, per_point_contours, i, (0, 0, 0), cv.FILLED)
        dist = cv.distanceTransform(sub_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dist_mats.append(dist)

    # Iterate through all contours and compare distance matrices to find all pixels "belonging" to a given vessel,
    # this will create the region mask beyond which the vessel should not expand in order to avoid vessels from
    # encroaching on other vessel's space

    for i in range(len(per_point_contours)):
        region = np.ones(img_shape, dtype=bool)
        for j in range(len(per_point_contours)):
            if i != j:
                region = region & (dist_mats[i] < dist_mats[j])

        regions.append(region)

    del dist_mats

    return regions


def arcsinh(data: list, cofactor: int = 5) -> np.ndarray:
    """
    Inverse hyperbolic sine transform

    :param data: array_like, [n_vessels, n_markers] -> Input data
    :param cofactor: int, Factor by which to divide data before arcsinh transform
    :return: array_like, [n_vessels, n_markers] -> Transformed data
    """

    if cofactor <= 0:
        raise ValueError("Expected cofactor > 0 or None. " "Got {}".format(cofactor))
    if cofactor is not None:
        data = data / cofactor

    return np.arcsinh(data)


def contract_vessel_region(cnt: np.ndarray,
                           img_shape: (int, int),
                           upper_bound: int = 5,
                           lower_bound: int = 0) -> np.ndarray:
    """
    Create a contracted vessel mask

    :param cnt: np.ndarray, Vessel contour
    :param img_shape: tuple, Point data size ex. (1024, 1024)
    :param upper_bound: int, Pixel upper bound to contract by
    :param lower_bound: int, Pixel lower bound to contract by
    :return: array_like, [point_size[0], point_size[1]] -> Vessel ring mask
    """
    if lower_bound == 0:
        lower_bound = 0.25

    upper_bound = abs(upper_bound)
    zeros = np.zeros(img_shape, np.uint8)
    cv.drawContours(zeros, [cnt], -1, (255, 255, 255), cv.FILLED)

    dist = cv.distanceTransform(zeros, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    ring = cv.inRange(dist, lower_bound, upper_bound)  # take all pixels at distance between 0 px and pixel_expansion px
    ring = (ring / 255).astype(np.uint8)

    return ring


def expand_vessel_region(cnt: np.ndarray, img_shape: (int, int),
                         upper_bound: int = 5,
                         lower_bound: int = 0) -> np.ndarray:
    """
    Create an expanded vessel mask

    :param lower_bound: int, Pixel Lower Bound
    :param cnt: np.ndarray, Vessel contour
    :param img_shape: tuple, Point data size ex. (1024, 1024)
    :param upper_bound: int, Pixel expansion
    :return: array_like, [point_size[0], point_size[1]] -> Vessel ring mask
    """

    inverted = np.ones(img_shape, np.uint8)
    cv.drawContours(inverted, [cnt], -1, (0, 0, 0), cv.FILLED)

    dist = cv.distanceTransform(inverted, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    ring = cv.inRange(dist, lower_bound, upper_bound)  # take all pixels at distance between 0 px and pixel_expansion px
    ring = (ring / 255).astype(np.uint8)

    return ring


def normalize_expression_data(expression_data_df: pd.DataFrame,
                              transformation: str = "arcsinh",
                              normalization: str = "percentile",
                              scaling_factor: int = 100,
                              n_markers: int = 34) -> np.ndarray:
    """
    Normalize expression vectors

    :param expression_data_df: pd.DataFrame, [n_vessels, n_markers] -> Marker expression data per vessel
    :param transformation: str, Transformation type
    :param normalization: str, Normalization type
    :param scaling_factor: int, Scaling factor
    :param n_markers: int, Number of markers
    :return:
    """

    expression_data = expression_data_df.to_numpy()

    if scaling_factor > 0:
        expression_data = expression_data * scaling_factor

    if transformation == "quantiletransform":
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0,
                                                                 n_quantiles=100)
        expression_data = quantile_transformer.fit_transform(expression_data)

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

    elif transformation == "square":
        expression_data = np.square(expression_data)

    if normalization == "percentile":
        try:
            expression_data = expression_data / np.percentile(expression_data,
                                                              config.percentile_to_normalize,
                                                              axis=0)
            expression_data = np.nan_to_num(expression_data)

        except IndexError:
            print("Caught Exception!", len(expression_data), n_markers)
            expression_data = np.zeros((1, n_markers))

    elif normalization == "normalizer":
        # Scale data by Sklearn normalizer
        expression_data = normalize(expression_data, axis=0)

    scaled_expression_df = pd.DataFrame(expression_data,
                                        columns=list(expression_data_df),
                                        index=expression_data_df.index)

    return scaled_expression_df


def preprocess_marker_data(marker_data: np.ndarray,
                           mask: np.ndarray,
                           expression_type: str = "area_normalized_counts") -> np.ndarray:
    """
    Take raw marker counts and return processed expression vectors

    :param marker_data: array_like, [point_size[0], point_size[1]] -> Raw marker counts
    :param mask: array_like, [point_size[0], point_size[1]] -> Segmentation mask
    :param expression_type: str, Expression type
    :return: array_like, [n_markers] -> Marker data vector
    """

    assert expression_type in ["mean", "area_normalized_counts", "counts"], "Unrecognized expression type!"

    if expression_type == "mean":
        # Get mean intensity of marker
        marker_data = cv.mean(marker_data)[0]

    elif expression_type == "area_normalized_counts":
        # Get cell area normalized count of marker
        if cv.countNonZero(marker_data) != 0:
            marker_data = np.sum(marker_data, axis=None) / (cv.countNonZero(mask) * (config.pixel_area_scaler ** 2))
        else:
            marker_data = 0

    elif expression_type == "counts":
        # Get cell area normalized count of marker
        marker_data = cv.countNonZero(marker_data)

    return marker_data


def expansion_ring_plots(per_point_contours: list,
                         expansion_image: np.ndarray,
                         pixel_expansion_upper_bound: int = 5,
                         pixel_expansion_lower_bound: int = 0,
                         color: (int, int, int) = (255, 255, 255)):
    """
    Draw rings on vessel mask

    :param color: tuple, Color to fill rings
    :param per_point_contours: list, Vessel contours per a given point
    :param expansion_image: array_like, image to draw vessel rings on
    :param pixel_expansion_upper_bound: int, Pixel upper bound to expand by
    :param pixel_expansion_lower_bound: int, Pixel lower bound to expand by
    """

    img_shape = expansion_image.shape
    regions = get_assigned_regions(per_point_contours, img_shape)

    for idx, cnt in enumerate(per_point_contours):

        if pixel_expansion_upper_bound < 0:
            mask_expanded = contract_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_upper_bound)
        else:
            mask_expanded = expand_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_upper_bound)

        if pixel_expansion_lower_bound != 0:
            if pixel_expansion_lower_bound < 0:
                mask = contract_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_lower_bound)
            else:
                mask = expand_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_lower_bound)
        else:
            mask = np.zeros(img_shape, np.uint8)

        result_mask = mask_expanded - mask
        result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))

        _, temp_contours, _ = cv.findContours(regions[idx].astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(expansion_image, temp_contours, 0, (255, 255, 255), 1)
        expansion_image[np.where(result_mask != 0)] = color[0]


def get_microenvironment_masks(per_point_marker_data: np.ndarray,
                               per_point_vessel_contours: list,
                               pixel_expansion_upper_bound: int = 5,
                               pixel_expansion_lower_bound: int = 0):
    """
    TODO: Unravel microenvironment masks to eliminate vessel geometries
    Get microenvironment masks for CNN analysis

    :param pixel_expansion_lower_bound: int, Lower bound to expand
    :param pixel_expansion_upper_bound: int, Upper bound to expand
    :param per_point_marker_data: array_like, [n_markers, point_size[0], point_size[1]] -> Pixel data for each marker
    :param per_point_vessel_contours: list, [n_vessels] -> Contours of cells in image
    """
    microenvironment_masks = []

    img_shape = per_point_marker_data[0].shape
    regions = get_assigned_regions(per_point_vessel_contours, img_shape)

    for idx, cnt in enumerate(per_point_vessel_contours):
        microenvironment_mask = []

        if pixel_expansion_upper_bound < 0:
            mask_expanded = contract_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_upper_bound)
        else:
            mask_expanded = expand_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_upper_bound)

        if pixel_expansion_lower_bound != 0:
            if pixel_expansion_lower_bound < 0:
                mask = contract_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_lower_bound)
            else:
                mask = expand_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_lower_bound)
        else:
            mask = cv.drawContours(np.zeros(img_shape, np.uint8), [cnt], -1, (1, 1, 1), cv.FILLED)

        result_mask = mask_expanded - mask
        result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))

        for marker in per_point_marker_data:
            x, y, w, h = cv.boundingRect(cnt)

            result = cv.bitwise_and(marker, marker, mask=result_mask)

            cv.imshow("ASD", result * 255)
            cv.waitKey(0)

            microenvironment_mask.append(result)

        microenvironment_mask = np.array(microenvironment_mask)
        microenvironment_masks.append(microenvironment_mask)


def calculate_inward_microenvironment_marker_expression(per_point_marker_data: np.ndarray,
                                                        per_point_vessel_contours: list,
                                                        pixel_expansion_upper_bound: int = 5,
                                                        pixel_expansion_lower_bound: int = 0) -> (np.ndarray, int):
    """
    Get normalized expression of markers in given cells

    :param pixel_expansion_lower_bound: int, Lower bound to expand
    :param pixel_expansion_upper_bound: int, Upper bound to expand
    :param per_point_marker_data: array_like, [n_markers, point_size[0], point_size[1]] -> Pixel data for each marker
    :param per_point_vessel_contours: list, [n_vessels] -> Contours of cells in image
    :returns per_point_vessel_expression_data: array_like, [n_vessels, n_markers] -> Per point vessels expression data,
    stopped_vessels: int, Number of vessels which couldn't expand inwards
    """

    scaling_factor = config.scaling_factor
    expression_type = config.expression_type
    transformation = config.transformation_type
    normalization = config.normalization_type
    plot = config.show_probability_distribution_for_expression
    n_markers = config.n_markers

    per_point_vessel_expression_data = []

    img_shape = per_point_marker_data[0].shape

    stopped_vessels = 0

    for idx, cnt in enumerate(per_point_vessel_contours):
        data_vec = []
        expression_image = []

        result_mask = contract_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_upper_bound,
                                             lower_bound=pixel_expansion_lower_bound)

        if config.show_vessel_masks_when_generating_expression:
            cv.imshow("Vessel Mask", result_mask * 255)
            cv.waitKey(0)

        if cv.countNonZero(result_mask) == 0:
            stopped_vessels += 1

        for marker in per_point_marker_data:
            x, y, w, h = cv.boundingRect(cnt)

            result = cv.bitwise_and(marker, marker, mask=result_mask)

            roi_result = result[y:y + h, x:x + w]

            expression_image.append(roi_result)

            marker_data = preprocess_marker_data(result,
                                                 result_mask,
                                                 expression_type=expression_type)
            data_vec.append(marker_data)

        per_point_vessel_expression_data.append(np.array(data_vec))

    per_point_vessel_expression_data = normalize_expression_data(per_point_vessel_expression_data,
                                                                 transformation=transformation,
                                                                 normalization=normalization,
                                                                 scaling_factor=scaling_factor,
                                                                 n_markers=n_markers)

    flat_list = sorted([item for sublist in per_point_vessel_expression_data for item in sublist])

    if plot:
        plt.plot(np.array(flat_list), stats.norm.pdf(np.array(flat_list)))
        plt.xlabel('Area Normalized Marker Expression')
        plt.ylabel('Probability')
        plt.title('PDF of Marker Expression')
        plt.show()

    return per_point_vessel_expression_data, stopped_vessels


def calculate_microenvironment_marker_expression(per_point_marker_data: np.ndarray,
                                                 per_point_vessel_contours: list,
                                                 marker_names: list,
                                                 pixel_expansion_upper_bound: int = 5,
                                                 pixel_expansion_lower_bound: int = 0,
                                                 point_num: int = 1,
                                                 expansion_num: int = 1) -> (np.ndarray,
                                                                             list,
                                                                             int,
                                                                             np.ndarray,
                                                                             np.ndarray):
    """
    Get normalized expression of markers in given cells

    :param marker_names: list, Marker names
    :param expansion_num: int, Current expansion number
    :param pixel_expansion_lower_bound: int, Lower bound to expand
    :param pixel_expansion_upper_bound: int, Upper bound to expand
    :param per_point_marker_data: array_like, [n_markers, point_size[0], point_size[1]] -> Pixel data for each marker
    :param per_point_vessel_contours: list, [n_vessels] -> Contours of cells in image
    :param point_num: int, Point from which samples came from

    :returns per_point_microenvironment_expression_data: pd.DataFrame, [n_vessels, n_markers] -> Per point
    microenvironment expression data,
    expression_images: list, [n_vessels, n_markers] -> ROI marker expressions,
    stopped_vessels: int, Number of vessels which couldn't expand inwards,
    """

    expression_type = config.expression_type
    plot = config.show_probability_distribution_for_expression
    plot_vesselnonvessel_mask = config.create_vessel_nonvessel_mask

    per_point_features = []
    expression_images = []

    img_shape = per_point_marker_data[0].shape
    regions = get_assigned_regions(per_point_vessel_contours, img_shape)

    stopped_vessels = 0

    if plot_vesselnonvessel_mask:
        example_img = np.zeros(img_shape, np.uint8)
        example_img = cv.cvtColor(example_img, cv.COLOR_GRAY2BGR)

    for idx, cnt in enumerate(per_point_vessel_contours):
        data_vec = []
        dark_space_vec = []
        vessel_space_vec = []

        expression_image = []

        mask_expanded = expand_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_upper_bound)

        if pixel_expansion_lower_bound != 0:
            mask = expand_vessel_region(cnt, img_shape, upper_bound=pixel_expansion_lower_bound)
        else:
            mask = cv.drawContours(np.zeros(img_shape, np.uint8), [cnt], -1, (1, 1, 1), cv.FILLED)

        result_mask = mask_expanded - mask
        result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))
        mask_expanded = cv.bitwise_and(mask_expanded, regions[idx].astype(np.uint8))
        dark_space_mask = regions[idx].astype(np.uint8) - mask_expanded

        if config.show_vessel_masks_when_generating_expression:
            cv.imshow("Microenvironment Mask", result_mask * 255)
            cv.imshow("Dark Space Mask", dark_space_mask * 255)
            cv.imshow("Expanded Mask", mask_expanded * 255)
            cv.waitKey(0)

        if plot_vesselnonvessel_mask:
            example_img[np.where(dark_space_mask == 1)] = config.nonvessel_mask_colour  # red
            example_img[np.where(mask_expanded == 1)] = config.vessel_space_colour  # green
            cv.drawContours(example_img, [cnt], -1, config.vessel_mask_colour, cv.FILLED)  # blue

        if cv.countNonZero(result_mask) == 0:
            stopped_vessels += 1

        for marker in per_point_marker_data:
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

        features = []

        microenvironment_features = pd.DataFrame(np.array([data_vec]), columns=marker_names)
        microenvironment_features.index = map(lambda a: (point_num, idx, expansion_num, "Data"),
                                              microenvironment_features.index)
        features.append(microenvironment_features)

        nonvascular_features = pd.DataFrame(np.array([dark_space_vec]), columns=marker_names)
        nonvascular_features.index = map(lambda a: (point_num, idx, expansion_num, "Non-Vascular Space"),
                                         nonvascular_features.index)
        features.append(nonvascular_features)

        vascular_features = pd.DataFrame(np.array([vessel_space_vec]), columns=marker_names)
        vascular_features.index = map(lambda a: (point_num, idx, expansion_num, "Vascular Space"),
                                      vascular_features.index)
        features.append(vascular_features)

        all_features = pd.concat(features).fillna(0)
        per_point_features.append(all_features)

        expression_images.append(expression_image)

    all_samples_features = pd.concat(per_point_features).fillna(0)
    all_samples_features.index = pd.MultiIndex.from_tuples(all_samples_features.index)

    if plot:
        idx = pd.IndexSlice
        d = all_samples_features.loc[idx[:, :, :, "Data"], :].to_numpy().flatten()
        plt.plot(np.array(d), stats.norm.pdf(np.array(d)))

        plt.xlabel('Area Normalized Marker Expression')
        plt.ylabel('Probability')
        plt.title('PDF of Marker Expression')
        plt.show()

    if plot_vesselnonvessel_mask:
        vesselnonvessel_label = "Point %s" % point_num

        output_dir = "%s/vessel_nonvessel_masks/%s_pixel_expansion" % (config.visualization_results_dir,
                                                                       str(pixel_expansion_upper_bound))
        mkdir_p(output_dir)
        cv.imwrite(os.path.join(output_dir, "vessel_non_vessel_point_%s.png" % vesselnonvessel_label), example_img)

    return all_samples_features, expression_images, stopped_vessels


def calculate_composition_marker_expression(per_point_marker_data: np.ndarray,
                                            per_point_vessel_contours: list,
                                            marker_names: list,
                                            point_num: int = 1) -> np.ndarray:
    """
    Get normalized expression of markers in given cells

    :param marker_names: list, Marker Names
    :param point_num: int, Point number for vessel ID plot
    :param per_point_marker_data: array_like, [n_markers, point_size[0], point_size[1]] -> Pixel data for each marker
    :param per_point_vessel_contours: list, [n_vessels] -> Contours of cells in image
    :return: per_point_vessel_expression_data: pd.DataFrame, [n_vessels, n_markers] -> Per point vessel expression data
    """

    expression_type = config.expression_type
    plot = config.show_probability_distribution_for_expression
    vessel_id_plot = config.create_vessel_id_plot
    embedded_id_plot = config.create_embedded_vessel_id_masks

    # per_point_vessel_expression_data = []
    per_point_features = []
    img_shape = per_point_marker_data[0].shape

    if vessel_id_plot:
        vessel_id_img = np.zeros(per_point_marker_data[0].shape)

    if embedded_id_plot:
        embedded_id_img = np.zeros(per_point_marker_data[0].shape, np.uint8)

    for idx, cnt in enumerate(per_point_vessel_contours):
        data_vec = []
        vessel_id = idx + 1  # Index from 1 rather than from 0

        mask = np.zeros(img_shape, np.uint8)
        cv.drawContours(mask, [cnt], -1, (1, 1, 1), cv.FILLED)

        if config.show_vessel_masks_when_generating_expression:
            cv.imshow("Vessel Mask", mask * 255)
            cv.waitKey(0)

        if vessel_id_plot:
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.drawContours(vessel_id_img, [cnt], -1, (255, 255, 255), 1)
            cv.putText(vessel_id_img, str(vessel_id), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if embedded_id_plot:
            cv.drawContours(embedded_id_img, [cnt], -1, (vessel_id, vessel_id, vessel_id), cv.FILLED)  # Give all
            # pixels in the contour region value of ID

        for marker in per_point_marker_data:
            result = cv.bitwise_and(marker, marker, mask=mask)

            marker_data = preprocess_marker_data(result,
                                                 mask,
                                                 expression_type=expression_type)

            data_vec.append(marker_data)

        vessel_features = pd.DataFrame(np.array([data_vec]), columns=marker_names)
        vessel_features.index = map(lambda a: (point_num, idx, 0, "Data"),
                                    vessel_features.index)

        per_point_features.append(vessel_features)

    all_samples_features = pd.concat(per_point_features).fillna(0)
    all_samples_features.index = pd.MultiIndex.from_tuples(all_samples_features.index)

    if plot:
        idx = pd.IndexSlice
        d = all_samples_features.loc[idx[:, :, :, "Data"], :].to_numpy().flatten()
        plt.plot(np.array(d), stats.norm.pdf(np.array(d)))
        plt.xlabel('Area Normalized Marker Expression')
        plt.ylabel('Probability')
        plt.title('PDF of Marker Expression')
        plt.show()

    if vessel_id_plot:
        vessel_id_label = "Point %s" % point_num
        output_dir = "%s/vessel_id_masks" % config.visualization_results_dir
        mkdir_p(output_dir)
        cv.imwrite(os.path.join(output_dir, "vessel_id_plot_%s.png" % vessel_id_label), vessel_id_img)

    if embedded_id_plot:
        vessel_id_label = "Point %s" % point_num
        output_dir = "%s/embedded_id_masks" % config.visualization_results_dir
        mkdir_p(output_dir)
        im = Image.fromarray(embedded_id_img)
        im.save(os.path.join(output_dir, "embedded_id_plot_%s.tif" % vessel_id_label))

    return all_samples_features
