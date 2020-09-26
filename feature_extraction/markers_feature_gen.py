import math
import random
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


def expand_vessel_region_distance_transform(cnt, shape, pixel_expansion=5):
    inverted = np.ones(shape, np.uint8)
    cv.drawContours(inverted, [cnt], -1, (0, 0, 0), cv.FILLED)

    dist = cv.distanceTransform(inverted, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    ring = cv.inRange(dist, 0, pixel_expansion)  # take all pixels at distance between 9.5px and 10.5px
    im2, contours, hierarchy = cv.findContours(ring, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours.sort(key=lambda x: cv.contourArea(x), reverse=True)

    return contours[0]


def expand_vessel_region(cnt, pixel_expansion=5):
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    left = tuple(cnt_norm[cnt_norm[:, :, 0].argmin()][0])
    right = tuple(cnt_norm[cnt_norm[:, :, 0].argmax()][0])
    top = tuple(cnt_norm[cnt_norm[:, :, 1].argmin()][0])
    bottom = tuple(cnt_norm[cnt_norm[:, :, 1].argmax()][0])

    origin = np.array([0, 0])

    max_dist = max(np.linalg.norm(left - origin),
                   np.linalg.norm(right - origin),
                   np.linalg.norm(top - origin),
                   np.linalg.norm(bottom - origin))

    scale = 1 + (pixel_expansion / max_dist)

    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def expansion_ring_plots(contours,
                         expansion_image,
                         pixel_expansion_amount=5,
                         prev_pixel_expansion_amount=0):

    img_shape = expansion_image.shape
    regions = get_assigned_regions(contours, img_shape)

    pix_val = random.randint(50, 255)

    for idx, cnt in enumerate(contours):
        expanded_cnt = expand_vessel_region_distance_transform(cnt, img_shape, pixel_expansion=pixel_expansion_amount)

        if prev_pixel_expansion_amount != 0:
            cnt = expand_vessel_region_distance_transform(cnt, img_shape, pixel_expansion=prev_pixel_expansion_amount)

        mask = np.zeros(img_shape, np.uint8)

        cv.drawContours(mask, [cnt], -1, (255, 255, 255), cv.FILLED)

        mask_expanded = np.zeros(img_shape, np.uint8)

        cv.drawContours(mask_expanded, [expanded_cnt], -1, (255, 255, 255), cv.FILLED)

        result_mask = mask_expanded - mask
        result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))

        _, temp_contours, _ = cv.findContours(regions[idx].astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(expansion_image, temp_contours, 0, (255, 255, 255), 1)
        expansion_image[np.where(result_mask != 0)] = pix_val


def calculate_microenvironment_marker_expression_single_vessel(markers_data, contours,
                                                               scaling_factor=100,
                                                               pixel_expansion_amount=5,
                                                               prev_pixel_expansion_amount=0,
                                                               expression_type="area_normalized_counts",
                                                               transformation="arcsinh",
                                                               normalization="percentile",
                                                               plot=True,
                                                               expansion_image=None,
                                                               n_markers=34):
    '''
    Get normalized expression of markers in given cells

    :param n_markers:
    :param expansion_image:
    :param prev_pixel_expansion_amount:
    :param pixel_expansion_amount:
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

    img_shape = markers_data[0].shape
    regions = get_assigned_regions(contours, img_shape)

    if expansion_image is not None:
        pix_val = random.randint(50, 255)

    for idx, cnt in enumerate(contours):
        data_vec = []
        expression_image = []

        expanded_cnt = expand_vessel_region_distance_transform(cnt, img_shape, pixel_expansion=pixel_expansion_amount)

        if prev_pixel_expansion_amount != 0:
            cnt = expand_vessel_region_distance_transform(cnt, img_shape, pixel_expansion=prev_pixel_expansion_amount)

        for marker in markers_data:
            x, y, w, h = cv.boundingRect(cnt)

            mask = np.zeros(marker.shape, np.uint8)

            cv.drawContours(mask, [cnt], -1, (255, 255, 255), cv.FILLED)

            mask_expanded = np.zeros(marker.shape, np.uint8)

            cv.drawContours(mask_expanded, [expanded_cnt], -1, (255, 255, 255), cv.FILLED)

            result_mask = mask_expanded - mask
            result_mask = cv.bitwise_and(result_mask, regions[idx].astype(np.uint8))

            result = cv.bitwise_and(marker, result_mask)

            roi_result = result[y:y + h, x:x + w]

            expression_image.append(roi_result)

            if plot:
                if idx == 5:
                    cv.imshow("ASD", regions[idx].astype(np.uint8) * 255)
                    cv.imshow("mask", mask)
                    cv.imshow("mask_expanded", mask_expanded)
                    cv.imshow("result_mask", result_mask*255)
                    cv.imshow("roi_result", roi_result * 255)
                    cv.waitKey(0)

            if expression_type == "mean":
                # Get mean intensity of marker
                marker_data = cv.mean(result)[0]

            elif expression_type == "area_normalized_counts":
                # Get cell area normalized count of marker
                marker_data = cv.countNonZero(result) / cv.contourArea(cnt)

            elif expression_type == "counts":
                # Get cell area normalized count of marker
                marker_data = cv.countNonZero(result)

            data_vec.append(marker_data)
            if expansion_image is not None:
                _, temp_contours, _ = cv.findContours(regions[idx].astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(expansion_image, temp_contours, 0, (255, 255, 255), 1)
                expansion_image[np.where(result_mask != 0)] = pix_val

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
        try:
            contour_mean_values = np.array(contour_mean_values) / np.percentile(np.array(contour_mean_values), 99)
        except IndexError:
            print("Caught Exception!", len(contour_mean_values), n_markers)
            contour_mean_values = np.zeros((1, n_markers))
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

    # expression_images = np.asarray(expression_images)

    return contour_mean_values, expression_images


def calculate_marker_composition_single_vessel(markers_data, contours,
                                               scaling_factor=100,
                                               expression_type="area_normalized_counts",
                                               transformation="log",
                                               normalization="percentile",
                                               plot=True,
                                               n_markers=34,
                                               vessel_id_plot=True,
                                               vessel_id_label="Point1"):
    '''
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
    '''

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

        for marker in markers_data:
            x, y, w, h = cv.boundingRect(cnt)

            if expression_type == "mean":
                # Get mean intensity of marker

                mask = np.zeros(marker.shape, np.uint8)
                cv.drawContours(mask, [cnt], -1, (255, 255, 255), 1)
                result = cv.bitwise_and(marker, mask)

                marker_data = cv.mean(result)[0]

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
        try:
            contour_mean_values = np.array(contour_mean_values) / np.percentile(np.array(contour_mean_values), 99)
        except IndexError:
            print("Caught Exception!", len(contour_mean_values), n_markers)
            contour_mean_values = np.zeros((1, n_markers))
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

    if vessel_id_plot:
        cv.imwrite("vessel_id_plot_%s.png" % vessel_id_label, vessel_id_img)

    return contour_mean_values
