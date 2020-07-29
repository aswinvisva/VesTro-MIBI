import json
import random
from collections import Counter

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def label_image_watershed(original, contours, indices, no_topics=20, show_plot=True):
    img = original.copy()
    data = np.full((img.shape[0], img.shape[1], 1), -1)

    with open('config/cluster_colours.json') as json_file:
        colors = json.load(json_file)

    index = 0

    for cnt in contours:
        M = cv.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        data[cY][cX] = int(indices[index])

        color = colors[str(indices[index])]
        cv.drawContours(img, [cnt], 0, color, thickness=-1)
        index = index + 1

    if show_plot:
        legend_color_values = [(colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255) for i in colors.keys()]

        patches = [mpatches.Patch(color=legend_color_values[i], label="Cell Type {l}".format(l=i)) for i in
                   range(no_topics)]
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img)
        axarr[0].legend(handles=patches, bbox_to_anchor=(-0.3, 1), loc=2, borderaxespad=0.)

        axarr[1].imshow(original)
        plt.show()
    else:
        cv.imshow("Segmented Image", img)
        cv.waitKey(0)

    return img, data


def plot_vessel_areas(contours, img):
    areas = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        ROI = img[y:y + h, x:x + w]
        contour_area = cv.contourArea(cnt)
        areas.append(contour_area)

    areas = sorted(areas)
    plt.hist(areas, bins=50)
    plt.show()


def oversegmentation_watershed(img,
                               show=False,
                               min_contour_area=0.1):
    if img.shape[2] == 3:
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        imgray = img

    ret, thresh = cv.threshold(imgray, 15, 255, 0)

    if show:
        cv.imshow("Threshold Image", thresh)
        cv.waitKey(0)

    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    images = []
    usable_contours = []

    for i, cnt in enumerate(contours):
        contour_area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        ROI = img[y:y + h, x:x + w]

        mean = hierarchy[0, i, 3]

        if contour_area < min_contour_area:
            if show:
                cv.imshow("Removed Vessel", ROI)
                cv.waitKey(0)
            continue

        # Remove contours which are not a part of the segmentation mask
        if mean != -1:
            if show:
                cv.imshow("Removed Vessel", ROI)
                cv.waitKey(0)
            continue

        images.append(ROI)
        usable_contours.append(cnt)

    if show:
        copy = img.copy()
        cv.imshow("Segmented Cells", cv.drawContours(copy, usable_contours, -1, (0, 255, 0), 3))
        cv.waitKey(0)
        del copy

    return images, usable_contours
