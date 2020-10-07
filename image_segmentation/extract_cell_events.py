import json
import random
from collections import Counter

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import seaborn as sns
import pandas as pd

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


def plot_vessel_areas(points_contours, points_img,
                      save_csv=False,
                      segmentation_type='allvessels',
                      show_outliers=False):
    brain_regions = [(1, 16), (17, 32), (33, 48)]
    region_data = []
    current_point = 1
    current_region = 0
    areas = []
    per_point_areas = []
    total_per_point_areas = []

    total_point_vessel_areas = []

    for idx, contours in enumerate(points_contours):
        img = points_img[idx]
        current_per_point_area = []

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            ROI = img[y:y + h, x:x + w]
            contour_area = cv.contourArea(cnt)
            areas.append(contour_area)
            current_per_point_area.append(contour_area)

        current_point += 1
        per_point_areas.append(current_per_point_area)
        total_point_vessel_areas.append(current_per_point_area)

        if not (brain_regions[current_region][0] <= current_point <= brain_regions[current_region][1]):
            current_region += 1
            region_data.append(sorted(areas))
            total_per_point_areas.append(per_point_areas)
            areas = []
            per_point_areas = []

    if save_csv:
        area = pd.DataFrame(total_point_vessel_areas)
        area.to_csv('vessel_areas.csv')

    for i, area in enumerate(region_data):
        area = sorted(area)
        plt.hist(area, bins=200)
        plt.title("Points %s to %s" % (str(brain_regions[i][0]), str(brain_regions[i][1])))
        plt.xlabel("Pixel Area")
        plt.ylabel("Count")
        plt.show()

    colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']

    fig = plt.figure(1, figsize=(9, 6))
    plt.title("%s Mask Points 1 to 48" % segmentation_type)

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(total_point_vessel_areas, showfliers=show_outliers, patch_artist=True)

    for w, region in enumerate(brain_regions):
        patches = bp['boxes'][region[0]-1:region[1]]

        for patch in patches:
            patch.set(facecolor=colors[w])

    plt.show()

    return total_point_vessel_areas


def extract_cell_contours(img,
                          show=False,
                          min_contour_area=30):
    if img.shape[2] == 3:
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        imgray = img

    imgray = cv.blur(imgray, (2, 2))
    im2, contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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

        if show:
            print("(x1: %s, x2: %s, y1: %s, y2: %s), w: %s, h: %s" % (x, x + w, y, y + h, w, h))

            cv.imshow("Vessel", ROI)
            cv.waitKey(0)

        images.append(ROI)
        usable_contours.append(cnt)

    if show:
        copy = img.copy()
        cv.imshow("Segmented Cells", cv.drawContours(copy, usable_contours, -1, (0, 255, 0), 3))
        cv.waitKey(0)
        del copy

    return images, usable_contours
