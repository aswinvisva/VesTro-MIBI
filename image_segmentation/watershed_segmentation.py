import json
import random

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def label_image_watershed(img, contours, indices, no_topics=20, show_plot=True):
    original = np.array(img)
    print(indices)

    with open('config/cluster_colours.json') as json_file:
        colors = json.load(json_file)

    index = 0

    for cnt in contours:
        color = colors[str(indices[index])]
        cv.drawContours(img, [cnt], 0, color, thickness=-1)
        index = index + 1

    if show_plot:
        legend_color_values = [(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255) for i in colors.keys()]

        patches = [mpatches.Patch(color=legend_color_values[i], label="Cell Type {l}".format(l=i)) for i in range(no_topics)]
        plt.imshow(img)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    else:
        cv.imshow("Combined Image", img)
        cv.waitKey(0)

    return img


def oversegmentation_watershed(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(imgray, 15, 255, 0)
    cv.imshow("Threshold Image", thresh)
    cv.waitKey(0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    images = []
    usable_contours = []

    MIN_CONTOUR_AREA = 125

    for cnt in contours:
        contour_area = cv.contourArea(cnt)

        if contour_area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        ROI = img[y:y + h, x:x + w]
        images.append(ROI)
        usable_contours.append(cnt)

    return images, usable_contours
