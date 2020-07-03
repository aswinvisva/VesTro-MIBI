import json
import random

import numpy as np
import cv2 as cv

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''

def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def label_image_watershed(img, contours, indices, topics=10):
    original = np.array(img)

    with open('config/cluster_colours.json') as json_file:
        colors = json.load(json_file)

    index = 0
    for cnt in contours:
        color = colors[str(indices[index])]
        cv.drawContours(img, [cnt], 0, color, thickness=-1)
        index = index + 1

    cv.imshow("Segmented", img)
    cv.waitKey(0)


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
