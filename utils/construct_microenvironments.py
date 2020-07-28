import json
import random

import numpy as np
import cv2 as cv

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def construct_microenvironments_from_contours(contours, img, r=256):
    microenvironments = []

    height, width, _ = img.shape

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if w > r or h > r:
            if w > h:
                r = w
            else:
                r = h

        cx = x + w / 2
        cy = y + h / 2

        x1 = int(cx - r)
        x2 = int(cx + r)
        y1 = int(cy - r)
        y2 = int(cy + r)

        if x1 < 0:
            x1 = 0

        if y1 < 0:
            y1 = 0

        if x2 > width:
            x2 = width

        if y2 > height:
            y2 = height

        ROI = img[y1:y2, x1:x2]
        cv.imshow("ASD", ROI)
        cv.waitKey(0)
        microenvironments.append(ROI)

    return microenvironments
