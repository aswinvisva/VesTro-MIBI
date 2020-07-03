import json
import random

import numpy as np
import cv2 as cv

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def split_image(img, n=100, show=False):
    images = []
    for r in range(0, img.shape[0], n):
        for c in range(0, img.shape[1], n):
            images.append(img[r:r + n, c:c + n, :])

            if show:
                cv.imshow("Split Image", img[r:r + n, c:c + n, :])
                cv.waitKey(0)

    return images


def label_image(img, indices, topics=10, n=100):
    colors = []
    x = 0
    original = np.array(img)

    for i in range(topics):
        color = list(np.random.choice(range(256), size=3))
        colors.append(color)
    # print(img.shape)
    for r in range(0, img.shape[0], n):
        for c in range(0, img.shape[1], n):

            if img[r:r + n, c:c + n, :] is not None:
                img[r:r + n, c:c + n, :] = colors[indices[x]]
                x = x + 1

    # cv.imshow("Segmented Image", img)
    # cv.imshow("Original Image", original)

    added_image = cv.addWeighted(img, 0.4, original, 0.1, 0)
    cv.imshow('combined', added_image)

    cv.waitKey(0)
