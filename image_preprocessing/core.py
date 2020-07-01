import os
import random

from PIL import Image
import numpy as np
import cv2 as cv


def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def marker_stitching(image_loc="data/Point16/TIFs",
                     plot=True):
    images = []

    for root, dirs, files in os.walk(image_loc):
        for file in files:
            path = os.path.join(root, file)

            img = np.asarray(Image.open(path))
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            img[np.where((img > 10).all(axis=2))] = random_color()
            #
            # if plot:
            #     cv.imshow(file, img)
            #     cv.waitKey(0)

            images.append(img)

    mean_img = np.mean(images, axis=0)
    mean_img = mean_img.astype('uint8') * 75

    # mean_img = cv.resize(mean_img, (4096, 4096))

    # mean_img = cv.cvtColor(mean_img, cv.COLOR_GRAY2BGR)

    # gray = cv.cvtColor(mean_img, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(mean_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    if plot:
        cv.imshow("Image", mean_img)
        cv.waitKey(0)

    print(mean_img.shape)

    return mean_img
