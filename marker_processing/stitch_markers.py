import json
import os
import random

from PIL import Image
import numpy as np
import cv2 as cv

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def image_stitching(point_name="Point16",
                    plot=True,
                    threshold=12):
    ignore = [
        "GAD",
        "Neurogranin",
        "ABeta40",
        "pTDP43",
        "polyubik63",
        "Background",
        "Au197",
        "Ca40",
        "Fe56",
        "Na23",
        "Si28",
        "La139",
        "Ta181",
        "C12"
    ]

    image_loc = "data/" + point_name + "/TIFs"

    marker_names = []
    marker_images = []
    images = []

    with open('config/marker_colours.json') as json_file:
        data = json.load(json_file)

    for root, dirs, files in os.walk(image_loc):
        for file in files:
            file_name = os.path.splitext(file)[0]

            path = os.path.join(root, file)

            img = np.asarray(Image.open(path))
            img = img.reshape(img.shape[0], img.shape[1], 1)

            if file_name not in ignore:
                marker_images.append(img.copy())
                marker_names.append(file_name)

            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            img[np.where((img > threshold).all(axis=2))] = eval(data[file_name])

            cv.imwrite(os.path.join("annotated_data/" + point_name, file_name + ".jpg"), img)

            images.append(img)

    mean_img = np.mean(images, axis=0)
    mean_img = mean_img.astype('uint8') * 100
    markers_img = np.array(marker_images)

    if plot:
        cv.imshow("Combined Image", mean_img)
        cv.waitKey(0)

    cv.imwrite(os.path.join("annotated_data/" + point_name, "total.jpg"), mean_img)

    return mean_img, markers_img, marker_names
