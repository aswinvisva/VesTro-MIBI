import datetime
import json
import os
import random
from collections import Counter

import numpy as np
import cv2 as cv

from utils import tiff_reader, image_denoising

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def stitch_markers(point_name="Point16",
                   plot=True,
                   plot_markers=False,
                   threshold=3,
                   linear_scaling_factor=100,
                   remove_noise=False):

    # Ignore these markers from analysis
    markers_to_ignore = [
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

    # Use these markers for segmentation
    markers_for_segmentation = [
        # Plaques
        "Abeta42",
        # Tangles
        "PHFTau",
        # Microglia
        "CD45",
        "Iba1",
        # Myelin
        "MOG",
        "MAG",
        # Astrocytes
        "S100b",
        "GlnSyn",
        "GFAP",
        # Large Vessels / BBB
        "SMA",
        "CD31",
        "GLUT1",
        "vWF"
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
            img = tiff_reader.read(path, describe=plot_markers)

            if remove_noise:
                start_denoise = datetime.datetime.now()
                img = image_denoising.knn_denoise(img)
                end_denoise = datetime.datetime.now()
                print("Finished %s in %s" % (file_name, end_denoise - start_denoise))

            if file_name not in markers_to_ignore:
                marker_images.append(img.copy())
                marker_names.append(file_name)

            if file_name in markers_for_segmentation:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

                img[np.where((img > threshold).all(axis=2))] = eval(data[file_name])

                cv.imwrite(os.path.join("annotated_data/" + point_name, file_name + ".jpg"), img)

                images.append(img)

    mean_img = np.mean(images, axis=0)
    mean_img = mean_img.astype('uint8')
    mean_img = mean_img * linear_scaling_factor
    markers_img = np.array(marker_images)

    if plot:
        cv.imshow("Combined Image", mean_img)
        cv.waitKey(0)

    cv.imwrite(os.path.join("annotated_data/" + point_name, "total.jpg"), mean_img)

    return mean_img, markers_img, marker_names


def concatenate_multiple_points(points_upper_bound=48):
    fovs = ["Point" + str(i+1) for i in range(points_upper_bound)]

    flattened_marker_images = []
    markers_data = []
    markers_names = []

    for fov in fovs:
        start = datetime.datetime.now()
        image, marker_data, marker_names = stitch_markers(point_name=fov, plot=False)
        end = datetime.datetime.now()

        print("Finished stitching %s in %s" % (fov, str(end - start)))

        flattened_marker_images.append(image)
        markers_data.append(marker_data)
        markers_names.append(marker_names)

    return flattened_marker_images, markers_data, markers_names


