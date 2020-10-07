import datetime
import json
import os
import random
from collections import Counter

import numpy as np
import cv2 as cv
from PIL import Image

from utils import tiff_reader, image_denoising

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def random_color():
    return tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


def read(point_name="Point15",
         plot=True,
         plot_markers=False,
         remove_noise=False,
         segmentation_type='allvessels'):
    '''
    Read the MIBI data from a single point

    :param point_name: Name of point to read
    :param plot: Should show a preview of the segmentation mask?
    :param plot_markers: Should output a description of the marker data?
    :param remove_noise: Should perform noise removal?
    :param segmentation_type: Type of segmentation mask i.e 'allvessels' etc.
    :return: Marker data and segmentation mask as numpy arrays
    '''

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

    marker_clusters = {
        "Nucleus": ["HH3"],
        "Microglia": ["CD45", "HLADR", "Iba1"],
        "Disease": ["CD47", "ABeta42", "polyubiK48", "PHFTau", "8OHGuanosine"],
        "Vessels": ["SMA", "CD31", "CollagenIV", "TrkA", "GLUT1", "Desmin", "vWF", "CD105"],
        "Astrocytes": ["S100b", "GlnSyn", "Cx30", "EAAT2", "CD44", "GFAP", "Cx43"],
        "Synapse": ["CD56", "Synaptophysin", "VAMP2", "PSD95"],
        "Oligodendrocytes": ["MOG", "MAG"],
        "Neurons": ["Calretinin", "Parvalbumin", "MAP2", "Gephyrin"]
    }

    image_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data',
                             point_name,
                             'TIFs')

    marker_names = []
    marker_images = []

    for key in marker_clusters.keys():
        for marker in marker_clusters[key]:
            path = os.path.join(image_loc, "%s.tif" % marker)

            file_name = marker

            # path = os.path.join(root, file)
            img = tiff_reader.read(path, describe=plot_markers)

            if remove_noise:
                start_denoise = datetime.datetime.now()
                img = image_denoising.knn_denoise(img)
                end_denoise = datetime.datetime.now()
                print("Finished %s in %s" % (file_name, end_denoise - start_denoise))

            if file_name not in markers_to_ignore:
                marker_images.append(img.copy())
                marker_names.append(file_name)

    markers_img = np.array(marker_images)
    segmentation_mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'masks',
                                          point_name,
                                          segmentation_type + '.tif')

    try:
        segmentation_mask = np.array(Image.open(segmentation_mask_path).convert("RGB"))
    except FileNotFoundError:
        # If there is no segmentation mask, return a blank image
        segmentation_mask = np.zeros((1024, 1024, 3), np.uint8)

    if plot:
        cv.imshow("Segmentation Mask", segmentation_mask)
        cv.waitKey(0)

    cv.imwrite(os.path.join("annotated_data/" + point_name, "total.jpg"), segmentation_mask)

    return segmentation_mask, markers_img, marker_names


def get_all_point_data(points_upper_bound=48,
                       segmentation_type='allvessels'):
    '''
    Concatenate all the point data

    :param segmentation_type:
    :param points_upper_bound: Point number upper bound
    :return: Marker data and segmentation mask as numpy arrays
    '''

    fovs = ["Point" + str(i + 1) for i in range(points_upper_bound)]

    flattened_marker_images = []
    markers_data = []
    markers_names = []

    for fov in fovs:
        start = datetime.datetime.now()
        image, marker_data, marker_names = read(point_name=fov, plot=False, segmentation_type=segmentation_type)
        end = datetime.datetime.now()

        print("Finished stitching %s in %s" % (fov, str(end - start)))

        flattened_marker_images.append(image)
        markers_data.append(marker_data)

    return flattened_marker_images, markers_data, marker_names
