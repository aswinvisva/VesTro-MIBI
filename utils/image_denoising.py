import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.stats as stats


def knn_denoise(img,
                n_neighbors=25):
    """
    Python implementation of KNN denoising used in A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging.

    :param n_neighbors:
    :param img:
    :return:
    """

    denoised = np.zeros(img.shape)
    denoised[np.where(img > 0)] = 1

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(denoised)
    distances, indices = nbrs.kneighbors(denoised)

    distance_map = distances.flatten()
    threshold = np.quantile(distance_map, 0.98)

    img[np.where(distances > threshold)] = 0

    return img
