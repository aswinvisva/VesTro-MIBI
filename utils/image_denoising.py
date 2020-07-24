import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial import cKDTree


def _euclidean_distance(x1, x2):
    return np.linalg.norm(np.asarray(x1) - np.asarray(x2))


def _get_neighbors(x1, matrix, img, n_neighbors):
    distances = []

    for x in range(img[x1]):
        distances.append((x1, 0))  # Treat a marker count as its own neighbor with a distance of 0

    for i in range(len(matrix)):
        x2 = matrix[i]

        if x1 == x2:
            continue

        dist = _euclidean_distance(x1, x2)
        distances.append((x2, dist))

    distances.sort(key=lambda tup: tup[1])

    neighbors = [distances[i][0] for i in range(n_neighbors)]
    distances = [distances[i][1] for i in range(n_neighbors)]

    return neighbors, distances


def knn_denoise(img,
                n_neighbors=10,
                q_threshold=0.5):
    """
    Python implementation of KNN denoising used in the research paper -
    "A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam
    Imaging."

    :param q_threshold: Distance quantile threshold to consider marker as noise
    :param n_neighbors: Number of neighbors to use in KNN search
    :param img: Marker data to be denoised
    :return:
    """

    indices = np.where(img > 0)  # Get indices of marker data with positive counts

    positive_indices = list(zip(*indices))

    neigh_dist = []
    neigh_ind = []

    tree = cKDTree(positive_indices)  # Use KD Tree algorithm to compute nearest neighbors

    for x1 in positive_indices:
        _dist, _neigh = tree.query(x1, k=n_neighbors)  # Query the tree to get nearest neighbors of each point

        neigh_ind.append(_neigh)
        neigh_dist.append(_dist)

    distances = np.array(neigh_dist)
    flat_dist = distances.flatten()
    threshold = np.quantile(flat_dist, q=q_threshold)  # Get distance at given quantile threshold
    img[np.where(img > threshold)] = 0  # Remove marker points with distances greater than the threshold

    return img
