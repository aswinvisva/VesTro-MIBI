from collections import Counter

import numpy as np

def generate(images, vector_size=5000):
    feature_vec = []

    for image in images:
        bag_of_cells_vec = np.zeros(vector_size)

        cell_counts = Counter(image.flatten())
        keys = sorted(list(cell_counts.keys()))

        for cell in keys:
            bag_of_cells_vec[cell] = cell_counts[cell]

        feature_vec.append(bag_of_cells_vec)

    return feature_vec