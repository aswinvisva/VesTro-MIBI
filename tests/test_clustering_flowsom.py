import unittest
import os

import numpy as np

from image_segmentation.extract_cell_events import extract_cell_contours
from models.flowsom_clustering import ClusteringFlowSOM
from feature_extraction.markers_feature_gen import calculate_composition_marker_expression
from utils.mibi_reader import read


class TestClusteringFlowSOM(unittest.TestCase):

    def test_predict(self):
        image, marker_data, marker_names = read(point_name="Point16", plot=False)
        images, contours = extract_cell_contours(image, show=False)

        data = calculate_composition_marker_expression(marker_data, contours, plot=False)

        model = ClusteringFlowSOM(data,
                                  "Point16",
                                  marker_names,
                                  clusters=25,
                                  pretrained=False,
                                  show_plots=False,
                                  save=False)
        model.fit_model()
        indices, cell_counts = model.predict()

        self.assertEqual(len(indices), len(contours))
        self.assertEqual(len(data), len(indices))


if __name__ == '__main__':
    unittest.main()
