import unittest
import os

import numpy as np

from image_segmentation.watershed_segmentation import oversegmentation_watershed
from marker_processing.flowsom_clustering import ClusteringFlowSOM
from marker_processing.markers_feature_gen import calculate_protein_expression_single_cell
from utils.stitch_markers import stitch_markers


class TestClusteringFlowSOM(unittest.TestCase):

    def test_predict(self):
        image, marker_data, marker_names = stitch_markers(point_name="Point16", plot=False)
        images, contours = oversegmentation_watershed(image, show=False)

        data = calculate_protein_expression_single_cell(marker_data, contours, plot=False)

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
