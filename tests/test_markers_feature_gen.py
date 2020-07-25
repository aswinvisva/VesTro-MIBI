import unittest
import os

import numpy as np

from image_segmentation.watershed_segmentation import oversegmentation_watershed
from marker_processing.markers_feature_gen import calculate_protein_expression_single_cell
from utils.stitch_markers import stitch_markers


class TestMarkersFeatureGen(unittest.TestCase):

    def test_calculate_protein_expression(self):
        image, marker_data, marker_names = stitch_markers(point_name="Point16", plot=False)
        images, contours = oversegmentation_watershed(image, show=False)

        data = calculate_protein_expression_single_cell(marker_data, contours, plot=False)

        self.assertEqual(len(data), len(contours))
        self.assertEqual(np.array(data).shape[1], len(marker_names))


if __name__ == '__main__':
    unittest.main()
