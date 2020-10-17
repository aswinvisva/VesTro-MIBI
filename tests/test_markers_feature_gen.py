import unittest

import numpy as np

from utils.extract_vessel_contours import extract
from utils.markers_feature_gen import calculate_composition_marker_expression
from utils.mibi_reader import read


class TestMarkersFeatureGen(unittest.TestCase):

    def test_calculate_protein_expression(self):
        image, marker_data, marker_names = read(point_name="Point16")
        images, contours = extract(image)

        data = calculate_composition_marker_expression(marker_data, contours)

        self.assertEqual(len(data), len(contours))
        self.assertEqual(np.array(data).shape[1], len(marker_names))


if __name__ == '__main__':
    unittest.main()
