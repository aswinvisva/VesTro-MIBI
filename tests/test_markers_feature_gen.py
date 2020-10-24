import os
import unittest

import numpy as np

import config.config_settings as config
from utils.extract_vessel_contours import extract
from utils.markers_feature_gen import calculate_composition_marker_expression
from utils.mibi_reader import read


class TestMarkersFeatureGen(unittest.TestCase):

    def test_calculate_protein_expression(self):
        # Get path to data selected through configuration settings
        data_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                config.data_dir,
                                "Point16",
                                config.tifs_dir)

        # Get path to mask selected through configuration settings
        mask_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                config.masks_dr,
                                "Point16",
                                "allvessels" + '.tif')

        image, marker_data, marker_names = read(data_loc, mask_loc)
        images, contours, removed_contours = extract(image)

        data = calculate_composition_marker_expression(marker_data, contours)

        self.assertEqual(len(data), len(contours))
        self.assertEqual(np.array(data).shape[1], len(marker_names))


if __name__ == '__main__':
    unittest.main()
