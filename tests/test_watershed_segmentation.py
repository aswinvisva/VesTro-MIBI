import unittest
import os

import config.config_settings as config
from utils.extract_vessel_contours import extract
from utils.mibi_reader import read


class TestWatershedSegmentation(unittest.TestCase):

    def test_oversegmentation_watershed(self):
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

        images, usable_contours, removed_contours = extract(image)

        self.assertAlmostEqual(len(usable_contours), 56)
        self.assertEqual(len(images), len(usable_contours))


if __name__ == '__main__':
    unittest.main()
