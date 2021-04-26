import unittest
import os

from config.config_settings import Config
from src.data_preprocessing.object_extractor import ObjectExtractor
from src.data_loading.mibi_reader import MIBIReader


class TestWatershedSegmentation(unittest.TestCase):

    def test_oversegmentation_watershed(self):
        config = Config()
        mibi_reader = MIBIReader(config)
        object_extractor = ObjectExtractor(config)

        # Get path to data selected through configuration settings
        data_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                config.data_dir,
                                "Point16",
                                config.tifs_dir)

        # Get path to mask selected through configuration settings
        mask_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                config.masks_dir,
                                "Point16",
                                "allvessels" + '.tif')

        image, marker_data, marker_names = mibi_reader.read(data_loc, mask_loc)

        images, usable_contours, removed_contours = object_extractor.extract(image)

        self.assertAlmostEqual(len(usable_contours), 56)
        self.assertEqual(len(images), len(usable_contours))


if __name__ == '__main__':
    unittest.main()
