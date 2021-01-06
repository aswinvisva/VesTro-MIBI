import unittest
import os

from config.config_settings import Config
from utils.mibi_reader import MIBIReader


class TestStitchMarkersMethods(unittest.TestCase):

    def test_stitch_markers(self):
        config = Config()
        mibi_reader = MIBIReader(config)

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

        segmentation_mask, markers_img, marker_names = mibi_reader.read(data_loc, mask_loc)

        # Ensure the correct number of markers
        self.assertEqual(len(markers_img), len(marker_names))

        self.assertEqual(segmentation_mask.shape[0], markers_img.shape[1])
        self.assertEqual(segmentation_mask.shape[1], markers_img.shape[2])

        self.assertIsNotNone(markers_img)
        self.assertIsNotNone(segmentation_mask)
        self.assertIsNotNone(marker_names)

    def test_concatenate_multiple_markers(self):
        config = Config()
        mibi_reader = MIBIReader(config)
        flattened_marker_images, markers_data, markers_names = mibi_reader.get_all_point_data()

        self.assertEqual(len(flattened_marker_images), len(markers_data))


if __name__ == '__main__':
    unittest.main()
