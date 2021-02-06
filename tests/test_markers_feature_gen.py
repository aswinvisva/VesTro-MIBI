import os
import unittest

from config.config_settings import Config
from src.object_extractor import ObjectExtractor
from src.markers_feature_gen import calculate_composition_marker_expression
from src.data_loading.mibi_reader import MIBIReader
from src.utils_functions import get_contour_areas_list


class TestMarkersFeatureGen(unittest.TestCase):

    def test_calculate_protein_expression(self):
        config = Config()

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

        mibi_reader = MIBIReader(config)
        object_extractor = ObjectExtractor(config)
        image, marker_data, marker_names = mibi_reader.read(data_loc, mask_loc)
        images, contours, removed_contours = object_extractor.extract(image)
        contour_areas = get_contour_areas_list(contours)

        data = calculate_composition_marker_expression(config,
                                                       marker_data,
                                                       contours,
                                                       contour_areas,
                                                       marker_names)

        self.assertEqual(len(data), len(contours))


if __name__ == '__main__':
    unittest.main()
