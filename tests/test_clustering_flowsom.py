import os
import unittest

import config.config_settings as config
from utils.extract_vessel_contours import extract
from dnn_vessel_heterogeneity.models.flowsom_clustering import ClusteringFlowSOM
from utils.markers_feature_gen import calculate_composition_marker_expression
from utils.mibi_reader import read


class TestClusteringFlowSOM(unittest.TestCase):

    def test_predict(self):
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
