import unittest

from utils.extract_vessel_contours import extract
from dnn_vessel_heterogeneity.models.flowsom_clustering import ClusteringFlowSOM
from utils.markers_feature_gen import calculate_composition_marker_expression
from utils.mibi_reader import read


class TestClusteringFlowSOM(unittest.TestCase):

    def test_predict(self):
        image, marker_data, marker_names = read(point_name="Point16")
        images, contours = extract(image)

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
