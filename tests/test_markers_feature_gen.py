import os
import random
import tempfile
import unittest

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from config.config_settings import Config
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_analysis.positive_vessel_summary_analyzer import PositiveVesselSummaryAnalyzer
from src.data_analysis.shape_quantification_analyzer import ShapeQuantificationAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_preprocessing.markers_feature_gen import contract_vessel_region, expand_vessel_region, \
    normalize_expression_data, arcsinh, preprocess_marker_data
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data
from src.data_analysis._shape_quantification_metrics import *


class TestMarkersFeatureGen(unittest.TestCase):

    def test_expand_vessel_region(self):
        points = np.array([[25, 25], [70, 10], [150, 50], [250, 250], [100, 350]])
        ring = expand_vessel_region(points, (500, 500), upper_bound=10, lower_bound=0)
        test_ring = cv.imread("data/example_contour_expanded.png")[:, :, 0]/255

        assert_array_equal(ring, test_ring)

    def test_contract_vessel_region(self):
        points = np.array([[25, 25], [70, 10], [150, 50], [250, 250], [100, 350]])
        ring = contract_vessel_region(points, (500, 500), upper_bound=10, lower_bound=0)

        test_ring = cv.imread("data/example_contour_contracted.png")[:, :, 0]/255

        assert_array_equal(ring, test_ring)

    def test_normalize_expression_data(self):
        config = Config()

        marker_names = ["HH3",
                        "CD45",
                        "HLADR",
                        "Iba1",
                        "CD47",
                        "ABeta42",
                        "polyubiK48",
                        "PHFTau",
                        "8OHGuanosine",
                        "SMA",
                        "CD31",
                        "CollagenIV",
                        "TrkA",
                        "GLUT1",
                        "Desmin",
                        "vWF",
                        "CD105",
                        "S100b",
                        "GlnSyn",
                        "Cx30",
                        "EAAT2",
                        "CD44",
                        "GFAP",
                        "Cx43",
                        "CD56",
                        "Synaptophysin",
                        "VAMP2",
                        "PSD95",
                        "MOG",
                        "MAG",
                        "Calretinin",
                        "Parvalbumin",
                        "MAP2",
                        "Gephyrin",
                        ]

        test_raw_data = pd.read_csv("data/dummy_test_data_unnormalized.csv", index_col=[0, 1, 2, 3],
                                    skipinitialspace=True)

        scaling_factor = config.scaling_factor
        transformation = config.transformation_type
        normalization = config.normalization_type
        n_markers = len(marker_names)

        hh3_data = test_raw_data["HH3"].to_numpy()
        hh3_data = hh3_data * scaling_factor
        hh3_data = arcsinh(hh3_data)
        percentile_99 = np.percentile(hh3_data, 99, axis=0)
        normalized_hh3 = hh3_data / percentile_99

        all_expansions_features = normalize_expression_data(config,
                                                            test_raw_data,
                                                            marker_names,
                                                            transformation=transformation,
                                                            normalization=normalization,
                                                            scaling_factor=scaling_factor,
                                                            n_markers=n_markers)

        test_normalized_hh3 = all_expansions_features["HH3"].to_numpy()

        assert_array_equal(normalized_hh3, test_normalized_hh3)

    def test_preprocess_marker_data(self):
        points = np.array([[25, 25], [70, 10], [150, 50], [250, 250], [100, 350]])
        ring = contract_vessel_region(points, (500, 500), upper_bound=10, lower_bound=0)

        np.random.seed(42)
        marker = np.random.uniform(0, 1, (500, 500))

        result = cv.bitwise_and(marker, marker, mask=ring)
        marker_data = preprocess_marker_data(result,
                                             ring,
                                             expression_type="area_normalized_counts")

        self.assertAlmostEqual(marker_data, 0.5, 3)
