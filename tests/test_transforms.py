import os
import tempfile
import unittest

from PIL import Image
import numpy as np
import pandas as pd

from config.config_settings import Config
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_analysis.positive_vessel_summary_analyzer import PositiveVesselSummaryAnalyzer
from src.data_analysis.shape_quantification_analyzer import ShapeQuantificationAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_preprocessing.transforms import loc_by_expansion, melt_markers
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data
from src.data_analysis._shape_quantification_metrics import *


class TestTransforms(unittest.TestCase):

    def test_loc_by_expansion(self):
        test_mibi_features = pd.read_csv("data/dummy_test_data.csv",
                                         index_col=[0, 1, 2, 3],
                                         skipinitialspace=True)

        transformed_features_mask_only = loc_by_expansion(test_mibi_features,
                                                          expansion_type="mask_only",
                                                          average=False)

        self.assertTrue((transformed_features_mask_only.index.get_level_values('Expansion') < 0).all())

        transformed_features_expansion_only = loc_by_expansion(test_mibi_features,
                                                               expansion_type="expansion_only",
                                                               average=False)

        self.assertTrue((transformed_features_expansion_only.index.get_level_values('Expansion') >= 0).all())

        transformed_features_average = loc_by_expansion(test_mibi_features,
                                                        expansion_type="mask_only",
                                                        average=True)

        self.assertTrue(transformed_features_expansion_only.shape[0] > transformed_features_average.shape[0])

    def test_melt_markers(self):
        test_mibi_features = pd.read_csv("data/dummy_test_data.csv",
                                         index_col=[0, 1, 2, 3],
                                         skipinitialspace=True)

        transformed_features = melt_markers(test_mibi_features,
                                            id_vars=["Contour Area", "Vessel Size", "Data Type", "SMA Presence"])

        markers = np.setdiff1d(test_mibi_features.columns,
                               ["Contour Area", "Vessel Size", "Data Type", "SMA Presence"])

        self.assertEqual(test_mibi_features.shape[0] * len(markers), transformed_features.shape[0])
