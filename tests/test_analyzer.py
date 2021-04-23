import os
import tempfile
import unittest

from PIL import Image
import numpy as np
import pandas as pd

from config.config_settings import Config
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_analysis.positive_vessel_summary_analyzer import PositiveVesselSummaryAnalyzer
from src.data_analysis.vessel_asymmetry_analyzer import VesselAsymmetryAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data
from src.data_analysis._shape_quantification_metrics import *


class TestAnalyzer(unittest.TestCase):
    def test_vessel_asymmetry_analyzer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )

            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = VesselAsymmetryAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir,
                             mask_type="expansion_only",
                             marker_settings="all_markers",
                             shape_quantification_method={
                                 "Name": "Solidity",
                                 "Metric": solidity
                             },
                             img_shape=(2048, 2048))

            self.assertIn("Solidity", pipe.all_expansions_features.columns)

    def test_dimensionality_reduction_clustering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = DimensionalityReductionClusteringAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir)

            self.assertIn("UMAP0", pipe.all_expansions_features.columns)
            self.assertIn("UMAP1", pipe.all_expansions_features.columns)
            self.assertIn("K-Means", pipe.all_expansions_features.columns)
            self.assertIn("Hierarchical", pipe.all_expansions_features.columns)

    def test_positive_vessel_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = PositiveVesselSummaryAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir)

            self.assertTrue(os.path.isfile(os.path.join(temp_dir,
                                                        "vessel_positive_proportion.csv")))
