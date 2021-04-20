import os
import tempfile
import unittest

from PIL import Image
import numpy as np

from config.config_settings import Config
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data


class TestMIBIPipeline(unittest.TestCase):

    def test_add_feed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=1, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=1
            )

            pipe = MIBIPipeline(config)
            pipe.add_feed(example_feed)

            self.assertEqual(pipe.mibi_loader.feeds[example_feed.name], example_feed)

    def test_load_preprocess_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config)
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data(max_inward_expansions=1,
                                      max_outward_expansions=1,
                                      expansions=[1])

            self.assertIsNotNone(pipe.all_expansions_features)
            self.assertIsNotNone(pipe.visualizer)

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config,
                                csv_loc="data/dummy_test_data.csv"
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data(max_inward_expansions=1,
                                      max_outward_expansions=1,
                                      expansions=[1])

            self.assertIsNotNone(pipe.all_expansions_features)
            self.assertIsNotNone(pipe.visualizer)
