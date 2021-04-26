import os
import tempfile
import unittest

from PIL import Image
import numpy as np

from config.config_settings import Config
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.utils.test_utils import create_test_data


class TestMIBILoader(unittest.TestCase):

    def test_add_feed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=1, resolution=(2048, 2048))

            config = Config()
            loader = MIBILoader(config)
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=1
            )

            loader.add_feed(example_feed)

            self.assertEqual(loader.feeds[example_feed.name], example_feed)

    def test_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=1, resolution=(2048, 2048))

            config = Config()
            loader = MIBILoader(config)
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=1
            )

            loader.add_feed(example_feed)

            all_feeds_metadata, all_feeds_data, all_feeds_mask, marker_names = loader.read()

            self.assertEqual(len(all_feeds_data), 1)


if __name__ == '__main__':
    unittest.main()
