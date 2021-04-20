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


class TestMIBIDataFeed(unittest.TestCase):
    def test_get_locs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            data_loc, mask_loc = example_feed.get_locs(1)

            self.assertIsNotNone(data_loc)
            self.assertIsNotNone(mask_loc)
