import os
import tempfile
import unittest

from PIL import Image
import numpy as np

from config.config_settings import Config
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data, generate_random_mask


class TestMIBIPointContours(unittest.TestCase):
    def test_init(self):
        conf = Config()
        mask = generate_random_mask((2048, 2048, 1))
        mibi_contours = MIBIPointContours(mask, 1, conf)

        self.assertIsNotNone(mibi_contours.contours)
        self.assertIsNotNone(mibi_contours.removed_contours)
        self.assertIsNotNone(mibi_contours.removed_areas)
