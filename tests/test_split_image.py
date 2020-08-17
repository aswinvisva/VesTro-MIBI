import unittest
import os

import numpy as np

from image_segmentation import watershed_segmentation
from image_segmentation.sliding_window_segmentation import split_image
from utils.mibi_reader import read


class TestSplitImage(unittest.TestCase):

    def test_split(self):
        image, marker_data, marker_names = read(point_name="Point16", plot=False)

        segmented_images = split_image(image, n=256)

        self.assertEqual(len(segmented_images), (image.shape[0]/256)**2)


if __name__ == '__main__':
    unittest.main()
