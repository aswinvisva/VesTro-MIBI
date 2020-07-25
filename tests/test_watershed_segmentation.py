import unittest
import os

from image_segmentation import watershed_segmentation
from utils.stitch_markers import stitch_markers


class TestWatershedSegmentation(unittest.TestCase):

    def test_oversegmentation_watershed(self):
        image, marker_data, marker_names = stitch_markers(point_name="Point16", plot=False)

        images, usable_contours = watershed_segmentation.oversegmentation_watershed(image, show=False)

        self.assertAlmostEqual(len(usable_contours), 66)
        self.assertEqual(len(images), len(usable_contours))


if __name__ == '__main__':
    unittest.main()
