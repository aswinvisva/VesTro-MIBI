import unittest
import os

from image_segmentation import extract_cell_events
from utils.mibi_reader import read


class TestWatershedSegmentation(unittest.TestCase):

    def test_oversegmentation_watershed(self):
        image, marker_data, marker_names = read(point_name="Point16", plot=False)

        images, usable_contours = extract_cell_events.extract_cell_contours(image, show=False)

        self.assertAlmostEqual(len(usable_contours), 36)
        self.assertEqual(len(images), len(usable_contours))


if __name__ == '__main__':
    unittest.main()
