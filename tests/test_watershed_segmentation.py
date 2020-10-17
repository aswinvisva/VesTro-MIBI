import unittest
import os

from utils.extract_vessel_contours import extract
from utils.mibi_reader import read


class TestWatershedSegmentation(unittest.TestCase):

    def test_oversegmentation_watershed(self):
        image, marker_data, marker_names = read(point_name="Point16")

        images, usable_contours = extract(image)

        self.assertAlmostEqual(len(usable_contours), 56)
        self.assertEqual(len(images), len(usable_contours))


if __name__ == '__main__':
    unittest.main()
