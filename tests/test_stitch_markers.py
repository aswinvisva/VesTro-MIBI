import unittest
import os

from utils import mibi_reader


class TestStitchMarkersMethods(unittest.TestCase):

    def test_stitch_markers(self):
        segmentation_mask, markers_img, marker_names = mibi_reader.read(point_name="Point16", plot=False)

        # Ensure the correct number of markers
        self.assertEqual(len(markers_img), len(marker_names))

        self.assertEqual(segmentation_mask.shape[0], markers_img.shape[1])
        self.assertEqual(segmentation_mask.shape[1], markers_img.shape[2])

        self.assertIsNotNone(markers_img)
        self.assertIsNotNone(segmentation_mask)
        self.assertIsNotNone(marker_names)

    def test_concatenate_multiple_markers(self):
        flattened_marker_images, markers_data, markers_names = mibi_reader.get_all_point_data()

        self.assertEqual(len(flattened_marker_images), len(markers_data))


if __name__ == '__main__':
    unittest.main()
