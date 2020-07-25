import unittest
import os

from utils import tiff_reader


class TestTiffReader(unittest.TestCase):

    def test_read(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data',
                            "Point16",
                            'TIFs',
                            'ABeta42.tif'
                            )

        self.assertIsNotNone(tiff_reader.read(path))
        self.assertNotEqual(tiff_reader.read(path).shape[0], 0)
        self.assertNotEqual(tiff_reader.read(path).shape[1], 0)


if __name__ == '__main__':
    unittest.main()
