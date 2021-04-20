import tempfile
import unittest
import os

from config.config_settings import Config
from src.data_loading import tiff_reader
from src.utils.test_utils import create_test_data


class TestTiffReader(unittest.TestCase):

    def test_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=1, resolution=(2048, 2048))
            path = os.path.join(temp_dir,
                                "data",
                                "Point1",
                                'TIFs',
                                'ABeta42.tif'
                                )

            self.assertIsNotNone(tiff_reader.read(path))
            self.assertNotEqual(tiff_reader.read(path).shape[0], 0)
            self.assertNotEqual(tiff_reader.read(path).shape[1], 0)


if __name__ == '__main__':
    unittest.main()
