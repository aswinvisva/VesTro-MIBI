import unittest
import os

from config.config_settings import Config
from src import tiff_reader


class TestTiffReader(unittest.TestCase):

    def test_read(self):
        config = Config()
        path = os.path.join(config.data_dir,
                            "Point16",
                            'TIFs',
                            'ABeta42.tif'
                            )

        self.assertIsNotNone(tiff_reader.read(path))
        self.assertNotEqual(tiff_reader.read(path).shape[0], 0)
        self.assertNotEqual(tiff_reader.read(path).shape[1], 0)


if __name__ == '__main__':
    unittest.main()
