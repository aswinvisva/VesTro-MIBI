import unittest

from config.config_settings import Config
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader


class TestMIBILoader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = Config()
        self.loader = MIBILoader(self.config)
        self.example_feed = MIBIDataFeed(
            feed_data_loc="/media/large_storage/oliveria_data/data/hires",
            feed_mask_loc="/media/large_storage/oliveria_data/masks/hires",
            feed_name="Hires",
            n_points=48
        )

    def test_add_feed(self):
        self.loader.add_feed(self.example_feed)

        self.assertEqual(self.loader.feeds[self.example_feed.name], self.example_feed)

    def test_read(self):
        self.loader.add_feed(self.example_feed)

        all_feeds_metadata, all_feeds_data, all_feeds_mask, marker_names = self.loader.read()

        self.assertEqual(len(all_feeds_data), 1)


if __name__ == '__main__':
    unittest.main()
