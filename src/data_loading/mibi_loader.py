from tqdm import tqdm

from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_reader import MIBIReader
from src.data_preprocessing.markers_feature_gen import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


class MIBILoader:

    def __init__(self, config: Config):
        """
        MIBI loader for loading data from multiple data feeds
        """

        self.feeds = {}
        self.mibi_reader = MIBIReader(config)

    def add_feed(self, feed: MIBIDataFeed):
        """
        Add a data feed to the loader

        :param feed: MIBIDataFeed, data feed to add
        """

        self.feeds[feed.name] = feed

    def _get_point_data(self, point_num: int, feed_name: str) -> dict:
        """
        Get data for a given point

        :param feed_name: str, Feed to get data from
        :param point_num: int, Point number
        """

        data_loc, mask_loc = self.feeds[feed_name].get_locs(point_num)
        segmentation_mask, markers_img, marker_names = self.mibi_reader.read(data_loc, mask_loc)

        metadata_dict = {
            "Feed Name": [feed_name],
            "Segmentation Mask Size": [segmentation_mask.shape],
            "Number of Markers": [len(marker_names)]
        }

        return metadata_dict, markers_img, segmentation_mask, marker_names

    def read(self):
        """
        Get all data from loader

        :return:
        """
        all_feeds_metadata = []
        all_feeds_data = []
        all_feeds_mask = []

        feed_idx = 0

        logging.info("Loading data feeds\n")

        for feed_name in self.feeds.keys():
            metadata_per_feed = []
            feed_data = []
            feed_mask = []

            for i in tqdm(range(self.feeds[feed_name].total_points)):
                metadata_dict, markers_img, segmentation_mask, marker_names = self._get_point_data(i + 1, feed_name)

                metadata_df = pd.DataFrame(metadata_dict, index=[(feed_idx, i)])
                metadata_per_feed.append(metadata_df)
                feed_data.append(markers_img)
                feed_mask.append(segmentation_mask)

            metadata_per_feed = pd.concat(metadata_per_feed).fillna(0)
            all_feeds_metadata.append(metadata_per_feed)
            all_feeds_data.append(np.array(feed_data))
            all_feeds_mask.append(np.array(feed_mask))

            feed_idx += 1

        all_feeds_metadata = pd.concat(all_feeds_metadata).fillna(0)
        all_feeds_data = np.array(all_feeds_data)
        all_feeds_mask = np.array(all_feeds_mask)

        all_feeds_metadata.index = pd.MultiIndex.from_tuples(all_feeds_metadata.index,
                                                             names=("Feed Index", "Point Index"))

        return all_feeds_metadata, all_feeds_data, all_feeds_mask, marker_names
