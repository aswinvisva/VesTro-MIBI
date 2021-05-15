import pandas as pd

from src.utils.utils_functions import mkdir_p


def feed_features_iterator(mibi_features: pd.DataFrame,
                           all_feeds_data: pd.DataFrame,
                           all_feeds_contour_data: pd.DataFrame,
                           all_feeds_metadata: pd.DataFrame,
                           save_to_dir: bool = False,
                           parent_dir: str = None):
    """
    Feed Features Iterator
    """

    for feed_idx in range(all_feeds_data.shape[0]):
        idx = pd.IndexSlice
        feed_name = all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]
        feed_features = mibi_features.loc[mibi_features["Data Type"] == feed_name]
        feed = all_feeds_metadata.loc[idx[feed_idx, 0], "Feed"]
        feed_contours = all_feeds_contour_data.loc[feed_idx]

        if save_to_dir:
            assert parent_dir is not None, "No directory specified!"

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            yield feed_idx, feed, feed_features, feed_contours, feed_dir
        else:
            yield feed_idx, feed, feed_features, feed_contours
