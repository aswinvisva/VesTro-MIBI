def vessel_nonvessel_heatmap(n_expansions: int, **kwargs):
    """
    Vessel/Non-vessel heatmaps for marker expression

    :param n_expansions: int, Number of expansions
    :return:
    """

    for feed_idx in range(self.all_feeds_data.shape[0]):
        idx = pd.IndexSlice
        feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

        feed_dir = "%s/%s" % (parent_dir, feed_name)
        mkdir_p(feed_dir)

        feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]
