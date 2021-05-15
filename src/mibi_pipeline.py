import datetime
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *

from src.data_preprocessing.object_extractor import ObjectExtractor
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


class MIBIPipeline:

    def __init__(self,
                 config: Config,
                 results_dir: str,
                 max_inward_expansions=10,
                 max_outward_expansions=10,
                 expansions=[10],
                 n_workers=5,
                 run_async=True,
                 **kwargs):
        """
        MIBI Pipeline Class

        :param config: configuration settings
        """
        self.config = config

        self.mibi_loader = MIBILoader(self.config)

        self.visualizer = None
        self.analyzer = None
        self.analyzers = []

        self.marker_names = None
        self.all_feeds_metadata = None
        self.all_feeds_data = None
        self.all_feeds_mask = None
        self.all_feeds_contour_data = None
        self.all_expansions_features = None

        self.csv_loc = kwargs.get("csv_loc", None)
        self.results_dir = results_dir

        self.max_inward_expansions = max_inward_expansions
        self.max_outward_expansions = max_outward_expansions
        self.expansions = expansions
        self.n_workers = n_workers
        self.run_async = run_async

    def add_feed(self, data_feed: MIBIDataFeed):
        """
        Add a data feed to the pipeline

        :param data_feed: MIBIDataFeed, data feed to add
        :return:
        """

        if data_feed.name not in self.mibi_loader.feeds.keys():
            self.mibi_loader.add_feed(data_feed)
        else:
            raise Exception("Duplicate feed trying to be processed, please rename feed!")

    def add_analyzer(self, analyzer_type: BaseAnalyzer):
        """
        Add an analyzer to the pipeline

        :param analyzer_type: BaseAnalyzer, analyzer to add
        :return:
        """

        analyzer = analyzer_type(
            self.config,
            self.all_expansions_features,
            self.marker_names,
            self.all_feeds_contour_data,
            self.all_feeds_metadata,
            self.all_feeds_data
        )

        self.analyzers.append(analyzer)

    def analyze_data(self, **kwargs):
        """
        Analyze MIBI data

        :return:
        """

        for analyzer in self.analyzers:
            analyzer.analyze(self.results_dir, **kwargs)

    def save_to_csv(self, csv_name="data.csv"):
        """
        Save data to a csv
        :return:
        """

        if self.all_expansions_features is not None:
            self.all_expansions_features.to_csv(self.results_dir + csv_name)

    def _load_csv(self, csv_loc: str):
        """
        Load data from a csv

        :return:
        """

        self.all_expansions_features = pd.read_csv(csv_loc, index_col=[0, 1, 2, 3], skipinitialspace=True)

    def normalize_data(self,
                       all_expansions_features: pd.DataFrame,
                       marker_names: list):
        """
        Normalize Expansion Features

        :param marker_names: array_like, [n_markers] -> List of marker names
        :param all_expansions_features: pd.DataFrame, Expansion features
        :return: pd.DataFrame, Normalized expansion features
        """
        logging.info("Performing data normalization:\n")

        scaling_factor = self.config.scaling_factor
        transformation = self.config.transformation_type
        normalization = self.config.normalization_type
        n_markers = self.config.n_markers

        all_expansions_features = normalize_expression_data(self.config,
                                                            all_expansions_features,
                                                            marker_names,
                                                            transformation=transformation,
                                                            normalization=normalization,
                                                            scaling_factor=scaling_factor,
                                                            n_markers=n_markers)

        all_expansions_features["SMA Presence"] = ["Positive" if sma > self.config.SMA_positive_threshold
                                                   else "Negative" for sma in all_expansions_features["SMA"]]

        all_expansions_features = all_expansions_features.sort_index()
        all_expansions_features.index.rename(['Point', 'Vessel', 'Expansion', 'Expansion Type'], inplace=True)

        return all_expansions_features

    def _get_outward_expansion_data_multiprocessing_target(self, multiprocess_data: (int,
                                                                                     int,
                                                                                     int,
                                                                                     list,
                                                                                     list,
                                                                                     np.array)) -> pd.DataFrame:
        """
        Multiprocessing target for get_outward_expansion_data() method

        :param multiprocess_data: (int, int, int, list, list, np.array), Point number and expansion number to compute
        :return: pd.DataFrame, Expansion data from point
        """

        n_point = multiprocess_data[0]
        n_expansion = multiprocess_data[1]
        current_interval = multiprocess_data[2]
        feed_contours = multiprocess_data[3]
        marker_data = multiprocess_data[4]
        feed_name = multiprocess_data[5]

        contours = feed_contours.contours
        contour_areas = feed_contours.areas

        data, expression_images, stopped_vessels = calculate_microenvironment_marker_expression(
            self.config,
            marker_data,
            contours,
            contour_areas,
            self.marker_names,
            pixel_expansion_upper_bound=current_interval,
            pixel_expansion_lower_bound=current_interval - self.config.pixel_interval,
            point_num=n_point + 1,
            expansion_num=n_expansion,
            data_name=feed_name
        )

        return data

    def _get_outward_expansion_data(self,
                                    max_outward_expansions=10,
                                    n_workers=5,
                                    run_async=True) -> (list, list, list):
        """
        Collect outward expansion data for each expansion, for each point, for each vessel

        :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Outward microenvironment expansion data,
        list, [n_expansions, n_points, n_vessels, n_markers] -> nonvessel space expansion data,
        list, [n_expansions, n_points, n_vessels, n_markers] -> vessel space expansion data
        """

        # Store all data in lists
        expansion_data = []
        current_interval = self.config.pixel_interval

        logging.info("Computing outward expansion data:\n")

        # Iterate through each expansion
        for x in range(max_outward_expansions + 1):
            for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
                feed_data = self.all_feeds_contour_data.loc[feed_idx]

                n_points = len(feed_data)

                idx = pd.IndexSlice
                feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

                logging.info("Expanding Distance... : %s%s\n" % (
                    str(x * self.config.pixels_to_distance * self.config.pixel_interval),
                    self.config.data_resolution_units))

                # Data for multiprocessing map
                map_data = [(i,
                             x,
                             current_interval,
                             feed_data.loc[i, "Contours"],
                             self.all_feeds_data[feed_idx, i],
                             feed_name) for i in range(n_points)]

                start_expression = datetime.datetime.now()

                with Pool(n_workers) as p:
                    if run_async:
                        result = p.map_async(self._get_outward_expansion_data_multiprocessing_target,
                                             map_data)
                        current_expansion_data = result.get()
                    else:
                        current_expansion_data = map(self._get_outward_expansion_data_multiprocessing_target,
                                                     map_data)

                end_expression = datetime.datetime.now()

                logging.info("Finished calculating expression in %s" % (end_expression - start_expression))

                all_points_features = pd.concat(current_expansion_data).fillna(0)
                expansion_data.append(all_points_features)

            current_interval += self.config.pixel_interval

            logging.debug("Current interval %s, previous interval %s" %
                          (str(current_interval), str(current_interval - self.config.pixel_interval)))

        all_expansions_features = pd.concat(expansion_data).fillna(0)

        return all_expansions_features

    def _get_inward_expansion_data(self,
                                   max_inward_expansions=10) -> (list, int):
        """
        Collect inward expansion data for each expansion, for each point, for each vessel

        :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Inward microenvironment expansion data,
        int, Final number of expansions needed to complete
        """

        expansion_data = []
        current_interval = self.config.pixel_interval
        current_expansion_no = 0
        stopped_vessel_lookup = {}
        expansion_num = 0

        stopped_vessel_dict = {
            "Expansion Distance (%s)" % self.config.data_resolution_units: [],
            "# of Stopped Vessels": []
        }

        logging.info("Computing inward expansion data:\n")

        # Continue expanding inward until all vessels have been stopped
        while current_expansion_no < max_inward_expansions:
            for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
                feed_data = self.all_feeds_contour_data.loc[feed_idx]

                n_points = len(feed_data)

                idx = pd.IndexSlice
                feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

                logging.info("Expanding Distance... : %s%s\n" % (str(current_expansion_no
                                                                     * self.config.pixels_to_distance
                                                                     * self.config.pixel_interval),
                                                                 self.config.data_resolution_units))

                logging.debug(
                    "Current inward expansion: %s, Interval: %s" % (str(current_expansion_no), str(current_interval)))
                expansion_num -= 1

                current_expansion_data = []
                all_points_stopped_vessels = 0

                # Iterate through each point
                for point_idx in tqdm(range(n_points)):
                    contours = feed_data.loc[point_idx, "Contours"].contours
                    contour_areas = feed_data.loc[point_idx, "Contours"].areas
                    marker_data = self.all_feeds_data[feed_idx, point_idx]
                    start_expression = datetime.datetime.now()

                    data, stopped_vessels = calculate_inward_microenvironment_marker_expression(
                        self.config,
                        marker_data,
                        point_idx + 1,
                        expansion_num,
                        stopped_vessel_lookup,
                        contours,
                        contour_areas,
                        self.marker_names,
                        pixel_expansion_upper_bound=current_interval,
                        pixel_expansion_lower_bound=current_interval - self.config.pixel_interval,
                        data_name=feed_name)

                    all_points_stopped_vessels += stopped_vessels

                    if data is not None:
                        current_expansion_data.append(data)

                    end_expression = datetime.datetime.now()

                    logging.debug(
                        "Finished calculating expression for Point %s in %s" % (
                            str(point_idx + 1), end_expression - start_expression))

                if len(current_expansion_data) > 0:
                    all_points_features = pd.concat(current_expansion_data).fillna(0)
                    expansion_data.append(all_points_features)

                current_interval += self.config.pixel_interval
                current_expansion_no += 1

                stopped_vessel_dict["Expansion Distance (%s)"
                                    % self.config.data_resolution_units].append(current_expansion_no
                                                                                * self.config.pixels_to_distance
                                                                                * self.config.pixel_interval)
                stopped_vessel_dict["# of Stopped Vessels"].append(all_points_stopped_vessels)

        all_expansions_features = pd.concat(expansion_data).fillna(0)

        stopped_vessel_df = pd.DataFrame.from_dict(stopped_vessel_dict)
        logging.info("\n" + stopped_vessel_df.to_markdown())

        stopped_vessel_df.to_csv(os.path.join(self.results_dir, "inward_vessel_expansion_summary.csv"))

        return all_expansions_features, current_expansion_no

    def load_preprocess_data(self):
        """
        Create the visualizations for inward and outward vessel expansions and populate all results in the directory
        set in the configuration settings.
        """

        assert self.max_outward_expansions >= max(self.expansions), "More expansions selected than available!"

        self.all_feeds_metadata, self.all_feeds_data, self.all_feeds_mask, self.marker_names = self.mibi_loader.read()

        # Collect all marker and mask data

        self.config.display()

        # Collect vessel contours from each segmentation mask
        all_feeds_contour_data = []

        object_extractor = ObjectExtractor(self.config, self.results_dir)

        for feed_idx in range(self.all_feeds_mask.shape[0]):
            all_points_contour_data = []

            feed = self.all_feeds_metadata.loc[pd.IndexSlice[feed_idx, 0], "Feed"]

            for point_idx in range(self.all_feeds_mask.shape[1]):
                removed_contour_threshold = self.config.minimum_contour_area_to_remove / float(
                    (1.0 / feed.pixels_to_distance) ** 2)

                mibi_contours = MIBIPointContours(self.all_feeds_mask[feed_idx, point_idx],
                                                  point_idx,
                                                  self.config,
                                                  object_extractor,
                                                  removed_contour_threshold=removed_contour_threshold)

                contour_df = pd.DataFrame({
                    "Contours": mibi_contours
                }, index=[(feed_idx, point_idx)])

                contour_df.index = pd.MultiIndex.from_tuples(contour_df.index,
                                                             names=("Feed Index", "Point Index"))
                all_points_contour_data.append(contour_df)

            all_points_contour_data = pd.concat(all_points_contour_data).fillna(0)
            all_feeds_contour_data.append(all_points_contour_data)

        self.all_feeds_contour_data = pd.concat(all_feeds_contour_data).fillna(0)

        if self.csv_loc is None:
            # Inward expansion data
            if self.config.perform_inward_expansions:
                all_inward_expansions_features, current_expansion_no = self._get_inward_expansion_data(
                    max_inward_expansions=self.max_inward_expansions
                )

                logging.debug("Finished inward expansions with a maximum of %s %s"
                              % (
                                  str(
                                      current_expansion_no * self.config.pixel_interval * self.config.pixels_to_distance),
                                  str(self.config.data_resolution_units)))

            # Collect outward microenvironment expansion data, nonvessel space expansion data and vessel space expansion
            # data
            all_expansions_features = self._get_outward_expansion_data(
                max_outward_expansions=self.max_outward_expansions,
                n_workers=self.n_workers,
                run_async=self.run_async)

            if self.config.perform_inward_expansions:
                all_expansions_features = all_expansions_features.append(all_inward_expansions_features)

            # Normalize all features
            self.all_expansions_features = self.normalize_data(all_expansions_features,
                                                               self.marker_names)

        else:
            self._load_csv(self.csv_loc)

        self.visualizer = Visualizer(
            self.config,
            self.all_expansions_features,
            self.marker_names,
            self.all_feeds_contour_data,
            self.all_feeds_metadata,
            self.all_feeds_data,
            self.results_dir
        )
