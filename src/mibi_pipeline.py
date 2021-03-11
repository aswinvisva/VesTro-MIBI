import datetime
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

from src.data_analysis.base_analyzer import BaseAnalyzer
from src.data_analysis.vessel_asymmetry_analyzer import VesselAsymmetryAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.data_loading.mibi_point_contours import MIBIPointContours
from src.data_preprocessing.markers_feature_gen import *
from src.data_visualization.visualizer import Visualizer
from config.config_settings import Config

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


class MIBIPipeline:

    def __init__(self, config: Config):
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

    def analyze_data(self):
        """
        Analyze MIBI data

        :return:
        """

        for analyzer in self.analyzers:
            analyzer.analyze()

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
        all_expansions_features.index.rename(['Point', 'Vessel', 'Expansion', 'Data Type'], inplace=True)

        if self.config.save_to_csv:
            all_expansions_features.to_csv(self.config.csv_loc)

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

    def _get_outward_expansion_data(self) -> (list, list, list):
        """
        Collect outward expansion data for each expansion, for each point, for each vessel

        :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Outward microenvironment expansion data,
        list, [n_expansions, n_points, n_vessels, n_markers] -> nonvessel space expansion data,
        list, [n_expansions, n_points, n_vessels, n_markers] -> vessel space expansion data
        """

        # Store all data in lists
        expansion_data = []
        current_interval = self.config.pixel_interval
        n_points = self.config.n_points

        logging.info("Computing outward expansion data:\n")

        # Iterate through each expansion
        for x in range(self.config.max_expansions + 1):
            for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
                feed_data = self.all_feeds_contour_data.loc[feed_idx]

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

                with Pool(self.config.n_workers) as p:
                    result = p.map_async(self._get_outward_expansion_data_multiprocessing_target,
                                         map_data)
                    current_expansion_data = result.get()

                end_expression = datetime.datetime.now()

                logging.info("Finished calculating expression in %s" % (end_expression - start_expression))

                all_points_features = pd.concat(current_expansion_data).fillna(0)
                expansion_data.append(all_points_features)

            current_interval += self.config.pixel_interval

            logging.debug("Current interval %s, previous interval %s" %
                          (str(current_interval), str(current_interval - self.config.pixel_interval)))

        all_expansions_features = pd.concat(expansion_data).fillna(0)

        return all_expansions_features

    def _get_inward_expansion_data(self) -> (list, int):
        """
        Collect inward expansion data for each expansion, for each point, for each vessel

        :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Inward microenvironment expansion data,
        int, Final number of expansions needed to complete
        """

        expansion_data = []
        current_interval = self.config.pixel_interval
        current_expansion_no = 0
        n_points = self.config.n_points
        stopped_vessel_lookup = {}
        expansion_num = 0

        stopped_vessel_dict = {
            "Expansion Distance (%s)" % self.config.data_resolution_units: [],
            "# of Stopped Vessels": []
        }

        logging.info("Computing inward expansion data:\n")

        # Continue expanding inward until all vessels have been stopped
        while current_expansion_no < self.config.max_inward_expansion:
            for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
                feed_data = self.all_feeds_contour_data.loc[feed_idx]

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

        if self.config.save_to_csv:
            stopped_vessel_df.to_csv(
                os.path.join(self.config.visualization_results_dir, "inward_vessel_expansion_summary.csv"))

        return all_expansions_features, current_expansion_no

    def generate_visualizations(self):
        """
        Generate Visualizations

        :return:
        """

        assert self.visualizer is not None, "Please run preprocess_data() first!"

        expansions = self.config.expansion_to_run

        # Expression Histograms
        if self.config.create_expression_histogram:
            self.visualizer.expression_histogram()

        # Spatial Probability Maps
        if self.config.create_spatial_probability_maps:
            self.visualizer.spatial_probability_maps()

        # Marker expression overlay masks
        if self.config.create_marker_expression_overlay_masks:
            self.visualizer.marker_expression_masks()

        # Removed vessel expression box plots
        if self.config.create_removed_vessels_expression_boxplot:
            self.visualizer.removed_vessel_expression_boxplot()

        # Vessel areas histograms and boxplots
        if self.config.create_vessel_areas_histograms_and_boxplots:
            self.visualizer.vessel_areas_histogram()

        # Vessel expansion ring plots
        if self.config.create_expansion_ring_plots:
            self.visualizer.pixel_expansion_ring_plots()

        if self.config.create_biaxial_scatter_plot:
            self.visualizer.biaxial_scatter_plot()

        if self.config.create_expanded_vessel_masks:
            self.visualizer.obtain_expanded_vessel_masks()

        if self.config.create_embedded_vessel_id_masks:
            self.visualizer.obtain_embedded_vessel_masks()

        if self.config.create_vessel_asymmetry_area_spread_plot:
            self.visualizer.vessel_asymmetry_area_spread_plot()

        if self.config.create_categorical_violin_plot:
            self.visualizer.categorical_violin_plot()

        if self.config.create_categorical_scatter_plots:
            self.visualizer.continuous_scatter_plot()

        # Iterate through selected expansions to create heatmaps and line plots
        for x in expansions:

            # Brain region expansion heatmaps
            if self.config.create_brain_region_expansion_heatmaps:
                self.visualizer.brain_region_expansion_heatmap(x)

            # Categorical split expansion heatmaps
            if self.config.create_categorical_split_expansion_heatmaps:
                self.visualizer.categorical_split_expansion_heatmap_clustermap(x)

            # Per brain region line plots
            if self.config.create_brain_region_expansion_line_plots:
                self.visualizer.brain_region_plots(x)

            # Violin Plots
            if self.config.create_expansion_violin_plots:
                self.visualizer.violin_plot_brain_expansion(x)

            # All points average line plots
            if self.config.create_allpoints_expansion_line_plots:
                self.visualizer.all_points_plots(x)

            # Box Plots
            if self.config.create_expansion_box_plots:
                self.visualizer.box_plot_brain_expansions(x)

            # Mask/Non-mask heatmaps
            if self.config.create_vessel_nonvessel_heatmaps:
                self.visualizer.vessel_nonvessel_heatmap(x)

            # Vessel/Non-vessel masks
            if self.config.create_vessel_nonvessel_mask:
                self.visualizer.vessel_nonvessel_masks(x)

            # Per point line plots
            if self.config.create_point_expansion_line_plots:
                self.visualizer.point_region_plots(x)

            # Per vessel line plots
            if self.config.create_vessel_expansion_line_plots:
                self.visualizer.vessel_region_plots(x)

    def load_preprocess_data(self):
        """
        Create the visualizations for inward and outward vessel expansions and populate all results in the directory
        set in the configuration settings.
        """

        n_expansions = self.config.max_expansions
        expansions = self.config.expansion_to_run  # Expansions that you want to run

        n_expansions += 1  # Intuitively, 5 expansions means 5 expansions excluding the original composition of the
        # vessel, but we mean 5 expansions including the original composition - thus 4 expansions. Therefore lets add 1
        # so we are on the same page.

        assert n_expansions >= max(expansions), "More expansions selected than available!"

        self.all_feeds_metadata, self.all_feeds_data, self.all_feeds_mask, self.marker_names = self.mibi_loader.read()
        # Collect all marker and mask data

        self.config.display()

        # Collect vessel contours from each segmentation mask
        all_feeds_contour_data = []

        for feed_idx in range(self.all_feeds_mask.shape[0]):
            all_points_contour_data = []

            for point_idx in range(self.all_feeds_mask.shape[1]):
                mibi_contours = MIBIPointContours(self.all_feeds_mask[feed_idx, point_idx], point_idx, self.config)

                contour_df = pd.DataFrame({
                    "Contours": mibi_contours
                }, index=[(feed_idx, point_idx)])

                contour_df.index = pd.MultiIndex.from_tuples(contour_df.index,
                                                             names=("Feed Index", "Point Index"))
                all_points_contour_data.append(contour_df)

            all_points_contour_data = pd.concat(all_points_contour_data).fillna(0)
            all_feeds_contour_data.append(all_points_contour_data)

        self.all_feeds_contour_data = pd.concat(all_feeds_contour_data).fillna(0)

        # Inward expansion data
        if self.config.perform_inward_expansions:
            all_inward_expansions_features, current_expansion_no = self._get_inward_expansion_data()

            logging.debug("Finished inward expansions with a maximum of %s %s"
                          % (
                              str(
                                  current_expansion_no * self.config.pixel_interval * self.config.pixels_to_distance),
                              str(self.config.data_resolution_units)))

        # Collect outward microenvironment expansion data, nonvessel space expansion data and vessel space expansion
        # data
        all_expansions_features = self._get_outward_expansion_data()

        if self.config.perform_inward_expansions:
            all_expansions_features = all_expansions_features.append(all_inward_expansions_features)

        # Normalize all features
        self.all_expansions_features = self.normalize_data(all_expansions_features,
                                                           self.marker_names)

        self.visualizer = Visualizer(
            self.config,
            self.all_expansions_features,
            self.marker_names,
            self.all_feeds_contour_data,
            self.all_feeds_metadata,
            self.all_feeds_data
        )
