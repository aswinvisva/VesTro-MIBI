import logging
from tqdm import tqdm

from utils.object_extractor import ObjectExtractor

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from utils.mibi_reader import MIBIReader
from utils.object_extractor import ObjectExtractor
from utils.markers_feature_gen import *
from utils.utils_functions import get_contour_areas_list
from utils.visualizer import Visualizer
from config.config_settings import Config

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
        self.mibi_reader = MIBIReader(self.config)
        self.object_extractor = ObjectExtractor(self.config)
        self.visualizer = None

    def normalize_data(self,
                       all_expansions_features: pd.DataFrame,
                       markers_names: list):
        """
        Normalize Expansion Features

        :param markers_names: array_like, [n_markers] -> List of marker names
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
                                                            markers_names,
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

    def _get_outward_expansion_data(self,
                                    all_points_vessel_contours: list,
                                    all_points_vessel_contours_areas: list,
                                    all_points_marker_data: list,
                                    marker_names: list,
                                    pixel_interval: int,
                                    n_expansions: int) -> (list, list, list):
        """
        Collect outward expansion data for each expansion, for each point, for each vessel

        :param all_points_vessel_contours_areas: list -> Vessel contour areas
        :param marker_names: list -> Marker names
        :param n_expansions: int -> Number of expansions to run
        :param pixel_interval: int -> Pixel interval
        :param all_points_vessel_contours: array_like, [n_points, n_vessels] -> list of vessel contours for each point
        :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]]
        -> list of marker data for each point

        :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Outward microenvironment expansion data,
        list, [n_expansions, n_points, n_vessels, n_markers] -> nonvessel space expansion data,
        list, [n_expansions, n_points, n_vessels, n_markers] -> vessel space expansion data
        """

        # Store all data in lists
        expansion_data = []
        current_interval = pixel_interval
        n_points = self.config.n_points

        logging.info("Computing outward expansion data:\n")

        # Iterate through each expansion
        for x in range(n_expansions):

            logging.info("Expanding Distance... : %s%s\n" % (
                str(x * self.config.pixels_to_distance * self.config.pixel_interval),
                self.config.data_resolution_units))

            current_expansion_data = []

            all_points_stopped_vessels = 0

            # Iterate through each point
            for i in tqdm(range(n_points)):
                contours = all_points_vessel_contours[i]
                contour_areas = all_points_vessel_contours_areas[i]
                marker_data = all_points_marker_data[i]
                start_expression = datetime.datetime.now()

                # If we are on the first expansion, calculate the marker expression within the vessel itself. Otherwise,
                # calculate the marker expression in the outward microenvironment

                if x == 0:
                    data = calculate_composition_marker_expression(
                        self.config,
                        marker_data,
                        contours,
                        contour_areas,
                        marker_names,
                        point_num=i + 1)
                else:
                    data, expression_images, stopped_vessels = calculate_microenvironment_marker_expression(
                        self.config,
                        marker_data,
                        contours,
                        contour_areas,
                        marker_names,
                        pixel_expansion_upper_bound=current_interval,
                        pixel_expansion_lower_bound=current_interval - pixel_interval,
                        point_num=i + 1,
                        expansion_num=x)

                    all_points_stopped_vessels += stopped_vessels

                end_expression = datetime.datetime.now()

                logging.debug(
                    "Finished calculating expression for Point %s in %s" % (
                        str(i + 1), end_expression - start_expression))

                current_expansion_data.append(data)

            logging.debug("There were %s vessels which could not expand inward/outward by %s pixels" % (
                all_points_stopped_vessels, x * pixel_interval))

            all_points_features = pd.concat(current_expansion_data).fillna(0)
            expansion_data.append(all_points_features)

            if x != 0:
                current_interval += pixel_interval

            logging.debug("Current interval %s, previous interval %s" % (str(current_interval), str(current_interval -
                                                                                                    pixel_interval)))
        all_expansions_features = pd.concat(expansion_data).fillna(0)

        return all_expansions_features

    def _get_inward_expansion_data(self,
                                   all_points_vessel_contours: list,
                                   all_points_vessel_contours_areas: list,
                                   all_points_marker_data: list,
                                   markers_names: list) -> (list, int):
        """
        Collect inward expansion data for each expansion, for each point, for each vessel

        :param all_points_vessel_contours_areas: list -> Vessel contour areas
        :param markers_names: array_like, [n_markers] -> List of marker names
        :param all_points_vessel_contours: array_like, [n_points, n_vessels] -> list of vessel contours for each point
        :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]]
        -> list of marker data for each point

        :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Inward microenvironment expansion data,
        int, Final number of expansions needed to complete
        """

        expansion_data = []
        current_interval = self.config.pixel_interval
        all_vessels_count = len([item for sublist in all_points_vessel_contours for item in sublist])
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
                contours = all_points_vessel_contours[point_idx]
                contour_areas = all_points_vessel_contours_areas[point_idx]
                marker_data = all_points_marker_data[point_idx]
                start_expression = datetime.datetime.now()

                data, stopped_vessels = calculate_inward_microenvironment_marker_expression(
                    self.config,
                    marker_data,
                    point_idx + 1,
                    expansion_num,
                    stopped_vessel_lookup,
                    contours,
                    contour_areas,
                    markers_names,
                    pixel_expansion_upper_bound=current_interval,
                    pixel_expansion_lower_bound=current_interval - self.config.pixel_interval)

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

            logging.debug(
                "There are %s / %s vessels which have failed to expand inward" % (str(all_points_stopped_vessels),
                                                                                  str(all_vessels_count)))

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

        # Iterate through selected expansions to create heatmaps and line plots
        for x in expansions:

            # Brain region expansion heatmaps
            if self.config.create_brain_region_expansion_heatmaps:
                self.visualizer.brain_region_expansion_heatmap(x + 1)

            # Violin Plots
            if self.config.create_expansion_violin_plots:
                self.visualizer.violin_plot_brain_expansion(x + 1)

            # Vessel/Non-vessel masks
            if self.config.create_vessel_nonvessel_mask:
                self.visualizer.vessel_nonvessel_masks(x + 1)

            # Mask/Non-mask heatmaps
            if self.config.create_vessel_nonvessel_heatmaps:
                self.visualizer.vessel_nonvessel_heatmap(x + 1)

            # Per brain region line plots
            if self.config.create_brain_region_expansion_line_plots:
                self.visualizer.brain_region_plots(x + 1)

            # Per point line plots
            if self.config.create_point_expansion_line_plots:
                self.visualizer.point_region_plots(x + 1)

            # Per vessel line plots
            if self.config.create_vessel_expansion_line_plots:
                self.visualizer.vessel_region_plots(x + 1)

            # All points average line plots
            if self.config.create_allpoints_expansion_line_plots:
                self.visualizer.all_points_plots(x + 1)

    def load_preprocess_data(self):
        """
        Create the visualizations for inward and outward vessel expansions and populate all results in the directory
        set in the configuration settings.
        """

        self.config.display()

        n_expansions = self.config.max_expansions
        interval = self.config.pixel_interval
        expansions = self.config.expansion_to_run  # Expansions that you want to run

        n_expansions += 1  # Intuitively, 5 expansions means 5 expansions excluding the original composition of the
        # vessel, but we mean 5 expansions including the original composition - thus 4 expansions. Therefore lets add 1
        # so we are on the same page.

        assert n_expansions >= max(expansions), "More expansions selected than available!"

        all_points_segmentation_masks, all_points_marker_data, markers_names = self.mibi_reader.get_all_point_data()  #
        # Collect all marker and mask data

        all_points_vessel_contours = []
        all_points_vessel_regions_of_interest = []
        all_points_removed_vessel_contours = []
        all_points_vessel_contours_areas = []

        # Collect vessel contours from each segmentation mask
        for point_idx, segmentation_mask in enumerate(all_points_segmentation_masks):
            vessel_regions_of_interest, contours, removed_contours = self.object_extractor.extract(segmentation_mask,
                                                                                                   point_name=str(
                                                                                                       point_idx + 1))
            all_points_vessel_contours.append(contours)
            all_points_vessel_contours_areas.append(get_contour_areas_list(contours))
            all_points_vessel_regions_of_interest.append(vessel_regions_of_interest)
            all_points_removed_vessel_contours.append(removed_contours)

        # Inward expansion data
        if self.config.perform_inward_expansions:
            all_inward_expansions_features, current_expansion_no = self._get_inward_expansion_data(
                all_points_vessel_contours,
                all_points_vessel_contours_areas,
                all_points_marker_data,
                markers_names)

            logging.debug("Finished inward expansions with a maximum of %s %s"
                          % (
                              str(
                                  current_expansion_no * self.config.pixel_interval * self.config.pixels_to_distance),
                              str(self.config.data_resolution_units)))

        # Collect outward microenvironment expansion data, nonvessel space expansion data and vessel space expansion
        # data
        all_expansions_features = self._get_outward_expansion_data(all_points_vessel_contours,
                                                                   all_points_vessel_contours_areas,
                                                                   all_points_marker_data,
                                                                   markers_names,
                                                                   interval,
                                                                   n_expansions)

        if self.config.perform_inward_expansions:
            all_expansions_features = all_expansions_features.append(all_inward_expansions_features)

        # Normalize all features
        all_expansions_features = self.normalize_data(all_expansions_features,
                                                      markers_names)

        self.visualizer = Visualizer(
            self.config,
            all_expansions_features,
            markers_names,
            all_points_vessel_contours,
            all_points_removed_vessel_contours,
            all_points_vessel_contours_areas,
            all_points_marker_data
        )
