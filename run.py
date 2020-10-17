from utils.mibi_reader import get_all_point_data
from utils.extract_vessel_contours import *
from utils.markers_feature_gen import *
from utils.visualizer import vessel_nonvessel_heatmap, point_region_plots, vessel_region_plots, brain_region_plots, \
    all_points_plots, brain_region_expansion_heatmap, marker_expression_masks, vessel_areas_histogram, \
    pixel_expansion_ring_plots
import config.config_settings as config

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


def get_outward_expansion_data(all_points_vessel_contours: list,
                               all_points_marker_data: list,
                               pixel_interval: int,
                               n_expansions: int) -> (list, list, list):
    """
    Collect outward expansion data for each expansion, for each point, for each vessel

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
    dark_space_expansion_data = []
    vessel_space_expansion_data = []
    current_interval = pixel_interval
    n_points = config.n_points

    # Iterate through each expansion
    for x in range(n_expansions):

        current_expansion_data = []
        current_dark_space_expansion_data = []
        current_vessel_space_expansion_data = []

        all_points_stopped_vessels = 0

        # Iterate through each point
        for i in range(n_points):
            contours = all_points_vessel_contours[i]
            marker_data = all_points_marker_data[i]
            start_expression = datetime.datetime.now()

            # If we are on the first expansion, calculate the marker expression within the vessel itself. Otherwise,
            # calculate the marker expression in the outward microenvironment

            if x == 0:
                data = calculate_composition_marker_expression(marker_data, contours,
                                                               vessel_id_label="Point_%s" % str(i + 1))
                current_vessel_space_expansion_data.append(data)

            else:
                data, _, stopped_vessels, dark_space_data, vessel_space_data = calculate_microenvironment_marker_expression(
                    marker_data,
                    contours,
                    pixel_expansion_upper_bound=current_interval,
                    pixel_expansion_lower_bound=current_interval - pixel_interval,
                    vesselnonvessel_label="Point_%s" % str(i + 1))

                all_points_stopped_vessels += stopped_vessels
                current_dark_space_expansion_data.append(dark_space_data)
                current_vessel_space_expansion_data.append(vessel_space_data)

            end_expression = datetime.datetime.now()

            print("Finished calculating expression for Point %s in %s" % (str(i+1), end_expression - start_expression))

            current_expansion_data.append(data)

        print("There were %s vessels which could not expand inward/outward by %s pixels" % (
            all_points_stopped_vessels, x * pixel_interval))

        expansion_data.append(current_expansion_data)
        dark_space_expansion_data.append(current_dark_space_expansion_data)
        vessel_space_expansion_data.append(current_vessel_space_expansion_data)

        if x != 0:
            current_interval += pixel_interval

        print("Current interval %s, previous interval %s" % (str(current_interval), str(current_interval -
                                                                                        pixel_interval)))

    return expansion_data, dark_space_expansion_data, vessel_space_expansion_data


def get_inward_expansion_data(all_points_vessel_contours: list,
                              all_points_marker_data: list,
                              pixel_interval: int) -> (list, int):
    """
    Collect inward expansion data for each expansion, for each point, for each vessel

    :param pixel_interval: int -> Pixel interval
    :param all_points_vessel_contours: array_like, [n_points, n_vessels] -> list of vessel contours for each point
    :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]]
    -> list of marker data for each point

    :return: list, [n_expansions, n_points, n_vessels, n_markers] -> Inward microenvironment expansion data,
    int, Final number of expansions needed to complete
    """

    expansion_data = []
    current_interval = pixel_interval
    all_vessels_count = len([item for sublist in all_points_vessel_contours for item in sublist])
    current_expansion_no = 0
    all_points_stopped_vessels = 0
    n_points = config.n_points

    # Continue expanding inward until all vessels have been stopped
    while all_vessels_count - all_points_stopped_vessels >= 0:
        print("Current inward expansion: %s, Interval: %s" % (str(current_expansion_no), str(current_interval)))

        current_expansion_data = []
        all_points_stopped_vessels = 0

        # Iterate through each point
        for i in range(n_points):
            contours = all_points_vessel_contours[i]
            marker_data = all_points_marker_data[i]
            start_expression = datetime.datetime.now()

            data, stopped_vessels = calculate_inward_microenvironment_marker_expression(
                marker_data,
                contours,
                pixel_expansion_upper_bound=current_interval,
                pixel_expansion_lower_bound=current_interval - pixel_interval)

            all_points_stopped_vessels += stopped_vessels
            current_expansion_data.append(data)
            end_expression = datetime.datetime.now()

            print(
                "Finished calculating expression for Point %s in %s" % (str(i + 1), end_expression - start_expression))

        expansion_data.append(current_expansion_data)

        current_interval += pixel_interval
        current_expansion_no += 1

        print("There are %s / %s vessels which have failed to expand inward" % (str(all_points_stopped_vessels),
                                                                                str(all_vessels_count)))

    return expansion_data.reverse(), current_expansion_no


def run_vis():
    """
    Create the visualizations for inward and outward vessel expansions and populate all results in the directory
    set in the configuration settings.
    """

    n_expansions = config.max_expansions
    interval = config.pixel_interval
    expansions = config.expansion_to_run  # Expansions that you want to run

    n_expansions += 1  # Intuitively, 5 expansions means 5 expansions excluding the original composition of the
    # vessel, but we mean 5 expansions including the original composition - thus 4 expansions. Therefore lets add 1
    # so we are on the same page.

    assert n_expansions >= max(expansions), "More expansions selected than available!"

    marker_segmentation_masks, all_points_marker_data, markers_names = get_all_point_data()  # Collect all marker and
    # mask data

    all_points_vessel_contours = []
    all_points_vessel_regions_of_interest = []

    # Collect vessel contours from each segmentation mask
    for segmentation_mask in marker_segmentation_masks:
        vessel_regions_of_interest, contours = extract(segmentation_mask)
        all_points_vessel_contours.append(contours)
        all_points_vessel_regions_of_interest.append(vessel_regions_of_interest)

    # Marker expression overlay masks
    if config.create_marker_expression_overlay_masks:
        marker_expression_masks(all_points_vessel_contours, all_points_marker_data, markers_names)

    # Inward expansion data
    if config.perform_inward_expansions:
        get_inward_expansion_data(all_points_vessel_contours, all_points_marker_data, markers_names)

    # Vessel areas histograms and boxplots
    if config.create_vessel_areas_histograms_and_boxplots:
        vessel_areas_histogram()

    # Vessel expansion ring plots
    if config.create_expansion_ring_plots:
        pixel_expansion_ring_plots()

    # Collect outward microenvironment expansion data, nonvessel space expansion data and vessel space expansion data
    expansion_data, nonvessel_space_expansion_data, vessel_space_expansion_data = get_outward_expansion_data(
        all_points_vessel_contours,
        all_points_marker_data,
        interval,
        n_expansions)

    # Iterate through selected expansions to create heatmaps and line plots
    for x in expansions:

        # Brain region expansion heatmaps
        if config.create_brain_region_expansion_heatmaps:
            brain_region_expansion_heatmap(vessel_space_expansion_data,
                                           nonvessel_space_expansion_data,
                                           markers_names,
                                           x + 1,
                                           interval)
        # Mask/Non-mask heatmaps
        if config.create_vessel_nonvessel_heatmaps:
            vessel_nonvessel_heatmap(vessel_space_expansion_data, nonvessel_space_expansion_data, markers_names, x + 1)

        # Per brain region line plots
        if config.create_brain_region_expansion_line_plots:
            brain_region_plots(x + 1, interval, markers_names, expansion_data)

        # Per point line plots
        if config.create_point_expansion_line_plots:
            point_region_plots(x + 1, interval, markers_names, expansion_data)

        # Per vessel line plots
        if config.create_vessel_expansion_line_plots:
            vessel_region_plots(x + 1, interval, markers_names, expansion_data)

        # All points average line plots
        if config.create_allpoints_expansion_line_plots:
            all_points_plots(x + 1, interval, markers_names, expansion_data)


if __name__ == '__main__':
    run_vis()
