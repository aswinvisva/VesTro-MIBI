import sys
import json

sys.path.insert(0, '..')

marker_groups_spec = json.load(open("specs/marker_groups.json"))
experiment_spec = json.load(open("specs/10_um_expansion_spec.json"))


class Config:
    # High-Level Settings
    n_workers = 5

    marker_clusters = marker_groups_spec["marker_clusters"][0]
    mask_marker_clusters = marker_groups_spec["marker_clusters"][0]

    show_segmentation_masks_when_reading = False
    describe_markers_when_reading = False

    # Settings for extracting vessels from segmentation mask

    pixel_interval = experiment_spec["pixel_interval"]

    show_vessel_contours_when_extracting = False
    minimum_contour_area_to_remove = float(experiment_spec["minimum_contour_area_to_remove"])
    use_guassian_blur_when_extracting_vessels = True

    guassian_blur = tuple(experiment_spec["guassian_blur"])

    # Settings for calculation marker expression

    scaling_factor = float(experiment_spec["scaling_factor"])
    expression_type = experiment_spec["expression_type"]
    transformation_type = experiment_spec["transformation_type"]
    normalization_type = experiment_spec["normalization_type"]

    if normalization_type == "percentile":
        percentile_to_normalize = experiment_spec["percentile_to_normalize"]

    show_probability_distribution_for_expression = False
    show_vessel_masks_when_generating_expression = False

    expansion_to_run = experiment_spec["expansion_to_run"]
    perform_inward_expansions = experiment_spec["perform_inward_expansions"]

    if perform_inward_expansions:
        max_inward_expansion = experiment_spec["max_inward_expansions"]

    max_expansions = max(expansion_to_run)

    SMA_positive_threshold = experiment_spec["SMA_positive_threshold"]
    large_vessel_threshold = experiment_spec["large_vessel_threshold"]
    medium_vessel_threshold = experiment_spec["medium_vessel_threshold"]  # 300
    small_vessel_threshold = experiment_spec["small_vessel_threshold"]  # 50

    def display(self):
        """Display Configurations."""

        print("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()
