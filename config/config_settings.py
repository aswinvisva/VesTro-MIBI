import os
from datetime import datetime

import matplotlib
import matplotlib.pylab as pl

from src.utils.utils_functions import mkdir_p


class Config:
    # High-Level Settings
    n_workers = 5
    data_resolution = "hires"
    save_to_csv = False

    if save_to_csv:
        csv_name = "data.csv"

    # Marker settings for reading data

    markers_to_ignore = [
        "GAD",
        "Neurogranin",
        "ABeta40",
        "pTDP43",
        "polyubik63",
        "Background",
        "Au197",
        "Ca40",
        "Fe56",
        "Na23",
        "Si28",
        "La139",
        "Ta181",
        "C12"
    ]

    marker_clusters = {
        "Nucleus": ["HH3"],
        "Microglia": ["CD45", "HLADR", "Iba1"],
        "Disease": ["CD47", "ABeta42", "polyubiK48", "PHFTau", "8OHGuanosine"],
        "Vessels": ["SMA", "CD31", "CollagenIV", "TrkA", "GLUT1", "Desmin", "vWF", "CD105"],
        "Astrocytes": ["S100b", "GlnSyn", "Cx30", "EAAT2", "CD44", "GFAP", "Cx43"],
        "Synapse": ["CD56", "Synaptophysin", "VAMP2", "PSD95"],
        "Oligodendrocytes": ["MOG", "MAG"],
        "Neurons": ["Calretinin", "Parvalbumin", "MAP2", "Gephyrin"]
    }

    mask_marker_clusters = {
        "Vessels": ["GLUT1", "vWF", "CD31", "SMA"]
    }

    all_masks = [
        'astrocytes',
        'BBB',
        'largevessels',
        'microglia',
        'myelin',
        'plaques',
        'tangles',
        'allvessels'
    ]

    non_marker_vars = ["Contour Area",
                       "Vessel Size",
                       "SMA Presence",
                       "Data Type",
                       "Asymmetry",
                       "Asymmetry Score",
                       "Region"
                       # "UMAP0",
                       # "UMAP1",
                       # "TSNE0",
                       # "TSNE1",
                       # "PCA0",
                       # "PCA1",
                       # "SVD0",
                       # "SVD1",
                       # "K-Means",
                       # "Hierarchical"
                       ]

    n_markers = 34

    if data_resolution == "hires":
        brain_region_point_ranges = [(1, 16), (17, 32), (33, 48)]
    elif data_resolution == "medres":
        brain_region_point_ranges = [(1, 100), (101, 200), (201, 300)]

    if data_resolution == "hires":
        segmentation_mask_size = (1024, 1024)
    elif data_resolution == "medres":
        segmentation_mask_size = (512, 512)

    data_resolution_size = (500, 500)
    data_resolution_units = "μm"
    pixels_to_distance = float(data_resolution_size[0]) / float(segmentation_mask_size[0])

    brain_region_names = ["MFG", "HIP", "CAUD"]

    selected_segmentation_mask_type = "allvessels"

    data_dir = os.path.join("/media/large_storage/oliveria_data/data", data_resolution)
    masks_dir = os.path.join("/media/large_storage/oliveria_data/masks", data_resolution)
    point_dir = "Point"
    tifs_dir = "TIFs"

    caud_hip_mfg_separate_dir = data_resolution == "medres"

    if caud_hip_mfg_separate_dir:
        caud_dir = "CAUD"
        hip_dir = "HIP"
        mfg_dir = "MFG"

    if data_resolution == "hires":
        n_points = 48
    elif data_resolution == "medres":
        n_points = 300

    if caud_hip_mfg_separate_dir:
        n_points_per_dir = 100
    else:
        n_points_per_dir = n_points

    show_segmentation_masks_when_reading = False
    describe_markers_when_reading = False

    # Settings for extracting vessels from segmentation mask

    show_vessel_contours_when_extracting = False
    minimum_contour_area_to_remove = float(125) / float((1.0 / pixels_to_distance) ** 2)
    use_guassian_blur_when_extracting_vessels = True
    create_removed_vessels_mask = False
    create_blurred_vessels_mask = False

    if use_guassian_blur_when_extracting_vessels:
        guassian_blur = (2, 2)

    # Settings for calculation marker expression

    scaling_factor = 100
    expression_type = "area_normalized_counts"
    transformation_type = "arcsinh"
    normalization_type = "percentile"

    if normalization_type == "percentile":
        percentile_to_normalize = 99

    show_probability_distribution_for_expression = False
    show_vessel_masks_when_generating_expression = False

    distance_interval = 0.5

    if distance_interval is None:
        pixel_interval = 2.0
    else:
        pixel_interval = distance_interval / pixels_to_distance

    expansion_to_run = [10]
    perform_inward_expansions = True

    if perform_inward_expansions:
        max_inward_expansion = 10

    max_expansions = None  # Set to None to select max_expansions automatically

    # Split settings

    primary_categorical_splitter = "Asymmetry"
    secondary_categorical_splitter = "Vessel Size"

    SMA_positive_threshold = 0.1
    large_vessel_threshold = 500
    medium_vessel_threshold = 200  # 300
    small_vessel_threshold = 0  # 50

    if max_expansions is None:
        max_expansions = max(expansion_to_run)

    # Figures to generate
    create_vessel_id_plot = False
    create_vessel_nonvessel_mask = False  #
    create_marker_expression_overlay_masks = False
    create_vessel_areas_histograms_and_boxplots = False
    create_brain_region_expansion_heatmaps = False  #
    create_vessel_nonvessel_heatmaps = False  #
    create_categorical_split_expansion_heatmaps = False  #
    create_brain_region_expansion_line_plots = False  #
    create_point_expansion_line_plots = False  #
    create_vessel_expansion_line_plots = False
    create_allpoints_expansion_line_plots = False  #
    create_expansion_ring_plots = False
    create_embedded_vessel_id_masks = False
    create_removed_vessels_expression_boxplot = False
    create_biaxial_scatter_plot = False
    create_expanded_vessel_masks = False
    create_spatial_probability_maps = False  #
    create_expression_histogram = False
    create_expansion_violin_plots = False  #
    create_categorical_violin_plot = False  #
    create_expansion_box_plots = False  #
    create_vessel_asymmetry_area_spread_plot = False  #
    create_continuous_scatter_plot = False  #
    create_umap_projection_scatter_plots = False  #
    create_vessel_images_by_categorical_variable = False  #
    create_pseudo_time_heatmap = True  #
    create_categorical_violin_plot_with_images = True  #

    if create_vessel_areas_histograms_and_boxplots:
        show_boxplot_outliers = False

    if create_vessel_nonvessel_mask:
        vessel_space_colour = (51, 153, 0)
        nonvessel_mask_colour = (0, 0, 179)
        vessel_mask_colour = (153, 51, 0)

    line_plots_color_maps = {
        "Nucleus": matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow"]),
        "Microglia": pl.cm.Greens,
        "Disease": pl.cm.Greys,
        "Vessels": pl.cm.Blues,
        "Astrocytes": pl.cm.Reds,
        "Synapse": pl.cm.Purples,
        "Oligodendrocytes": pl.cm.Oranges,
        "Neurons": matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "darkmagenta"])
    }

    line_plots_bin_colors = {
        "Nucleus": "y",
        "Microglia": "g",
        "Disease": "k",
        "Vessels": "b",
        "Astrocytes": "r",
        "Synapse": "rebeccapurple",
        "Oligodendrocytes": "darkorange",
        "Neurons": "m"
    }

    vessel_line_plots_points = [1, 7, 26, 30, 43, 48]

    def display(self):
        """Display Configurations."""

        print("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()
