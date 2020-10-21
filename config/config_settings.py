import os

import matplotlib
import matplotlib.pylab as pl

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

n_markers = 34
brain_region_point_ranges = [(1, 16), (17, 32), (33, 48)]
# brain_region_point_ranges = [(1, 100), (101, 200), (201, 300)]

segmentation_mask_size = (1024, 1024)
# segmentation_mask_size = (512, 512)

brain_region_names = ["MFG", "HIP", "CAUD"]

selected_segmentation_mask_type = "allvessels"

data_type = "hires"
data_dir = os.path.join("data", data_type)
masks_dr = os.path.join("masks", data_type)
point_dir = "Point"
tifs_dir = "TIFs"
caud_hip_mfg_separate_dir = False

if caud_hip_mfg_separate_dir:
    caud_dir = "CAUD"
    hip_dir = "HIP"
    mfg_dir = "MFG"

n_points = 48

if caud_hip_mfg_separate_dir:
    n_points_per_dir = 100
else:
    n_points_per_dir = n_points

show_segmentation_masks_when_reading = False
describe_markers_when_reading = False

# Settings for extracting vessels from segmentation mask

show_vessel_contours_when_extracting = False
minimum_contour_area_to_remove = 30
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

# Visualization settings for plotting data

visualization_results_dir = "results"
pixel_interval = 5
expansion_to_run = [1, 2, 4, 8, 12]
max_expansions = None  # Set to None to select max_expansions automatically

if max_expansions is None:
    max_expansions = max(expansion_to_run)

perform_inward_expansions = False

# Figures to generate

create_vessel_id_plot = False
create_vessel_nonvessel_mask = False
create_marker_expression_overlay_masks = False
create_vessel_areas_histograms_and_boxplots = False
create_brain_region_expansion_heatmaps = True
create_vessel_nonvessel_heatmaps = False
create_brain_region_expansion_line_plots = False
create_point_expansion_line_plots = False
create_vessel_expansion_line_plots = False
create_allpoints_expansion_line_plots = False
create_expansion_ring_plots = False
create_embedded_vessel_id_masks = False

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
