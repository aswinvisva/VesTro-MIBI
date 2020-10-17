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
segmentation_mask_size = (1024, 1024)
brain_region_names = ["MFG", "HIP", "CAUD"]

selected_segmentation_mask_type = "allvessels"

data_dir = "data"
point_dir = "Point"
tifs_dir = "TIFs"
masks_dr = "masks"

n_points = 48

show_segmentation_masks_when_reading = False
describe_markers_when_reading = False

# Settings for extracting vessels from segmentation mask

show_vessel_contours_when_extracting = False
minimum_contour_area_to_remove = 30
use_guassian_blur_when_extracting_vessels = True

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

# Visualization settings for plotting data

visualization_results_dir = "results"
max_expansions = 12
pixel_interval = 5
expansion_to_run = [2, 4, 8, 12]

perform_inward_expansions = False

create_vessel_id_plot = False
create_vessel_nonvessel_mask = False
create_marker_expression_overlay_masks = False
create_vessel_areas_histograms_and_boxplots = False
create_brain_region_expansion_heatmaps = True
create_vessel_nonvessel_heatmaps = True
create_brain_region_expansion_line_plots = True
create_point_expansion_line_plots = True
create_vessel_expansion_line_plots = True
create_allpoints_expansion_line_plots = True
create_expansion_ring_plots = False

if create_vessel_areas_histograms_and_boxplots:
    show_boxplot_outliers = False

if create_vessel_nonvessel_mask:
    vessel_mask_colour = (51, 153, 0)
    nonvessel_mask_colour = (0, 0, 179)

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
