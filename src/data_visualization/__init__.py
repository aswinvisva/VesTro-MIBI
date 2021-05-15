import matplotlib
import matplotlib.pylab as pl
import numpy as np

SHOW_BOXPLOT_OUTLIERS = False
DEFAULT_VESSEL_SPACE_COLOUR = (51, 153, 0)
DEFAULT_NONVESSEL_SPACE_COLOUR = (0, 0, 179)
DEFAULT_VESSEL_MASK_COLOUR = (153, 51, 0)

DEFAULT_LINE_PLOTS_COLOR_MAPS = {
    "Nucleus": matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow"]),
    "Microglia": pl.cm.Greens,
    "Disease": pl.cm.Greys,
    "Vessels": pl.cm.Blues,
    "Astrocytes": pl.cm.Reds,
    "Synapse": pl.cm.Purples,
    "Oligodendrocytes": pl.cm.Oranges,
    "Neurons": matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "darkmagenta"])
}

DEFAULT_LINE_PLOT_BIN_COLORS = {
    "Nucleus": "y",
    "Microglia": "g",
    "Disease": "k",
    "Vessels": "b",
    "Astrocytes": "r",
    "Synapse": "rebeccapurple",
    "Oligodendrocytes": "darkorange",
    "Neurons": "m"
}

norm = matplotlib.colors.Normalize(-1, 1)
heatmap_colors = [[norm(-1.0), "black"],
          [norm(-0.5), "indigo"],
          [norm(0), "firebrick"],
          [norm(0.5), "orange"],
          [norm(1.0), "khaki"]]

DEFAULT_HEATMAP_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list("", heatmap_colors)
DEFAULT_VESSEL_LINE_PLOTS_POINTS = [1, 7, 26, 30, 43, 48]
DEFAULT_BINARY_COLORMAP = matplotlib.colors.ListedColormap(['blue', 'red'])(np.linspace(0, 1, 2))
DEFAULT_MULTICLASS_COLORMAP = matplotlib.colors.ListedColormap(['cyan', 'pink', 'yellow'])(np.linspace(0, 1, 3))