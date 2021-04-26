import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC

from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn.utils import axis_ticklabels_overlap

from src.data_loading.mibi_reader import MIBIReader
from src.data_preprocessing.object_extractor import ObjectExtractor
from src.data_preprocessing.markers_feature_gen import *
from src.utils.utils_functions import mkdir_p

from src.data_visualization.base import BaseMIBIPlot


class Heatmap(BaseMIBIPlot, ABC):

    def __init__(self,
                 x=None,
                 y=None,
                 data=None,
                 hue=None,
                 style=None,
                 color_map=None,
                 palette=None,
                 x_tick_labels=None,
                 y_tick_labels=None,
                 x_axis_label=None,
                 y_axis_label=None,
                 save=True,
                 show=False,
                 save_dir=None,
                 figsize=(22, 10),
                 fig_name="Test",
                 ax=None
                 ):
        """
        Heatmap Plot

        :param x: str, Column name for x data
        :param y: str, Column name for y data
        :param data: pd.DataFrame, Pandas Dataframe data
        :param hue: str, Column name for categorical variable
        :param style: str, Column name for style
        :param color_map: matplotlib.ColorMap, Color map
        :param palette: matplotlib.Palette, Palette
        :param x_tick_labels: array_like, X tick labels
        :param y_tick_labels: array_like, Y tick labels
        :param save: bool, Save figure
        :param show: bool, Show figure
        :param figsize: tuple, Figure size
        :param fig_name: str, Figure name
        :param ax: matplotlib.SubplotAxes, Matplotlib Axis
        """

        self.y_axis_label = y_axis_label
        self.x_axis_label = x_axis_label
        self.y_tick_labels = y_tick_labels
        self.x_tick_labels = x_tick_labels
        self.data = data
        self.ax = ax
        self.figsize = figsize
        self.show = show
        self.save = save
        self.palette = palette
        self.cmap = color_map
        self.style = style
        self.hue = hue
        self.y = y
        self.x = x

    def make_figure(self, **kwargs):
        """
        Make the figure

        :param kwargs:
        :return:
        """

        cluster = kwargs.get("cluster", False)
        row_cluster = kwargs.get("row_cluster", True)
        col_cluster = kwargs.get("col_cluster", False)
        x_tick_label_rotation = kwargs.get("x_tick_label_rotation", 45)

        if not cluster:
            plt.figure(figsize=self.figsize)

            ax = sns.heatmap(self.data,
                             cmap=self.cmap,
                             xticklabels=self.x_tick_labels,
                             yticklabels=self.y_tick_labels,
                             linewidths=0,
                             )

            ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

            if axis_ticklabels_overlap(ax.get_xticklabels()):
                ax.set_xticklabels(ax.get_xticklabels(), rotation=x_tick_label_rotation, ha="right")

            plt.xlabel("%s" % self.x_axis_label)

        else:
            ax = sns.clustermap(self.data,
                                cmap=self.cmap,
                                row_cluster=row_cluster,
                                col_cluster=col_cluster,
                                linewidths=0,
                                xticklabels=self.x_tick_labels,
                                yticklabels=self.y_tick_labels,
                                figsize=self.figsize
                                )

            ax_ax = ax.ax_heatmap
            ax_ax.set_xlabel("%s" % self.x_axis_label)

            ax_ax.set_xticklabels(ax_ax.get_xticklabels(), rotation="horizontal")

            if axis_ticklabels_overlap(ax_ax.get_xticklabels()):
                ax_ax.set_xticklabels(ax_ax.get_xticklabels(), rotation=x_tick_label_rotation, ha="right")

        return ax


def vessel_nonvessel_heatmap(config: Config,
                             markers_names: list,
                             n_expansions: int,
                             feed_features: pd.DataFrame,
                             brain_regions: list,
                             marker_clusters: list,
                             feed_dir: str):
    """
    Vessel/Non-vessel heatmaps helper method

    :param markers_names:
    :param config:
    :param feed_dir:
    :param brain_regions:
    :param marker_clusters:
    :param n_expansions:
    :param feed_features:
    :return:
    """

    # Vessel Space (SMA Positive)
    positve_sma = feed_features.loc[
        feed_features["SMA"] >= config.SMA_positive_threshold]

    idx = pd.IndexSlice

    try:
        all_vessels_sma_data = positve_sma.loc[idx[:, :,
                                               :n_expansions,
                                               "Data"], markers_names].to_numpy()
        all_vessels_sma_data = np.mean(all_vessels_sma_data, axis=0)
    except KeyError:
        all_vessels_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        mfg_vessels_sma_data = positve_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                               :,
                                               :n_expansions,
                                               "Data"], markers_names].to_numpy()
        mfg_vessels_sma_data = np.mean(mfg_vessels_sma_data, axis=0)

    except KeyError:
        mfg_vessels_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        hip_vessels_sma_data = positve_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                               :,
                                               :n_expansions,
                                               "Data"], markers_names].to_numpy()
        hip_vessels_sma_data = np.mean(hip_vessels_sma_data, axis=0)
    except KeyError:
        hip_vessels_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        caud_vessels_sma_data = positve_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                :,
                                                :n_expansions,
                                                "Data"], markers_names].to_numpy()
        caud_vessels_sma_data = np.mean(caud_vessels_sma_data, axis=0)

    except KeyError:
        caud_vessels_sma_data = np.zeros((config.n_markers,), np.uint8)

    # Vessel Space (SMA Negative)
    negative_sma = feed_features.loc[
        feed_features["SMA"] < config.SMA_positive_threshold]

    idx = pd.IndexSlice

    try:
        all_vessels_non_sma_data = negative_sma.loc[idx[:, :,
                                                    :n_expansions,
                                                    "Data"], markers_names].to_numpy()
        all_vessels_non_sma_data = np.mean(all_vessels_non_sma_data, axis=0)

    except KeyError:
        all_vessels_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        mfg_vessels_non_sma_data = negative_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                    :,
                                                    :n_expansions,
                                                    "Data"], markers_names].to_numpy()
        mfg_vessels_non_sma_data = np.mean(mfg_vessels_non_sma_data, axis=0)

    except KeyError:
        mfg_vessels_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        hip_vessels_non_sma_data = negative_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                    :,
                                                    :n_expansions,
                                                    "Data"], markers_names].to_numpy()
        hip_vessels_non_sma_data = np.mean(hip_vessels_non_sma_data, axis=0)

    except KeyError:
        hip_vessels_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        caud_vessels_non_sma_data = negative_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                     :,
                                                     :n_expansions,
                                                     "Data"], markers_names].to_numpy()
        caud_vessels_non_sma_data = np.mean(caud_vessels_non_sma_data, axis=0)

    except KeyError:
        caud_vessels_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    # Non-vessel Space

    try:
        all_nonmask_sma_data = positve_sma.loc[idx[:, :, :,
                                               "Non-Vascular Space"], markers_names].to_numpy()
        all_nonmask_sma_data = np.mean(all_nonmask_sma_data, axis=0)

    except KeyError:
        all_nonmask_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        mfg_nonmask_sma_data = positve_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                               :,
                                               :n_expansions,
                                               "Non-Vascular Space"], markers_names].to_numpy()
        mfg_nonmask_sma_data = np.mean(mfg_nonmask_sma_data, axis=0)

    except KeyError:
        mfg_nonmask_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        hip_nonmask_sma_data = positve_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                               :,
                                               :n_expansions,
                                               "Non-Vascular Space"], markers_names].to_numpy()
        hip_nonmask_sma_data = np.mean(hip_nonmask_sma_data, axis=0)

    except KeyError:
        hip_nonmask_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        caud_nonmask_sma_data = positve_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                :,
                                                :n_expansions,
                                                "Non-Vascular Space"], markers_names].to_numpy()
        caud_nonmask_sma_data = np.mean(caud_nonmask_sma_data, axis=0)

    except KeyError:
        caud_nonmask_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        all_nonmask_non_sma_data = negative_sma.loc[idx[:, :, :,
                                                    "Non-Vascular Space"], markers_names].to_numpy()
        all_nonmask_non_sma_data = np.mean(all_nonmask_non_sma_data, axis=0)

    except KeyError:
        all_nonmask_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        mfg_nonmask_non_sma_data = negative_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                    :,
                                                    :n_expansions,
                                                    "Non-Vascular Space"], markers_names].to_numpy()
        mfg_nonmask_non_sma_data = np.mean(mfg_nonmask_non_sma_data, axis=0)

    except KeyError:
        mfg_nonmask_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        hip_nonmask_non_sma_data = negative_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                    :,
                                                    :n_expansions,
                                                    "Non-Vascular Space"], markers_names].to_numpy()
        hip_nonmask_non_sma_data = np.mean(hip_nonmask_non_sma_data, axis=0)

    except KeyError:
        hip_nonmask_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        caud_nonmask_non_sma_data = negative_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                     :,
                                                     :n_expansions,
                                                     "Non-Vascular Space"], markers_names].to_numpy()
        caud_nonmask_non_sma_data = np.mean(caud_nonmask_non_sma_data, axis=0)

    except KeyError:
        caud_nonmask_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    # Vessel environment space

    try:
        all_vessels_environment_sma_data = positve_sma.loc[idx[:, :,
                                                           :n_expansions,
                                                           "Vascular Space"], markers_names].to_numpy()
        all_vessels_environment_sma_data = np.mean(all_vessels_environment_sma_data, axis=0)

    except KeyError:
        all_vessels_environment_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        mfg_vessels_environment_sma_data = positve_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                           :,
                                                           :n_expansions,
                                                           "Vascular Space"], markers_names].to_numpy()
        mfg_vessels_environment_sma_data = np.mean(mfg_vessels_environment_sma_data, axis=0)

    except KeyError:
        mfg_vessels_environment_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        hip_vessels_environment_sma_data = positve_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                           :,
                                                           :n_expansions,
                                                           "Vascular Space"], markers_names].to_numpy()
        hip_vessels_environment_sma_data = np.mean(hip_vessels_environment_sma_data, axis=0)

    except KeyError:
        hip_vessels_environment_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        caud_vessels_environment_sma_data = positve_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                            :,
                                                            :n_expansions,
                                                            "Vascular Space"], markers_names].to_numpy()
        caud_vessels_environment_sma_data = np.mean(caud_vessels_environment_sma_data, axis=0)

    except KeyError:
        caud_vessels_environment_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        all_vessels_environment_non_sma_data = negative_sma.loc[idx[:, :,
                                                                :n_expansions,
                                                                "Vascular Space"], markers_names].to_numpy()
        all_vessels_environment_non_sma_data = np.mean(all_vessels_environment_non_sma_data, axis=0)

    except KeyError:
        all_vessels_environment_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        mfg_vessels_environment_non_sma_data = negative_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                                :,
                                                                :n_expansions,
                                                                "Vascular Space"], markers_names].to_numpy()
        mfg_vessels_environment_non_sma_data = np.mean(mfg_vessels_environment_non_sma_data, axis=0)

    except KeyError:
        mfg_vessels_environment_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        hip_vessels_environment_non_sma_data = negative_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                                :,
                                                                :n_expansions,
                                                                "Vascular Space"], markers_names].to_numpy()
        hip_vessels_environment_non_sma_data = np.mean(hip_vessels_environment_non_sma_data, axis=0)

    except KeyError:
        hip_vessels_environment_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    try:
        caud_vessels_environment_non_sma_data = negative_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                                 :,
                                                                 :n_expansions,
                                                                 "Vascular Space"], markers_names].to_numpy()
        caud_vessels_environment_non_sma_data = np.mean(caud_vessels_environment_non_sma_data, axis=0)

    except KeyError:
        caud_vessels_environment_non_sma_data = np.zeros((config.n_markers,), np.uint8)

    all_data = [all_vessels_sma_data,
                all_vessels_non_sma_data,
                all_vessels_environment_sma_data,
                all_vessels_environment_non_sma_data,
                all_nonmask_sma_data,
                all_nonmask_non_sma_data,
                mfg_vessels_sma_data,
                mfg_vessels_non_sma_data,
                mfg_vessels_environment_sma_data,
                mfg_vessels_environment_non_sma_data,
                mfg_nonmask_sma_data,
                mfg_nonmask_non_sma_data,
                hip_vessels_sma_data,
                hip_vessels_non_sma_data,
                hip_vessels_environment_sma_data,
                hip_vessels_environment_non_sma_data,
                hip_nonmask_sma_data,
                hip_nonmask_non_sma_data,
                caud_vessels_sma_data,
                caud_vessels_non_sma_data,
                caud_vessels_environment_sma_data,
                caud_vessels_environment_non_sma_data,
                caud_nonmask_sma_data,
                caud_nonmask_non_sma_data]

    yticklabels = ["Vascular Space (SMA+) - All Points",
                   "Vascular Space (SMA-) - All Points",
                   "Vascular Expansion Space (SMA+) - All Points",
                   "Vascular Expansion Space (SMA-) - All Points",
                   "Non-Vascular Space (SMA+) - All Points",
                   "Non-Vascular Space (SMA-) - All Points",
                   "Vascular Space (SMA+) - MFG",
                   "Vascular Space (SMA-) - MFG",
                   "Vascular Expansion Space (SMA+) - MFG",
                   "Vascular Expansion Space (SMA-) - MFG",
                   "Non-Vascular Space (SMA+) - MFG",
                   "Non-Vascular Space (SMA-) - MFG",
                   "Vascular Space (SMA+) - HIP",
                   "Vascular Space (SMA-) - HIP",
                   "Vascular Expansion Space (SMA+) - HIP",
                   "Vascular Expansion Space (SMA-) - HIP",
                   "Non-Vascular Space (SMA+) - HIP",
                   "Non-Vascular Space (SMA-) - HIP",
                   "Vascular Space (SMA+) - CAUD",
                   "Vascular Space (SMA-) - CAUD",
                   "Vascular Expansion Space (SMA+) - CAUD",
                   "Vascular Expansion Space (SMA-) - CAUD",
                   "Non-Vascular Space (SMA+) - CAUD",
                   "Non-Vascular Space (SMA-) - CAUD", ]

    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "black"],
              [norm(-0.5), "indigo"],
              [norm(0), "firebrick"],
              [norm(0.5), "orange"],
              [norm(1.0), "khaki"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    hm = Heatmap(data=all_data,
                 color_map=cmap,
                 x_tick_labels=markers_names,
                 y_tick_labels=yticklabels,
                 figsize=(22, 10),
                 save=True,
                 show=False
                 )

    ax = hm.make_figure(cluster=False)

    v_line_idx = 0

    for key in marker_clusters.keys():
        if v_line_idx != 0:
            ax.axvline(v_line_idx, 0, len(yticklabels), linewidth=3, c='w')

        for _ in marker_clusters[key]:
            v_line_idx += 1

    h_line_idx = 0

    while h_line_idx < len(yticklabels):
        h_line_idx += 6
        ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

    hm.savefig_or_show(save_dir="%s/Heatmaps" % feed_dir,
                       fig_name='Expansion_%s.png' % str(n_expansions))

    ax = hm.make_figure(cluster=True)

    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    ax.ax_heatmap.yaxis.tick_left()
    ax.ax_heatmap.yaxis.set_label_position("left")

    hm.savefig_or_show(save_dir="%s/Clustermaps" % feed_dir,
                       fig_name='Expansion_%s.png' % str(n_expansions),
                       ax=ax)
