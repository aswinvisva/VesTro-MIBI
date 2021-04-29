from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from abc import ABC

from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn.utils import axis_ticklabels_overlap

from src.data_loading.mibi_reader import MIBIReader
from src.data_preprocessing.object_extractor import ObjectExtractor
from src.data_preprocessing.markers_feature_gen import *
from src.utils.utils_functions import mkdir_p, round_to_nearest_half

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
                 ax=None,
                 vmin=None,
                 vmax=None
                 ):
        """
        Base MIBI Plot

        :param x: str, Column name for x data
        :param y: str, Column name for y data
        :param data: pd.DataFrame, Pandas Dataframe data
        :param hue: str, Column name for categorical variable
        :param style: str, Column name for style
        :param color_map: matplotlib.ColorMap, Color map
        :param palette: matplotlib.Palette, Palette
        :param x_tick_labels: array_like, X tick labels
        :param y_tick_labels: array_like, Y tick labels
        :param ax: matplotlib.SubplotAxes, Matplotlib Axis
        """

        self.y_axis_label = y_axis_label
        self.x_axis_label = x_axis_label
        self.y_tick_labels = y_tick_labels
        self.x_tick_labels = x_tick_labels
        self.data = data
        self.ax = ax
        self.palette = palette
        self.cmap = color_map
        self.style = style
        self.hue = hue
        self.y = y
        self.x = x
        self.vmin = vmin
        self.vmax = vmax

    def make_figure(self, marker_clusters, **kwargs):
        """
        Make the figure
        """

        cluster = kwargs.get("cluster", False)
        row_cluster = kwargs.get("row_cluster", True)
        col_cluster = kwargs.get("col_cluster", False)
        x_tick_label_rotation = kwargs.get("x_tick_label_rotation", 45)
        x_tick_values = kwargs.get("x_tick_values", None)
        x_tick_indices = kwargs.get("x_tick_indices", None)

        if not cluster:

            if self.ax is None:
                plt.figure(figsize=(22, 10))

                ax = sns.heatmap(self.data,
                                 cmap=self.cmap,
                                 xticklabels=self.x_tick_labels,
                                 yticklabels=self.y_tick_labels,
                                 linewidths=0,
                                 vmin=self.vmin,
                                 vmax=self.vmax
                                 )

                ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

                if axis_ticklabels_overlap(ax.get_xticklabels()):
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                if len(self.x_tick_labels) != self.data.shape[
                    1] or x_tick_values is not None or x_tick_indices is not None:
                    xticks = ax.get_xticklabels()

                    for xtick in xticks:
                        xtick.set_visible(False)

                    if x_tick_values is not None:
                        for i, val in enumerate(self.x_tick_labels):
                            if val in x_tick_values:
                                xticks[i].set_visible(True)
                                x_tick_values.remove(val)

                    if x_tick_indices is not None:
                        for i in x_tick_indices:
                            xticks[i].set_visible(True)

                plt.xlabel("%s" % self.x_axis_label)

                h_line_idx = 0

                for key in marker_clusters.keys():
                    if h_line_idx != 0:
                        ax.axhline(h_line_idx, 0, len(self.y_tick_labels), linewidth=3, c='w')

                    for _ in marker_clusters[key]:
                        h_line_idx += 1
            else:
                ax = sns.heatmap(self.data,
                                 cmap=self.cmap,
                                 xticklabels=self.x_tick_labels,
                                 yticklabels=self.y_tick_labels,
                                 linewidths=0,
                                 vmin=self.vmin,
                                 vmax=self.vmax,
                                 ax=self.ax
                                 )

                ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

                if axis_ticklabels_overlap(ax.get_xticklabels()):
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_tick_label_rotation, ha="right")

                if len(self.x_tick_labels) != self.data.shape[
                    1] or x_tick_values is not None or x_tick_indices is not None:
                    xticks = ax.get_xticklabels()

                    for xtick in xticks:
                        xtick.set_visible(False)

                    if x_tick_values is not None:
                        for i, val in enumerate(self.x_tick_labels):
                            if val in x_tick_values:
                                xticks[i].set_visible(True)
                                x_tick_values.remove(val)

                    if x_tick_indices is not None:
                        for i in x_tick_indices:
                            xticks[i].set_visible(True)

                plt.xlabel("%s" % self.x_axis_label)

                h_line_idx = 0

                for key in marker_clusters.keys():
                    if h_line_idx != 0:
                        ax.axhline(h_line_idx, 0, len(self.markers_names), linewidth=3, c='w')

                    for _ in marker_clusters[key]:
                        h_line_idx += 1

        else:
            ax = sns.clustermap(self.data,
                                cmap=self.cmap,
                                row_cluster=row_cluster,
                                col_cluster=col_cluster,
                                linewidths=0,
                                xticklabels=self.x_tick_labels,
                                yticklabels=self.y_tick_labels,
                                figsize=(20, 10),
                                vmin=0,
                                vmax=1)
            ax_ax = ax.ax_heatmap
            ax_ax.set_xlabel("%s" % self.x_axis_label)

            ax_ax.set_xticklabels(ax_ax.get_xticklabels(), rotation="horizontal")

            if axis_ticklabels_overlap(ax_ax.get_xticklabels()):
                ax_ax.set_xticklabels(ax_ax.get_xticklabels(), rotation=x_tick_label_rotation, ha="right")

            if len(self.x_tick_labels) != self.data.shape[1] or x_tick_values is not None or x_tick_indices is not None:
                xticks = ax_ax.get_xticklabels()

                for xtick in xticks:
                    xtick.set_visible(False)

                if x_tick_values is not None:
                    for i, val in enumerate(self.x_tick_labels):
                        if val in x_tick_values:
                            xticks[i].set_visible(True)
                            x_tick_values.remove(val)

                if x_tick_indices is not None:
                    for i in x_tick_indices:
                        xticks[i].set_visible(True)

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
                 y_tick_labels=yticklabels
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


def _marker_expression_at_expansion(expansion: int,
                                    mibi_features: pd.DataFrame,
                                    marker_names: list,
                                    region: list = None,
                                    data_type: str = "Data"):
    """
    Helper Function to Calculate the Average Marker Expression at a Given Expansion
    """

    try:
        if region is None:
            marker_expression_matrix = mibi_features.loc[pd.IndexSlice[:, :,
                                                         expansion,
                                                         data_type], marker_names].to_numpy()
        else:
            marker_expression_matrix = mibi_features.loc[pd.IndexSlice[region[0]:region[1],
                                                         :,
                                                         expansion,
                                                         data_type], marker_names].to_numpy()

    except KeyError:
        return np.zeros((len(marker_names),), np.uint8)

    return np.mean(marker_expression_matrix, axis=0)


def brain_region_expansion_heatmap(expansion_features: pd.DataFrame,
                                   heatmaps_dir: str,
                                   clustermaps_dir: str,
                                   marker_names: list,
                                   pixel_interval: float,
                                   brain_regions: list,
                                   brain_region_names: list,
                                   marker_clusters: dict,
                                   data_resolution_units: str,
                                   ):
    """
    Brain Region Expansion Heatmap
    """

    regions_mask_data = []

    marker_mask_expression_map_fn_all_regions = partial(_marker_expression_at_expansion,
                                                        mibi_features=expansion_features,
                                                        marker_names=marker_names,
                                                        data_type="Data")

    all_regions_mask_data = np.array(list(map(marker_mask_expression_map_fn_all_regions,
                                              sorted(expansion_features.index.unique("Expansion").tolist()))))

    all_nonmask_data = expansion_features.loc[pd.IndexSlice[:, :,
                                              max(expansion_features.index.unique("Expansion").tolist()),
                                              "Non-Vascular Space"], marker_names].to_numpy()

    mean_nonmask_data = np.mean(all_nonmask_data, axis=0)

    all_mask_data = np.append(all_regions_mask_data, [mean_nonmask_data], axis=0)

    all_mask_data = np.transpose(all_mask_data)

    for region in brain_regions:
        region_marker_mask_expression_map_fn = partial(_marker_expression_at_expansion,
                                                       mibi_features=expansion_features,
                                                       marker_names=marker_names,
                                                       region=region,
                                                       data_type="Data")

        region_mask_data = np.array(list(map(region_marker_mask_expression_map_fn,
                                             sorted(expansion_features.index.unique("Expansion").tolist()))))

        region_non_mask_data = expansion_features.loc[pd.IndexSlice[region[0]:region[1],
                                                      :,
                                                      max(expansion_features.index.unique("Expansion").tolist()),
                                                      "Non-Vascular Space"], marker_names].to_numpy()

        region_mean_non_mask_data = np.mean(region_non_mask_data, axis=0)

        region_mask_data = np.append(region_mask_data, [region_mean_non_mask_data], axis=0)

        region_mask_data = np.transpose(region_mask_data)

        regions_mask_data.append(region_mask_data)

    x_tick_labels = np.array(sorted(expansion_features.index.unique("Expansion").tolist())) * pixel_interval
    x_tick_labels = x_tick_labels.tolist()
    x_tick_labels = [str(x) for x in x_tick_labels]
    x_tick_labels.append("Nonvessel Space")

    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "black"],
              [norm(-0.5), "indigo"],
              [norm(0), "firebrick"],
              [norm(0.5), "orange"],
              [norm(1.0), "khaki"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # Heatmaps Output

    hm = Heatmap(data=all_mask_data,
                 color_map=cmap,
                 x_tick_labels=x_tick_labels,
                 y_tick_labels=marker_names,
                 x_axis_label="Distance Expanded (%s)" % data_resolution_units,
                 )

    hm.make_figure(marker_clusters=marker_clusters,
                   cluster=False)

    hm.savefig_or_show(heatmaps_dir, "All_Points")

    hm_cluster = Heatmap(data=all_mask_data,
                         color_map=cmap,
                         x_tick_labels=x_tick_labels,
                         y_tick_labels=marker_names,
                         x_axis_label="Distance Expanded (%s)" % data_resolution_units,
                         )

    hm_cluster.make_figure(marker_clusters=marker_clusters,
                           cluster=True)

    hm_cluster.savefig_or_show(clustermaps_dir, "All_Points")

    for idx, region_data in enumerate(regions_mask_data):
        region_name = brain_region_names[idx]

        hm = Heatmap(data=region_data,
                     color_map=cmap,
                     x_tick_labels=x_tick_labels,
                     y_tick_labels=marker_names,
                     x_axis_label="Distance Expanded (%s)" % data_resolution_units,
                     )

        hm.make_figure(marker_clusters=marker_clusters,
                       cluster=False)

        hm.savefig_or_show(heatmaps_dir, "All_Points")

        hm_cluster = Heatmap(data=region_data,
                             color_map=cmap,
                             x_tick_labels=x_tick_labels,
                             y_tick_labels=marker_names,
                             x_axis_label="Distance Expanded (%s)" % data_resolution_units,
                             )

        hm_cluster.make_figure(marker_clusters=marker_clusters,
                               cluster=True)

        hm_cluster.savefig_or_show(clustermaps_dir, "%s_Region" % region_name)
