import datetime
import math
from collections import Collection

import matplotlib
import matplotlib.pylab as pl
import seaborn as sns
from matplotlib import patches, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import median
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from seaborn.utils import axis_ticklabels_overlap

from src.data_loading.mibi_reader import MIBIReader
from src.data_preprocessing.object_extractor import ObjectExtractor
from src.data_preprocessing.markers_feature_gen import *
from src.utils.utils_functions import mkdir_p

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


def round_to_nearest_half(number):
    """
    Round float to nearest 0.5
    :param number: float, number to round
    :return: float, number rounded to nearest 0.5
    """
    return round(number * 2) / 2


def save_figure(save_dir: str,
                fig_name: str,
                ax=None):
    """
    Save Figure

    :return:
    """

    mkdir_p(save_dir)

    if ax is not None:
        ax.savefig(save_dir + '/%s.png' % fig_name, bbox_inches='tight')
    else:
        plt.savefig(save_dir + '/%s.png' % fig_name, bbox_inches='tight')

    plt.clf()


class Visualizer:

    def __init__(self,
                 config: Config,
                 all_samples_features: pd.DataFrame,
                 markers_names: list,
                 all_feeds_contour_data: pd.DataFrame,
                 all_feeds_metadata: pd.DataFrame,
                 all_points_marker_data: np.array,
                 ):

        """
        Create kept vs. removed vessel expression comparison using Box Plots

        :param markers_names: array_like, [n_points, n_markers] -> Names of markers
        :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]] ->
        list of marker data for each point
        """

        self.config = config
        self.all_samples_features = all_samples_features
        self.markers_names = markers_names
        self.all_feeds_contour_data = all_feeds_contour_data
        self.all_feeds_metadata = all_feeds_metadata
        self.all_feeds_data = all_points_marker_data

    def vessel_region_plots(self, n_expansions: int, **kwargs):
        """
        Create vessel region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """
        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        output_dir = "%s/Line Plots Per Vessel" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir, str(round_to_nearest_half((n_expansions) *
                                                                                  self.config.pixel_interval *
                                                                                  self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        marker_color_dict = {}
        for marker_cluster in marker_clusters.keys():
            for marker in marker_clusters[marker_cluster]:
                marker_color_dict[marker] = colors[marker_cluster]

        perbin_marker_color_dict = {}
        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2

            for marker, marker_name in enumerate(marker_clusters[key]):
                perbin_marker_color_dict[marker_name] = colors_clusters[color_idx]
                color_idx += 1

        idx = pd.IndexSlice
        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      :n_expansions,
                                                      "Data"], :]

        plot_features = pd.melt(plot_features,
                                id_vars=self.config.non_marker_vars,
                                ignore_index=False)

        plot_features = plot_features.rename(columns={'variable': 'Marker',
                                                      'value': 'Expression'})

        plot_features.reset_index(level=['Expansion'], inplace=True)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(marker_clusters[key]):
                plot_features.loc[plot_features["Marker"] == marker_name, "Marker Label"] = key

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        for point in self.config.vessel_line_plots_points:
            point_dir = "%s/Point %s" % (output_dir, str(point))
            mkdir_p(point_dir)

            all_bins_dir = "%s/All Bins" % point_dir
            average_bins_dir = "%s/Average Bins" % point_dir
            per_bin_dir = "%s/Per Bin" % point_dir

            mkdir_p(all_bins_dir)
            mkdir_p(average_bins_dir)
            mkdir_p(per_bin_dir)

            idx = pd.IndexSlice
            n_vessels = len(self.all_samples_features.loc[idx[point,
                                                          :,
                                                          0,
                                                          "Data"], self.markers_names].to_numpy())
            for vessel in range(n_vessels):
                vessel_features = plot_features.loc[idx[point,
                                                        vessel,
                                                        "Data"], :]
                plt.figure(figsize=(22, 10))

                # Average Bins
                g = sns.lineplot(data=vessel_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker Label",
                                 style=self.config.primary_categorical_splitter,
                                 size=self.config.secondary_categorical_splitter,
                                 palette=self.config.line_plots_bin_colors,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                plt.savefig(average_bins_dir + '/Vessel_ID_%s.png' % str(vessel), bbox_inches='tight')
                plt.clf()

                # All Bins
                g = sns.lineplot(data=vessel_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker",
                                 palette=marker_color_dict,
                                 ci=None,
                                 legend=False)
                for key in marker_clusters.keys():
                    g.plot([], [], color=colors[key], label=key)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                plt.savefig(all_bins_dir + '/Vessel_ID_%s.png' % str(vessel), bbox_inches='tight')
                plt.clf()

                for key in marker_clusters.keys():
                    bin_dir = "%s/%s" % (per_bin_dir, key)
                    mkdir_p(bin_dir)

                    bin_features = vessel_features.loc[vessel_features["Marker Label"] == key]
                    # Per Bin
                    g = sns.lineplot(data=bin_features,
                                     x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                     y="Expression",
                                     hue="Marker",
                                     style=self.config.primary_categorical_splitter,
                                     size=self.config.secondary_categorical_splitter,
                                     palette=perbin_marker_color_dict,
                                     ci=None)

                    box = g.get_position()
                    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                    plt.savefig(bin_dir + '/Vessel_ID_%s.png' % str(vessel), bbox_inches='tight')
                    plt.clf()

    def point_region_plots(self, n_expansions: int, **kwargs):
        """
        Create point region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        output_dir = "%s/Line Plots Per Point" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir,
                                            str(round_to_nearest_half((n_expansions) *
                                                                      self.config.pixel_interval *
                                                                      self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        marker_color_dict = {}
        for marker_cluster in marker_clusters.keys():
            for marker in marker_clusters[marker_cluster]:
                marker_color_dict[marker] = colors[marker_cluster]

        perbin_marker_color_dict = {}
        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2

            for marker, marker_name in enumerate(marker_clusters[key]):
                perbin_marker_color_dict[marker_name] = colors_clusters[color_idx]
                color_idx += 1

        idx = pd.IndexSlice
        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      :n_expansions,
                                                      "Data"], :]

        plot_features = pd.melt(plot_features,
                                id_vars=self.config.non_marker_vars,
                                ignore_index=False)

        plot_features = plot_features.rename(columns={'variable': 'Marker',
                                                      'value': 'Expression'})

        plot_features.reset_index(level=['Expansion'], inplace=True)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(marker_clusters[key]):
                plot_features.loc[plot_features["Marker"] == marker_name, "Marker Label"] = key

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        for point in range(self.config.n_points):
            point_dir = "%s/Point %s" % (output_dir, str(point + 1))
            mkdir_p(point_dir)

            per_marker_dir = "%s/Per Marker" % point_dir
            mkdir_p(per_marker_dir)

            idx = pd.IndexSlice

            point_features = plot_features.loc[idx[point + 1,
                                               :,
                                               "Data"], :]
            plt.figure(figsize=(22, 10))

            # Average Bins
            g = sns.lineplot(data=point_features,
                             x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue="Marker Label",
                             style=self.config.primary_categorical_splitter,
                             palette=self.config.line_plots_bin_colors,
                             ci=None)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            plt.savefig(point_dir + '/Average_Bins.png', bbox_inches='tight')
            plt.clf()

            # All Bins
            g = sns.lineplot(data=point_features,
                             x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue="Marker",
                             palette=marker_color_dict,
                             ci=None,
                             legend=False)
            for key in marker_clusters.keys():
                g.plot([], [], color=colors[key], label=key)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            plt.savefig(point_dir + '/All_Bins.png', bbox_inches='tight')
            plt.clf()

            for key in marker_clusters.keys():
                bin_features = point_features.loc[point_features["Marker Label"] == key]
                # Per Bin
                g = sns.lineplot(data=bin_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker",
                                 style=self.config.primary_categorical_splitter,
                                 palette=perbin_marker_color_dict,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                plt.savefig(point_dir + '/%s.png' % str(key), bbox_inches='tight')
                plt.clf()

                for marker in marker_clusters[key]:
                    marker_features = plot_features.loc[plot_features["Marker"] == marker]
                    g = sns.lineplot(data=marker_features,
                                     x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                     y="Expression",
                                     hue="Marker",
                                     style=self.config.primary_categorical_splitter,
                                     size=self.config.secondary_categorical_splitter,
                                     palette=perbin_marker_color_dict,
                                     ci=None)

                    box = g.get_position()
                    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                    plt.savefig(per_marker_dir + '/%s.png' % str(marker), bbox_inches='tight')
                    plt.clf()

    def all_points_plots(self, n_expansions: int, **kwargs):
        """
        Create all points average region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        marker_color_dict = {}
        for marker_cluster in marker_clusters.keys():
            for marker in marker_clusters[marker_cluster]:
                marker_color_dict[marker] = colors[marker_cluster]

        perbin_marker_color_dict = {}
        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2

            for marker, marker_name in enumerate(marker_clusters[key]):
                perbin_marker_color_dict[marker_name] = colors_clusters[color_idx]
                color_idx += 1

        idx = pd.IndexSlice
        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      :n_expansions,
                                                      "Data"], :]

        plot_features = pd.melt(plot_features,
                                id_vars=self.config.non_marker_vars,
                                ignore_index=False)

        plot_features = plot_features.rename(columns={'variable': 'Marker',
                                                      'value': 'Expression'})

        plot_features.reset_index(level=['Expansion'], inplace=True)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(marker_clusters[key]):
                plot_features.loc[plot_features["Marker"] == marker_name, "Marker Label"] = key

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        output_dir = "%s/Line Plots All Points Average" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir,
                                            str(round_to_nearest_half((n_expansions) *
                                                                      self.config.pixel_interval *
                                                                      self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        per_marker_dir = "%s/Per Marker" % output_dir
        mkdir_p(per_marker_dir)

        per_bin_dir = "%s/Per Bin" % output_dir
        mkdir_p(per_bin_dir)

        plt.figure(figsize=(22, 10))

        # Average Bins
        g = sns.lineplot(data=plot_features,
                         x="Distance Expanded (%s)" % self.config.data_resolution_units,
                         y="Expression",
                         hue="Marker Label",
                         style=self.config.primary_categorical_splitter,
                         palette=self.config.line_plots_bin_colors,
                         ci=None)

        box = g.get_position()
        g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        plt.savefig(output_dir + '/Average_Bins.png', bbox_inches='tight')
        plt.clf()

        # All Bins
        g = sns.lineplot(data=plot_features,
                         x="Distance Expanded (%s)" % self.config.data_resolution_units,
                         y="Expression",
                         hue="Marker",
                         palette=marker_color_dict,
                         ci=None,
                         legend=False)
        for key in marker_clusters.keys():
            g.plot([], [], color=colors[key], label=key)

        box = g.get_position()
        g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        plt.savefig(output_dir + '/All_Bins.png', bbox_inches='tight')
        plt.clf()

        for key in marker_clusters.keys():
            bin_features = plot_features.loc[plot_features["Marker Label"] == key]
            # Per Bin
            g = sns.lineplot(data=bin_features,
                             x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue="Marker",
                             style=self.config.primary_categorical_splitter,
                             palette=perbin_marker_color_dict,
                             ci=None)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            plt.savefig(per_bin_dir + '/%s.png' % str(key), bbox_inches='tight')
            plt.clf()

            for marker in marker_clusters[key]:
                marker_features = plot_features.loc[plot_features["Marker"] == marker]
                g = sns.lineplot(data=marker_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker",
                                 style=self.config.primary_categorical_splitter,
                                 size=self.config.secondary_categorical_splitter,
                                 palette=perbin_marker_color_dict,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                plt.savefig(per_marker_dir + '/%s.png' % str(marker), bbox_inches='tight')
                plt.clf()

    def brain_region_plots(self, n_expansions: int,
                           **kwargs):
        """
        Create brain region average region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        marker_color_dict = {}
        for marker_cluster in marker_clusters.keys():
            for marker in marker_clusters[marker_cluster]:
                marker_color_dict[marker] = colors[marker_cluster]

        perbin_marker_color_dict = {}
        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2

            for marker, marker_name in enumerate(marker_clusters[key]):
                perbin_marker_color_dict[marker_name] = colors_clusters[color_idx]
                color_idx += 1

        idx = pd.IndexSlice
        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      :n_expansions,
                                                      "Data"], :]

        plot_features = pd.melt(plot_features,
                                id_vars=self.config.non_marker_vars,
                                ignore_index=False)

        plot_features = plot_features.rename(columns={'variable': 'Marker',
                                                      'value': 'Expression'})

        plot_features.reset_index(level=['Expansion', 'Point'], inplace=True)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(marker_clusters[key]):
                plot_features.loc[plot_features["Marker"] == marker_name, "Marker Label"] = key

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        plot_features['Region'] = pd.cut(plot_features['Point'],
                                         bins=[self.config.brain_region_point_ranges[0][0] - 1,
                                               self.config.brain_region_point_ranges[1][0] - 1,
                                               self.config.brain_region_point_ranges[2][0] - 1,
                                               float('Inf')],
                                         labels=[self.config.brain_region_names[0],
                                                 self.config.brain_region_names[1],
                                                 self.config.brain_region_names[2]])

        output_dir = "%s/Line Plots Per Region" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir,
                                            str(round_to_nearest_half((n_expansions) *
                                                                      self.config.pixel_interval *
                                                                      self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        for region in self.config.brain_region_names:
            region_dir = "%s/%s" % (output_dir, region)
            mkdir_p(region_dir)

            per_bin_dir = "%s/Per Bin" % region_dir
            mkdir_p(per_bin_dir)

            per_marker_dir = "%s/Per Marker" % region_dir
            mkdir_p(per_marker_dir)

            region_features = plot_features.loc[plot_features["Region"] == region]

            plt.figure(figsize=(22, 10))

            # Average Bins
            g = sns.lineplot(data=region_features,
                             x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue="Marker Label",
                             style=self.config.primary_categorical_splitter,
                             palette=self.config.line_plots_bin_colors,
                             ci=None)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            plt.savefig(region_dir + '/Average_Bins.png', bbox_inches='tight')
            plt.clf()

            # All Bins
            g = sns.lineplot(data=region_features,
                             x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue="Marker",
                             palette=marker_color_dict,
                             ci=None,
                             legend=False)
            for key in marker_clusters.keys():
                g.plot([], [], color=colors[key], label=key)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            plt.savefig(region_dir + '/All_Bins.png', bbox_inches='tight')
            plt.clf()

            for key in marker_clusters.keys():
                bin_features = region_features.loc[region_features["Marker Label"] == key]
                # Per Bin
                g = sns.lineplot(data=bin_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker",
                                 style=self.config.primary_categorical_splitter,
                                 palette=perbin_marker_color_dict,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                plt.savefig(per_bin_dir + '/%s.png' % str(key), bbox_inches='tight')
                plt.clf()

                for marker in marker_clusters[key]:
                    marker_features = plot_features.loc[plot_features["Marker"] == marker]
                    g = sns.lineplot(data=marker_features,
                                     x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                     y="Expression",
                                     hue="Marker",
                                     style=self.config.primary_categorical_splitter,
                                     size=self.config.secondary_categorical_splitter,
                                     palette=perbin_marker_color_dict,
                                     ci=None)

                    box = g.get_position()
                    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                    plt.savefig(per_marker_dir + '/%s.png' % str(marker), bbox_inches='tight')
                    plt.clf()

    def obtain_expanded_vessel_masks(self, **kwargs):
        """
        Create expanded region vessel masks

        :return:
        """

        expansion_upper_bound = kwargs.get('expansion_upper_bound', 60)

        parent_dir = "%s/Expanded Vessel Masks" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        parent_dir = "%s/%s %s" % (parent_dir,
                                   int(expansion_upper_bound * self.config.pixels_to_distance),
                                   self.config.data_resolution_units)
        mkdir_p(parent_dir)

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():

            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            distinct_dir = "%s/Original Mask Excluded" % output_dir
            mkdir_p(distinct_dir)

            nondistinct_dir = "%s/Original Mask Included" % output_dir
            mkdir_p(nondistinct_dir)

            for idx in range(self.config.n_points):
                point_contours = feed_data.loc[idx, "Contours"].contours

                original_not_included_point_mask = np.zeros(self.config.segmentation_mask_size, np.uint8)
                original_included_point_mask = np.zeros(self.config.segmentation_mask_size, np.uint8)

                regions = get_assigned_regions(point_contours, self.config.segmentation_mask_size)

                for vessel_idx, vessel in enumerate(point_contours):
                    original_not_included_mask = expand_vessel_region(vessel,
                                                                      self.config.segmentation_mask_size,
                                                                      upper_bound=expansion_upper_bound,
                                                                      lower_bound=0.5)

                    original_not_included_mask = cv.bitwise_and(original_not_included_mask,
                                                                regions[vessel_idx].astype(np.uint8))

                    original_included_mask = expand_vessel_region(vessel,
                                                                  self.config.segmentation_mask_size,
                                                                  upper_bound=expansion_upper_bound)

                    original_included_mask = cv.bitwise_and(original_included_mask,
                                                            regions[vessel_idx].astype(np.uint8))

                    original_not_included_point_mask = np.bitwise_or(original_not_included_point_mask,
                                                                     original_not_included_mask)

                    original_included_point_mask = np.bitwise_or(original_included_point_mask,
                                                                 original_included_mask)

                im = Image.fromarray(original_not_included_point_mask * 255)
                im.save(os.path.join(distinct_dir, "Point%s.tif" % str(idx + 1)))

                im = Image.fromarray(original_included_point_mask * 255)
                im.save(os.path.join(nondistinct_dir, "Point%s.tif" % str(idx + 1)))

    def obtain_embedded_vessel_masks(self, **kwargs):
        """
        Create expanded region vessel masks

        :return:
        """

        expansion_upper_bound = kwargs.get('expansion_upper_bound', 60)

        parent_dir = "%s/Embedded Vessel Masks" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        parent_dir = "%s/%s %s" % (parent_dir,
                                   int(expansion_upper_bound * self.config.pixels_to_distance),
                                   self.config.data_resolution_units)
        mkdir_p(parent_dir)

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():

            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            distinct_dir = "%s/Original Mask Excluded" % output_dir
            mkdir_p(distinct_dir)

            nondistinct_dir = "%s/Original Mask Included" % output_dir
            mkdir_p(nondistinct_dir)

            for idx in range(self.config.n_points):
                original_not_included_point_mask = np.zeros(self.config.segmentation_mask_size, np.uint8)
                original_included_point_mask = np.zeros(self.config.segmentation_mask_size, np.uint8)

                point_contours = feed_data.loc[idx, "Contours"].contours

                regions = get_assigned_regions(point_contours, self.config.segmentation_mask_size)

                for vessel_idx, vessel in enumerate(point_contours):
                    original_not_included_mask = expand_vessel_region(vessel,
                                                                      self.config.segmentation_mask_size,
                                                                      upper_bound=expansion_upper_bound,
                                                                      lower_bound=0.5)

                    original_not_included_mask = cv.bitwise_and(original_not_included_mask,
                                                                regions[vessel_idx].astype(np.uint8))

                    original_included_mask = expand_vessel_region(vessel,
                                                                  self.config.segmentation_mask_size,
                                                                  upper_bound=expansion_upper_bound)

                    original_included_mask = cv.bitwise_and(original_included_mask,
                                                            regions[vessel_idx].astype(np.uint8))

                    original_not_included_point_mask[np.where(original_not_included_mask == 1)] = vessel_idx + 1
                    original_included_point_mask[np.where(original_included_mask == 1)] = vessel_idx + 1

                im = Image.fromarray(original_not_included_point_mask)
                im.save(os.path.join(distinct_dir, "Point%s.tif" % str(idx + 1)))

                im = Image.fromarray(original_included_point_mask)
                im.save(os.path.join(nondistinct_dir, "Point%s.tif" % str(idx + 1)))

    def pixel_expansion_ring_plots(self, **kwargs):
        """
        Pixel Expansion Ring Plots

        """

        n_expansions = self.config.max_expansions
        interval = self.config.pixel_interval
        n_points = self.config.n_points
        expansions = self.config.expansion_to_run

        parent_dir = "%s/Ring Plots" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():

            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            for point_num in range(n_points):
                current_interval = interval

                expansion_image = np.zeros(self.config.segmentation_mask_size, np.uint8)

                colors = pl.cm.Greys(np.linspace(0, 1, n_expansions + 10))

                for x in range(n_expansions):
                    per_point_vessel_contours = feed_data.loc[point_num, "Contours"].contours
                    expansion_ring_plots(per_point_vessel_contours,
                                         expansion_image,
                                         pixel_expansion_upper_bound=current_interval,
                                         pixel_expansion_lower_bound=current_interval - interval,
                                         color=colors[x + 5] * 255)

                    if x + 1 in expansions:
                        child_dir = parent_dir + "/%s%s Expansion" % (str(round_to_nearest_half((n_expansions) *
                                                                                                self.config.pixel_interval *
                                                                                                self.config.pixels_to_distance)),
                                                                      self.config.data_resolution_units)
                        mkdir_p(child_dir)

                        cv.imwrite(child_dir + "/Point%s.png" % str(point_num + 1), expansion_image)

                    current_interval += interval

    def expression_histogram(self, **kwargs):
        """
        Histogram for visualizing Marker Expressions

        :return:
        """

        output_dir = "%s/Expression Histograms" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        idx = pd.IndexSlice

        x = "SMA"
        x_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], x].values

        plt.hist(x_data, density=True, bins=30, label="Data")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 301)
        kde = gaussian_kde(x_data)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.legend(loc="upper left")
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title("Histogram")
        plt.savefig(output_dir + "/%s" % x)

    def biaxial_scatter_plot(self, **kwargs):
        """
        Biaxial Scatter Plot for visualizing Marker Expressions

        :return:
        """
        output_dir = "%s/Biaxial Scatter Plots" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        idx = pd.IndexSlice

        x = "SMA"
        y = "GLUT1"

        x_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], x].values
        y_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], y].values

        positive_sma = len(self.all_samples_features.loc[self.all_samples_features[x] > 0.1].values)
        all_vess = len(self.all_samples_features.values)

        logging.debug("There are %s / %s vessels which are positive for SMA" % (positive_sma, all_vess))

        # Calculate the point density
        xy = np.vstack([x_data, y_data])
        z = gaussian_kde(xy)(xy)

        plt.scatter(x_data, y_data, c=z, s=35, edgecolor='')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title("%s vs %s" % (x, y))

        plt.savefig(output_dir + '/%s_%s.png' % (x, y), bbox_inches='tight')
        plt.clf()

        x = "SMA"
        y = "CD31"

        x_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], x].values
        y_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], y].values

        positive_sma = len(self.all_samples_features.loc[self.all_samples_features[x] > 0.1].values)
        all_vess = len(self.all_samples_features.values)

        logging.debug("There are %s / %s vessels which are positive for SMA" % (positive_sma, all_vess))

        # Calculate the point density
        xy = np.vstack([x_data, y_data])
        z = gaussian_kde(xy)(xy)

        plt.scatter(x_data, y_data, c=z, s=35, edgecolor='')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title("%s vs %s" % (x, y))

        plt.savefig(output_dir + '/%s_%s.png' % (x, y), bbox_inches='tight')
        plt.clf()

        x = "SMA"
        y = "vWF"

        x_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], x].values
        y_data = self.all_samples_features.loc[idx[:, :, 0, "Data"], y].values

        positive_sma = len(self.all_samples_features.loc[self.all_samples_features[x] > 0.1].values)
        all_vess = len(self.all_samples_features.values)

        logging.debug("There are %s / %s vessels which are positive for SMA" % (positive_sma, all_vess))

        # Calculate the point density
        xy = np.vstack([x_data, y_data])
        z = gaussian_kde(xy)(xy)

        plt.scatter(x_data, y_data, c=z, s=35, edgecolor='')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title("%s vs %s" % (x, y))

        plt.savefig(output_dir + '/%s_%s.png' % (x, y), bbox_inches='tight')
        plt.clf()

    def categorical_violin_plot(self, **kwargs):
        """
        Categorical Violin Plot
        """

        mask_type = kwargs.get('mask_type', "expansion_only")

        parent_dir = "%s/Categorical Violin Plots" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        marker_clusters = self.config.marker_clusters

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)
            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            if mask_type == "mask_only":
                feed_features = feed_features.loc[idx[:, :, :-1, "Data"], :]
            elif mask_type == "mask_and_expansion":
                feed_features = feed_features.loc[idx[:, :, :, "Data"], :]
            elif mask_type == "expansion_only":
                feed_features = feed_features.loc[idx[:, :, 0:, "Data"], :]

            feed_features = pd.melt(feed_features,
                                    id_vars=self.config.non_marker_vars,
                                    ignore_index=False)

            feed_features = feed_features.rename(columns={'variable': 'Marker',
                                                          'value': 'Expression'})

            feed_features['Size'] = pd.cut(feed_features['Contour Area'],
                                           bins=[self.config.small_vessel_threshold,
                                                 self.config.medium_vessel_threshold,
                                                 self.config.large_vessel_threshold,
                                                 float('Inf')],
                                           labels=["Small",
                                                   "Medium",
                                                   "Large"])

            split_dir = "%s/By %s" % (feed_dir, self.config.primary_categorical_splitter)
            mkdir_p(split_dir)

            cmap = colors.ListedColormap(['blue', 'red'])(np.linspace(0, 1, 2))

            for key in marker_clusters.keys():
                marker_cluster_dir = "%s/%s" % (split_dir, key)
                mkdir_p(marker_cluster_dir)

                for marker, marker_name in enumerate(marker_clusters[key]):
                    marker_features = feed_features[(feed_features["Marker"] == marker_name)]

                    plt.figure(figsize=(22, 10))

                    ax = sns.violinplot(x="Size",
                                        y="Expression",
                                        hue=self.config.primary_categorical_splitter,
                                        inner="box",
                                        data=marker_features,
                                        bw=0.2,
                                        palette=cmap)

                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                               title=self.config.primary_categorical_splitter)

                    plt.savefig(marker_cluster_dir + '/%s.png' % str(marker_name),
                                bbox_inches='tight')
                    plt.clf()

            non_split_dir = "%s/Without Split" % feed_dir
            mkdir_p(non_split_dir)

            for key in marker_clusters.keys():
                marker_cluster_dir = "%s/%s" % (non_split_dir, key)
                mkdir_p(marker_cluster_dir)

                for marker, marker_name in enumerate(marker_clusters[key]):
                    marker_features = feed_features[(feed_features["Marker"] == marker_name)]

                    plt.figure(figsize=(10, 10))

                    ax = sns.violinplot(x=self.config.primary_categorical_splitter,
                                        y="Expression",
                                        inner="box",
                                        data=marker_features,
                                        bw=0.2,
                                        palette=cmap)

                    plt.savefig(marker_cluster_dir + '/%s.png' % str(marker_name),
                                bbox_inches='tight')
                    plt.clf()

    def violin_plot_brain_expansion(self, n_expansions: int, **kwargs):
        """
        Violin Plots for Expansion Data

        :param n_expansions: int, Number of expansions

        :return:
        """

        dist_upper_end = 1.75

        output_dir = "%s/Expansion Violin Plots" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        bins_dir = "%s/Per Bin" % output_dir
        mkdir_p(bins_dir)

        per_bin_expansions_dir = "%s/%s%s Expansion" % (bins_dir,
                                                        str(round_to_nearest_half(n_expansions *
                                                                                  self.config.pixel_interval *
                                                                                  self.config.pixels_to_distance)),
                                                        self.config.data_resolution_units)
        mkdir_p(per_bin_expansions_dir)

        markers_dir = "%s/Per Marker" % output_dir
        mkdir_p(markers_dir)

        per_marker_expansions_dir = "%s/%s%s Expansion" % (markers_dir,
                                                           str(round_to_nearest_half(n_expansions *
                                                                                     self.config.pixel_interval *
                                                                                     self.config.pixels_to_distance)),
                                                           self.config.data_resolution_units)
        mkdir_p(per_marker_expansions_dir)

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        marker_color_dict = {}
        for marker_cluster in marker_clusters.keys():
            for marker in marker_clusters[marker_cluster]:
                marker_color_dict[marker] = colors[marker_cluster]

        perbin_marker_color_dict = {}
        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2

            for marker, marker_name in enumerate(marker_clusters[key]):
                perbin_marker_color_dict[marker_name] = colors_clusters[color_idx]
                color_idx += 1

        idx = pd.IndexSlice
        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      :n_expansions,
                                                      "Data"], :]

        plot_features = pd.melt(plot_features,
                                id_vars=self.config.non_marker_vars,
                                ignore_index=False)

        plot_features = plot_features.rename(columns={'variable': 'Marker',
                                                      'value': 'Expression'})

        plot_features.reset_index(level=['Expansion', 'Point'], inplace=True)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(marker_clusters[key]):
                plot_features.loc[plot_features["Marker"] == marker_name, "Marker Label"] = key

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        plot_features['Region'] = pd.cut(plot_features['Point'],
                                         bins=[self.config.brain_region_point_ranges[0][0] - 1,
                                               self.config.brain_region_point_ranges[1][0] - 1,
                                               self.config.brain_region_point_ranges[2][0] - 1,
                                               float('Inf')],
                                         labels=[self.config.brain_region_names[0],
                                                 self.config.brain_region_names[1],
                                                 self.config.brain_region_names[2]])

        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, 6))[3:]

            for marker, marker_name in enumerate(marker_clusters[key]):
                marker_features = plot_features[(plot_features["Marker"] == marker_name)]

                max_expression = np.max(marker_features["Expression"].values)

                plt.figure(figsize=(22, 10))

                if max_expression < dist_upper_end:
                    plt.ylim(-0.15, dist_upper_end)
                else:
                    plt.ylim(-0.15, max_expression)

                ax = sns.violinplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                    y="Expression",
                                    hue=self.config.primary_categorical_splitter,
                                    palette=colors_clusters,
                                    inner=None,
                                    data=marker_features,
                                    bw=0.2)

                if self.config.primary_categorical_splitter is None:
                    sns.pointplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                  y="Expression",
                                  hue="Region", data=marker_features,
                                  markers=["o", "x", "^"],
                                  join=False
                                  )

                plt.savefig(per_marker_expansions_dir + '/%s.png' % str(marker_name),
                            bbox_inches='tight')
                plt.clf()

        for key in marker_clusters.keys():
            marker_features = plot_features.loc[plot_features["Marker Label"] == key]

            colors_clusters = color_maps[key](np.linspace(0, 1, 6))[3:]

            plt.figure(figsize=(22, 10))

            if max_expression < dist_upper_end:
                plt.ylim(-0.15, dist_upper_end)
            else:
                plt.ylim(-0.15, max_expression)

            ax = sns.violinplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                y="Expression",
                                hue=self.config.primary_categorical_splitter,
                                palette=colors_clusters,
                                inner=None,
                                data=marker_features,
                                bw=0.2)
            if self.config.primary_categorical_splitter is None:
                sns.pointplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                              y="Expression",
                              hue="Region",
                              data=marker_features,
                              markers=["o", "x", "^"],
                              join=False
                              )

            plt.savefig(per_bin_expansions_dir + '/%s.png' % str(key),
                        bbox_inches='tight')
            plt.clf()

    def box_plot_brain_expansions(self, n_expansions: int, **kwargs):
        """
        Box Plots for Expansion Data

        :param n_expansions: int, Number of expansions

        :return:
        """

        dist_upper_end = 1.75

        output_dir = "%s/Expansion Box Plots" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        bins_dir = "%s/Per Bin" % output_dir
        mkdir_p(bins_dir)

        per_bin_expansions_dir = "%s/%s%s Expansion" % (bins_dir,
                                                        str(round_to_nearest_half(n_expansions *
                                                                                  self.config.pixel_interval *
                                                                                  self.config.pixels_to_distance)),
                                                        self.config.data_resolution_units)
        mkdir_p(per_bin_expansions_dir)

        markers_dir = "%s/Per Marker" % output_dir
        mkdir_p(markers_dir)

        per_marker_expansions_dir = "%s/%s%s Expansion" % (markers_dir,
                                                           str(round_to_nearest_half(n_expansions *
                                                                                     self.config.pixel_interval *
                                                                                     self.config.pixels_to_distance)),
                                                           self.config.data_resolution_units)
        mkdir_p(per_marker_expansions_dir)

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        marker_color_dict = {}
        for marker_cluster in marker_clusters.keys():
            for marker in marker_clusters[marker_cluster]:
                marker_color_dict[marker] = colors[marker_cluster]

        perbin_marker_color_dict = {}
        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2

            for marker, marker_name in enumerate(marker_clusters[key]):
                perbin_marker_color_dict[marker_name] = colors_clusters[color_idx]
                color_idx += 1

        idx = pd.IndexSlice
        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      :n_expansions,
                                                      "Data"], :]

        plot_features = pd.melt(plot_features,
                                id_vars=self.config.non_marker_vars,
                                ignore_index=False)

        plot_features = plot_features.rename(columns={'variable': 'Marker',
                                                      'value': 'Expression'})

        plot_features.reset_index(level=['Expansion', 'Point'], inplace=True)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(marker_clusters[key]):
                plot_features.loc[plot_features["Marker"] == marker_name, "Marker Label"] = key

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        plot_features['Region'] = pd.cut(plot_features['Point'],
                                         bins=[self.config.brain_region_point_ranges[0][0] - 1,
                                               self.config.brain_region_point_ranges[1][0] - 1,
                                               self.config.brain_region_point_ranges[2][0] - 1,
                                               float('Inf')],
                                         labels=[self.config.brain_region_names[0],
                                                 self.config.brain_region_names[1],
                                                 self.config.brain_region_names[2]])

        for key in marker_clusters.keys():
            colors_clusters = color_maps[key](np.linspace(0, 1, 6))[3:]

            for marker, marker_name in enumerate(marker_clusters[key]):
                marker_features = plot_features[(plot_features["Marker"] == marker_name)]

                max_expression = np.max(marker_features["Expression"].values)

                plt.figure(figsize=(22, 10))

                if max_expression < dist_upper_end:
                    plt.ylim(-0.15, dist_upper_end)
                else:
                    plt.ylim(-0.15, max_expression)

                ax = sns.boxplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue=self.config.primary_categorical_splitter,
                                 palette=colors_clusters,
                                 data=marker_features)

                plt.savefig(per_marker_expansions_dir + '/%s.png' % str(marker_name),
                            bbox_inches='tight')
                plt.clf()

        for key in marker_clusters.keys():
            marker_features = plot_features.loc[plot_features["Marker Label"] == key]

            colors_clusters = color_maps[key](np.linspace(0, 1, 6))[3:]

            plt.figure(figsize=(22, 10))

            if max_expression < dist_upper_end:
                plt.ylim(-0.15, dist_upper_end)
            else:
                plt.ylim(-0.15, max_expression)

            ax = sns.boxplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue=self.config.primary_categorical_splitter,
                             palette=colors_clusters,
                             data=marker_features)

            plt.savefig(per_bin_expansions_dir + '/%s.png' % str(key),
                        bbox_inches='tight')
            plt.clf()

    def spatial_probability_maps(self, **kwargs):
        """
        Spatial Probability Maps

        :return:
        """

        parent_dir = "%s/Pixel Expression Spatial Maps" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            vessels_dir = "%s/Vessels" % output_dir
            mkdir_p(vessels_dir)

            astrocytes_dir = "%s/Astrocytes" % output_dir
            mkdir_p(astrocytes_dir)

            all_markers_dir = "%s/All Markers" % output_dir
            mkdir_p(all_markers_dir)

            feed_contours = self.all_feeds_contour_data.loc[feed_idx]

            for point_idx in range(self.config.n_points):

                marker_data = self.all_feeds_data[feed_idx, point_idx]

                marker_dict = dict(zip(self.markers_names, marker_data))
                data = []

                for marker in self.config.mask_marker_clusters["Vessels"]:
                    data.append(marker_dict[marker])

                data = np.nanmean(np.array(data), axis=0)
                blurred_data = gaussian_filter(data, sigma=4)
                color_map = plt.imshow(blurred_data)
                color_map.set_cmap("jet")
                plt.colorbar()
                plt.savefig("%s/Point%s" % (vessels_dir, str(point_idx + 1)))
                plt.clf()

                point_contours = feed_contours.loc[point_idx, "Contours"].contours

                point_dir = "%s/Point %s" % (vessels_dir, str(point_idx + 1))
                mkdir_p(point_dir)

                for contour_idx, c in enumerate(point_contours):
                    mask = np.zeros(self.config.segmentation_mask_size, np.uint8)
                    cv.drawContours(mask, [c], -1, (1, 1, 1), cv.FILLED)

                    my_cm = matplotlib.cm.get_cmap('jet')
                    normed_data = (blurred_data - np.min(blurred_data)) / (np.max(blurred_data) - np.min(blurred_data))
                    mapped_data = my_cm(normed_data)

                    result = (cv.bitwise_and(mapped_data, mapped_data, mask=mask) * 255).astype("uint8")
                    result = cv.cvtColor(result, cv.COLOR_RGBA2RGB)

                    x, y, w, h = cv.boundingRect(c)
                    roi = result[y:y + h, x:x + w]

                    color_map = plt.imshow(roi)
                    divider = make_axes_locatable(plt.gca())
                    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=matplotlib.cm.jet, orientation='vertical')
                    plt.gcf().add_axes(ax_cb)
                    plt.savefig(os.path.join(point_dir, "Vessel_ID_%s.png" % str(contour_idx + 1)))

                    plt.clf()

            for point_idx in range(self.config.n_points):

                marker_data = self.all_feeds_data[feed_idx, point_idx]

                marker_dict = dict(zip(self.markers_names, marker_data))
                data = []

                for marker in self.config.marker_clusters["Astrocytes"]:
                    data.append(marker_dict[marker])

                data = np.nanmean(np.array(data), axis=0)
                blurred_data = gaussian_filter(data, sigma=4)
                color_map = plt.imshow(blurred_data)
                color_map.set_cmap("jet")
                plt.colorbar()

                point_contours = feed_contours.loc[point_idx, "Contours"].contours

                plt.savefig("%s/Point%s" % (astrocytes_dir, str(point_idx + 1)))
                plt.clf()

                point_dir = "%s/Point %s" % (astrocytes_dir, str(point_idx + 1))
                mkdir_p(point_dir)

                for contour_idx, c in enumerate(point_contours):
                    mask = np.zeros(self.config.segmentation_mask_size, np.uint8)
                    cv.drawContours(mask, [c], -1, (1, 1, 1), cv.FILLED)

                    my_cm = matplotlib.cm.get_cmap('jet')
                    normed_data = (blurred_data - np.min(blurred_data)) / (np.max(blurred_data) - np.min(blurred_data))
                    mapped_data = my_cm(normed_data)

                    result = (cv.bitwise_and(mapped_data, mapped_data, mask=mask) * 255).astype("uint8")
                    result = cv.cvtColor(result, cv.COLOR_RGBA2RGB)

                    x, y, w, h = cv.boundingRect(c)
                    roi = result[y:y + h, x:x + w]

                    color_map = plt.imshow(roi)
                    divider = make_axes_locatable(plt.gca())
                    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=matplotlib.cm.jet, orientation='vertical')
                    plt.gcf().add_axes(ax_cb)
                    plt.savefig(os.path.join(point_dir, "Vessel_ID_%s.png" % str(contour_idx + 1)))

                    plt.clf()

            for point_idx in range(self.config.n_points):

                marker_data = self.all_feeds_data[feed_idx, point_idx]

                marker_dict = dict(zip(self.markers_names, marker_data))

                point_dir = "%s/Point%s" % (all_markers_dir, str(point_idx + 1))
                mkdir_p(point_dir)

                for key in marker_dict.keys():
                    marker_name = key
                    data = marker_dict[key]

                    blurred_data = gaussian_filter(data, sigma=4)
                    color_map = plt.imshow(blurred_data)
                    color_map.set_cmap("jet")
                    plt.colorbar()

                    point_contours = feed_contours.loc[point_idx, "Contours"].contours

                    for contour_idx, c in enumerate(point_contours):
                        M = cv.moments(c)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        plt.text(cX, cY, "ID: %s" % str(contour_idx + 1), fontsize='xx-small', color="w")

                    plt.savefig("%s/%s" % (point_dir, marker_name))
                    plt.clf()

    def vessel_images_by_categorical_variable(self, **kwargs):
        """
        Vessel Images by Categorical Variable
        :return:
        """

        assert self.config.primary_categorical_splitter is not None, "There must be a primary categorical splitter"

        parent_dir = "%s/%s Vessel Images" % (self.config.visualization_results_dir,
                                              self.config.primary_categorical_splitter)

        img_shape = self.config.segmentation_mask_size

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():
            feed_data = self.all_feeds_contour_data.loc[feed_idx]
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            for val in feed_features[self.config.primary_categorical_splitter].unique():

                output_dir = "%s/%s" % (feed_dir, val)
                mkdir_p(output_dir)

                split_features = feed_features[feed_features[self.config.primary_categorical_splitter] == val]

                for i in split_features.index:
                    point_idx = i[0]
                    cnt_idx = i[1]

                    cnt = feed_data.loc[point_idx - 1, "Contours"].contours[cnt_idx]

                    example_mask = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
                    cv.drawContours(example_mask, [cnt], -1, (255, 255, 255), cv.FILLED)

                    cv.imwrite(os.path.join(output_dir,
                                            "Point_Num_%s_Vessel_ID_%s.png" % (str(point_idx),
                                                                               str(cnt_idx + 1))),
                               example_mask)

    def _scatter_plot_color_bar(self,
                                figure_name,
                                output_dir,
                                data,
                                x="UMAP0",
                                y="UMAP1",
                                hue="SMA",
                                cmap="coolwarm",
                                min_val=None,
                                max_val=None):
        """
        Scatter Plot with Colorbar Mapped to Continuous Third Variable

        :param figure_name: str, Figure name for plot
        :param output_dir: str, Output directory for plot
        :param x: str, X axis column name in dataframe
        :param y: str, Y axis column name in dataframe
        :param hue: str, Third variable column name in dataframe
        :param cmap: Union[str, Matplotlib.Colormap], Color map for Scatter Plot
        :return:
        """

        if min_val is None:
            min_val = data[hue].min()

        if max_val is None:
            max_val = data[hue].max()

        mkdir_p(output_dir)

        plt.figure(figsize=(15, 10))

        ax = data.plot.scatter(x=x,
                               y=y,
                               c=hue,
                               colormap=cmap,
                               vmin=min_val,
                               vmax=max_val)

        plt.savefig(output_dir + '/%s.png' % figure_name)
        plt.clf()

    def scatter_plot_umap_marker_projection(self, **kwargs):
        """
        UMAP projection with marker channel colorbar

        :return:
        """

        inward_expansions_only = kwargs.get('inward_expansions_only', True)

        if inward_expansions_only:
            plot_features = self.all_samples_features.loc[pd.IndexSlice[:, :, :-1, "Data"], :]
        else:
            plot_features = self.all_samples_features.loc[pd.IndexSlice[:, :, :, "Data"], :]

        output_dir = self.config.visualization_results_dir + "/UMAP Scatter Plot Projection"

        for marker_cluster in self.config.marker_clusters.keys():
            plot_features[marker_cluster] = \
                plot_features.loc[pd.IndexSlice[:,
                                  :,
                                  :,
                                  :], self.config.marker_clusters[marker_cluster]].mean(axis=1)

        for marker_cluster in self.config.marker_clusters.keys():

            self._scatter_plot_color_bar(marker_cluster,
                                         output_dir + "/Marker Clusters",
                                         plot_features,
                                         hue=marker_cluster,
                                         min_val=0,
                                         max_val=1)

            for marker in self.config.marker_clusters[marker_cluster]:
                self._scatter_plot_color_bar(marker,
                                             output_dir + "/Individual Markers",
                                             plot_features,
                                             hue=marker)

        self._scatter_plot_color_bar("umap_projection_by_size",
                                     output_dir + "/Size",
                                     plot_features,
                                     hue="Contour Area",
                                     min_val=self.config.small_vessel_threshold,
                                     max_val=1000)

        plot_features.reset_index(level=['Point'], inplace=True)

        plot_features['Region'] = pd.cut(plot_features['Point'],
                                         bins=[self.config.brain_region_point_ranges[0][0] - 1,
                                               self.config.brain_region_point_ranges[1][0] - 1,
                                               self.config.brain_region_point_ranges[2][0] - 1,
                                               float('Inf')],
                                         labels=[self.config.brain_region_names[0],
                                                 self.config.brain_region_names[1],
                                                 self.config.brain_region_names[2]])

        region_dir = output_dir + "/Region"
        mkdir_p(region_dir)

        g = sns.scatterplot(data=plot_features,
                            x="UMAP0",
                            y="UMAP1",
                            hue="Region",
                            ci=None,
                            palette="tab20")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)

        plt.savefig(region_dir + '/umap_projection_by_region.png', bbox_inches='tight')
        plt.clf()

    def _vessel_nonvessel_heatmap(self,
                                  n_expansions: int,
                                  feed_features: pd.DataFrame,
                                  brain_regions: list,
                                  marker_clusters: list,
                                  feed_dir: str):
        """
        Vessel/Non-vessel heatmaps helper method

        :param n_expansions:
        :param feed_features:
        :return:
        """

        # Vessel Space (SMA Positive)
        positve_sma = feed_features.loc[
            feed_features["SMA"] >= self.config.SMA_positive_threshold]

        idx = pd.IndexSlice

        try:
            all_vessels_sma_data = positve_sma.loc[idx[:, :,
                                                   :n_expansions,
                                                   "Data"], self.markers_names].to_numpy()
            all_vessels_sma_data = np.mean(all_vessels_sma_data, axis=0)
        except KeyError:
            all_vessels_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            mfg_vessels_sma_data = positve_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                   :,
                                                   :n_expansions,
                                                   "Data"], self.markers_names].to_numpy()
            mfg_vessels_sma_data = np.mean(mfg_vessels_sma_data, axis=0)

        except KeyError:
            mfg_vessels_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            hip_vessels_sma_data = positve_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                   :,
                                                   :n_expansions,
                                                   "Data"], self.markers_names].to_numpy()
            hip_vessels_sma_data = np.mean(hip_vessels_sma_data, axis=0)
        except KeyError:
            hip_vessels_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            caud_vessels_sma_data = positve_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                    :,
                                                    :n_expansions,
                                                    "Data"], self.markers_names].to_numpy()
            caud_vessels_sma_data = np.mean(caud_vessels_sma_data, axis=0)

        except KeyError:
            caud_vessels_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        # Vessel Space (SMA Negative)
        negative_sma = feed_features.loc[
            feed_features["SMA"] < self.config.SMA_positive_threshold]

        idx = pd.IndexSlice

        try:
            all_vessels_non_sma_data = negative_sma.loc[idx[:, :,
                                                        :n_expansions,
                                                        "Data"], self.markers_names].to_numpy()
            all_vessels_non_sma_data = np.mean(all_vessels_non_sma_data, axis=0)

        except KeyError:
            all_vessels_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            mfg_vessels_non_sma_data = negative_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                        :,
                                                        :n_expansions,
                                                        "Data"], self.markers_names].to_numpy()
            mfg_vessels_non_sma_data = np.mean(mfg_vessels_non_sma_data, axis=0)

        except KeyError:
            mfg_vessels_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            hip_vessels_non_sma_data = negative_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                        :,
                                                        :n_expansions,
                                                        "Data"], self.markers_names].to_numpy()
            hip_vessels_non_sma_data = np.mean(hip_vessels_non_sma_data, axis=0)

        except KeyError:
            hip_vessels_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            caud_vessels_non_sma_data = negative_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                         :,
                                                         :n_expansions,
                                                         "Data"], self.markers_names].to_numpy()
            caud_vessels_non_sma_data = np.mean(caud_vessels_non_sma_data, axis=0)

        except KeyError:
            caud_vessels_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        # Non-vessel Space

        try:
            all_nonmask_sma_data = positve_sma.loc[idx[:, :, :,
                                                   "Non-Vascular Space"], self.markers_names].to_numpy()
            all_nonmask_sma_data = np.mean(all_nonmask_sma_data, axis=0)

        except KeyError:
            all_nonmask_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            mfg_nonmask_sma_data = positve_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                   :,
                                                   :n_expansions,
                                                   "Non-Vascular Space"], self.markers_names].to_numpy()
            mfg_nonmask_sma_data = np.mean(mfg_nonmask_sma_data, axis=0)

        except KeyError:
            mfg_nonmask_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            hip_nonmask_sma_data = positve_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                   :,
                                                   :n_expansions,
                                                   "Non-Vascular Space"], self.markers_names].to_numpy()
            hip_nonmask_sma_data = np.mean(hip_nonmask_sma_data, axis=0)

        except KeyError:
            hip_nonmask_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            caud_nonmask_sma_data = positve_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                    :,
                                                    :n_expansions,
                                                    "Non-Vascular Space"], self.markers_names].to_numpy()
            caud_nonmask_sma_data = np.mean(caud_nonmask_sma_data, axis=0)

        except KeyError:
            caud_nonmask_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            all_nonmask_non_sma_data = negative_sma.loc[idx[:, :, :,
                                                        "Non-Vascular Space"], self.markers_names].to_numpy()
            all_nonmask_non_sma_data = np.mean(all_nonmask_non_sma_data, axis=0)

        except KeyError:
            all_nonmask_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            mfg_nonmask_non_sma_data = negative_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                        :,
                                                        :n_expansions,
                                                        "Non-Vascular Space"], self.markers_names].to_numpy()
            mfg_nonmask_non_sma_data = np.mean(mfg_nonmask_non_sma_data, axis=0)

        except KeyError:
            mfg_nonmask_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            hip_nonmask_non_sma_data = negative_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                        :,
                                                        :n_expansions,
                                                        "Non-Vascular Space"], self.markers_names].to_numpy()
            hip_nonmask_non_sma_data = np.mean(hip_nonmask_non_sma_data, axis=0)

        except KeyError:
            hip_nonmask_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            caud_nonmask_non_sma_data = negative_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                         :,
                                                         :n_expansions,
                                                         "Non-Vascular Space"], self.markers_names].to_numpy()
            caud_nonmask_non_sma_data = np.mean(caud_nonmask_non_sma_data, axis=0)

        except KeyError:
            caud_nonmask_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        # Vessel environment space

        try:
            all_vessels_environment_sma_data = positve_sma.loc[idx[:, :,
                                                               :n_expansions,
                                                               "Vascular Space"], self.markers_names].to_numpy()
            all_vessels_environment_sma_data = np.mean(all_vessels_environment_sma_data, axis=0)

        except KeyError:
            all_vessels_environment_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            mfg_vessels_environment_sma_data = positve_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                               :,
                                                               :n_expansions,
                                                               "Vascular Space"], self.markers_names].to_numpy()
            mfg_vessels_environment_sma_data = np.mean(mfg_vessels_environment_sma_data, axis=0)

        except KeyError:
            mfg_vessels_environment_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            hip_vessels_environment_sma_data = positve_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                               :,
                                                               :n_expansions,
                                                               "Vascular Space"], self.markers_names].to_numpy()
            hip_vessels_environment_sma_data = np.mean(hip_vessels_environment_sma_data, axis=0)

        except KeyError:
            hip_vessels_environment_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            caud_vessels_environment_sma_data = positve_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                                :,
                                                                :n_expansions,
                                                                "Vascular Space"], self.markers_names].to_numpy()
            caud_vessels_environment_sma_data = np.mean(caud_vessels_environment_sma_data, axis=0)

        except KeyError:
            caud_vessels_environment_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            all_vessels_environment_non_sma_data = negative_sma.loc[idx[:, :,
                                                                    :n_expansions,
                                                                    "Vascular Space"], self.markers_names].to_numpy()
            all_vessels_environment_non_sma_data = np.mean(all_vessels_environment_non_sma_data, axis=0)

        except KeyError:
            all_vessels_environment_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            mfg_vessels_environment_non_sma_data = negative_sma.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                                    :,
                                                                    :n_expansions,
                                                                    "Vascular Space"], self.markers_names].to_numpy()
            mfg_vessels_environment_non_sma_data = np.mean(mfg_vessels_environment_non_sma_data, axis=0)

        except KeyError:
            mfg_vessels_environment_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            hip_vessels_environment_non_sma_data = negative_sma.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                                    :,
                                                                    :n_expansions,
                                                                    "Vascular Space"], self.markers_names].to_numpy()
            hip_vessels_environment_non_sma_data = np.mean(hip_vessels_environment_non_sma_data, axis=0)

        except KeyError:
            hip_vessels_environment_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

        try:
            caud_vessels_environment_non_sma_data = negative_sma.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                                     :,
                                                                     :n_expansions,
                                                                     "Vascular Space"], self.markers_names].to_numpy()
            caud_vessels_environment_non_sma_data = np.mean(caud_vessels_environment_non_sma_data, axis=0)

        except KeyError:
            caud_vessels_environment_non_sma_data = np.zeros((self.config.n_markers,), np.uint8)

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

        plt.figure(figsize=(22, 10))

        ax = sns.heatmap(all_data,
                         cmap=cmap,
                         xticklabels=self.markers_names,
                         yticklabels=yticklabels,
                         linewidths=0,
                         )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        v_line_idx = 0

        for key in marker_clusters.keys():
            if v_line_idx != 0:
                ax.axvline(v_line_idx, 0, len(yticklabels), linewidth=3, c='w')

            for _ in marker_clusters[key]:
                v_line_idx += 1

        h_line_idx = 0

        while h_line_idx < len(yticklabels):
            h_line_idx += 6
            ax.axhline(h_line_idx, 0, len(self.markers_names), linewidth=3, c='w')

        output_dir = "%s/Heatmaps" % feed_dir
        mkdir_p(output_dir)

        plt.savefig(output_dir + '/Expansion_%s.png' % str((n_expansions)), bbox_inches='tight')
        plt.clf()

        ax = sns.clustermap(all_data,
                            cmap=cmap,
                            row_cluster=False,
                            col_cluster=True,
                            linewidths=0,
                            xticklabels=self.markers_names,
                            yticklabels=yticklabels,
                            figsize=(20, 10)
                            )

        output_dir = "%s/Clustermaps" % feed_dir
        mkdir_p(output_dir)

        ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        ax.ax_heatmap.yaxis.tick_left()
        ax.ax_heatmap.yaxis.set_label_position("left")

        ax.savefig(output_dir + '/Expansion_%s.png' % str(n_expansions))
        plt.clf()

    def continuous_scatter_plot(self, **kwargs):
        """
        Plot marker expressions as a function of some continuous variable

        :return:
        """

        analysis_variable = kwargs.get('analysis_variable', "Asymmetry Score")
        mask_type = kwargs.get('mask_type', "expansion_only")

        assert self.config.primary_categorical_splitter is not None, "Must have a primary categorical variable"
        assert self.config.secondary_categorical_splitter is not None, "Must have a secondary categorical variable"

        parent_dir = "%s/%s Scatter Plots" % (self.config.visualization_results_dir, analysis_variable)
        mkdir_p(parent_dir)

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            by_primary_analysis_variable_dir = "%s/By %s" % (feed_dir, self.config.primary_categorical_splitter)
            mkdir_p(by_primary_analysis_variable_dir)

            by_secondary_analysis_variable_dir = "%s/By %s" % (feed_dir, self.config.secondary_categorical_splitter)
            mkdir_p(by_secondary_analysis_variable_dir)

            with_vessel_id_dir = "%s/%s" % (feed_dir, "With Vessel ID")
            mkdir_p(with_vessel_id_dir)

            secondary_separate_dir = "%s/Separate By %s" % (feed_dir, self.config.secondary_categorical_splitter)
            mkdir_p(secondary_separate_dir)

            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            if mask_type == "mask_only":
                feed_features = feed_features.loc[idx[:, :, :-1, "Data"], :]
            elif mask_type == "mask_and_expansion":
                feed_features = feed_features.loc[idx[:, :, :, "Data"], :]
            elif mask_type == "expansion_only":
                feed_features = feed_features.loc[idx[:, :, 0:, "Data"], :]

            feed_features = pd.melt(feed_features,
                                    id_vars=self.config.non_marker_vars,
                                    ignore_index=False)

            feed_features = feed_features.rename(columns={'variable': 'Marker',
                                                          'value': 'Expression'})

            feed_features['Mean Expression'] = \
                feed_features.groupby(['Vessel', 'Point', 'Marker'])['Expression'].transform('mean')

            feed_features.reset_index(level=['Vessel', 'Point', 'Expansion'], inplace=True)

            feed_features = feed_features[feed_features["Expansion"] == feed_features["Expansion"].max()]

            cmap = colors.ListedColormap(['blue', 'red'])(np.linspace(0, 1, 2))

            for marker in feed_features["Marker"].unique():
                marker_features = feed_features[feed_features["Marker"] == marker]

                g = sns.scatterplot(data=marker_features,
                                    x=analysis_variable,
                                    y="Mean Expression",
                                    hue=self.config.primary_categorical_splitter,
                                    ci=None,
                                    palette=cmap)

                plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                           borderaxespad=0.,
                           title=self.config.primary_categorical_splitter)

                plt.savefig(by_primary_analysis_variable_dir + '/%s.png' % str(marker), bbox_inches='tight')
                plt.clf()

            cmap = colors.ListedColormap(['cyan', 'pink', 'yellow'])(np.linspace(0, 1, 3))

            for marker in feed_features["Marker"].unique():
                marker_features = feed_features[feed_features["Marker"] == marker]

                g = sns.scatterplot(data=marker_features,
                                    x=analysis_variable,
                                    y="Mean Expression",
                                    hue=self.config.secondary_categorical_splitter,
                                    ci=None,
                                    palette=cmap,
                                    edgecolor='k',
                                    linewidth=1)

                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                           title=self.config.secondary_categorical_splitter)

                plt.savefig(by_secondary_analysis_variable_dir + '/%s.png' % str(marker), bbox_inches='tight')
                plt.clf()

            for marker in feed_features["Marker"].unique():
                marker_features = feed_features[feed_features["Marker"] == marker]

                g = sns.scatterplot(data=marker_features,
                                    x=analysis_variable,
                                    y="Mean Expression",
                                    ci=None,
                                    palette="tab20")

                for line in range(0, 50):
                    point_label = str(marker_features["Point"].values[line]) + ":" \
                                  + str(marker_features["Vessel"].values[line])

                    if not math.isnan(marker_features[analysis_variable].values[line]) \
                            and not math.isnan(marker_features["Mean Expression"].values[line]):
                        g.text(marker_features[analysis_variable].values[line],
                               marker_features["Mean Expression"].values[line],
                               point_label, horizontalalignment='left',
                               size='medium', color='black', weight='semibold')

                plt.savefig(with_vessel_id_dir + '/%s.png' % str(marker), bbox_inches='tight')
                plt.clf()

            for split_val in feed_features[self.config.secondary_categorical_splitter].unique():
                out_dir = "%s/%s" % (secondary_separate_dir, split_val)
                mkdir_p(out_dir)

                split_features = feed_features[feed_features[self.config.secondary_categorical_splitter] == split_val]

                for marker in split_features["Marker"].unique():
                    marker_features = split_features[split_features["Marker"] == marker]

                    plt.xlim([-0.05, 1.05])

                    g = sns.scatterplot(data=marker_features,
                                        x=analysis_variable,
                                        y="Mean Expression",
                                        hue=self.config.primary_categorical_splitter,
                                        ci=None,
                                        palette="tab20")

                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                               title=self.config.primary_categorical_splitter)

                    plt.savefig(out_dir + '/%s.png' % str(marker), bbox_inches='tight')
                    plt.clf()

    def vessel_nonvessel_heatmap(self, n_expansions: int, **kwargs):
        """
        Vessel/Non-vessel heatmaps for marker expression

        :param n_expansions: int, Number of expansions
        :return:
        """
        brain_regions = self.config.brain_region_point_ranges
        marker_clusters = self.config.marker_clusters

        parent_dir = "%s/Heatmaps & Clustermaps" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            if self.config.primary_categorical_splitter is None:
                self._vessel_nonvessel_heatmap(n_expansions, feed_features, brain_regions, marker_clusters, feed_dir)
            else:
                for i in feed_features[self.config.primary_categorical_splitter].unique():
                    split_dir = "%s/%s: %s" % (feed_dir, self.config.primary_categorical_splitter, i)
                    mkdir_p(split_dir)

                    split_features = feed_features.loc[feed_features[self.config.primary_categorical_splitter] == i]

                    self._vessel_nonvessel_heatmap(n_expansions, split_features, brain_regions, marker_clusters,
                                                   split_dir)

    def _categorical_split_expansion_heatmap_helper(self,
                                                    n_expansions: int,
                                                    expansion_features: pd.DataFrame,
                                                    heatmaps_dir: str,
                                                    primary_splitter: str,
                                                    secondary_splitter: str = None,
                                                    marker: str = "Astrocytes",
                                                    cluster=True
                                                    ):
        """
        Helper method to generate categorical split expansion heatmap

        :param n_expansions: int, Number of expansions
        :param expansion_features: pd.DataFrame, Features across all expansions
        :param heatmaps_dir: str, Directory of heatmaps
        :return:
        """
        title = marker

        if cluster:
            marker = self.config.marker_clusters[marker]

        heatmap_data = []
        y_tick_labels = []
        pixel_interval = round_to_nearest_half(abs(self.config.pixel_interval) * self.config.pixels_to_distance)

        classes_to_ignore = ["NA"]

        for split_val in expansion_features[self.config.primary_categorical_splitter].unique():

            if split_val in classes_to_ignore:
                continue

            y_lab_split = "%s : %s" % (primary_splitter, split_val)
            y_tick_labels.append(y_lab_split)

            split_features = expansion_features.loc[expansion_features[
                                                        self.config.primary_categorical_splitter]
                                                    == split_val]

            curr_split_data = []

            for i in sorted(expansion_features.index.unique("Expansion").tolist()):

                try:
                    curr_expansion = split_features.loc[pd.IndexSlice[:,
                                                        :,
                                                        i,
                                                        "Data"],
                                                        marker].to_numpy()
                except KeyError:
                    curr_expansion = np.array([])

                if curr_expansion.size > 0:
                    curr_split_data.append(np.mean(np.array(curr_expansion)))
                else:
                    curr_split_data.append(0)

            heatmap_data.append(curr_split_data)

        if secondary_splitter is not None:

            for split_idx, secondary_split_val in \
                    enumerate(expansion_features[self.config.secondary_categorical_splitter].unique()):
                secondary_split_features = expansion_features.loc[expansion_features[
                                                                      self.config.secondary_categorical_splitter]
                                                                  == secondary_split_val]

                for split_val in expansion_features[self.config.primary_categorical_splitter].unique():

                    if split_val in classes_to_ignore:
                        continue

                    y_lab_split_first_level = "%s : %s" % (primary_splitter, split_val)

                    split_features = secondary_split_features.loc[expansion_features[
                                                                      self.config.primary_categorical_splitter]
                                                                  == split_val]

                    curr_split_data = []

                    for i in sorted(expansion_features.index.unique("Expansion").tolist()):
                        y_lab_split_second_level = "%s : %s" % (secondary_splitter, secondary_split_val)

                        try:
                            curr_expansion = split_features.loc[pd.IndexSlice[:,
                                                                :,
                                                                i,
                                                                "Data"],
                                                                marker].to_numpy()
                        except KeyError:
                            curr_expansion = np.array([])

                        if curr_expansion.size > 0:
                            curr_split_data.append(np.mean(np.array(curr_expansion)))
                        else:
                            curr_split_data.append(0)

                    heatmap_data.append(curr_split_data)

                    y_tick_labels.append("%s  %s" % (y_lab_split_first_level, y_lab_split_second_level))

        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1.0), "black"],
                  [norm(-0.5), "indigo"],
                  [norm(0), "firebrick"],
                  [norm(0.5), "orange"],
                  [norm(1.0), "khaki"]]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        plt.figure(figsize=(22, 10))

        x_tick_labels = np.array(sorted(expansion_features.index.unique("Expansion").tolist())) * pixel_interval
        x_tick_labels = x_tick_labels.tolist()
        x_tick_labels = [str(x) for x in x_tick_labels]

        ax = sns.heatmap(heatmap_data,
                         cmap=cmap,
                         xticklabels=x_tick_labels,
                         yticklabels=y_tick_labels,
                         linewidths=0,
                         )

        ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

        if axis_ticklabels_overlap(ax.get_xticklabels()):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        plt.xlabel("Distance Expanded (%s)" % self.config.data_resolution_units)
        plt.title("%s" % title)

        output_dir = "%s/%s%s Expansion" % (heatmaps_dir,
                                            str(round_to_nearest_half(n_expansions *
                                                                      self.config.pixel_interval *
                                                                      self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        plt.savefig(output_dir + '/%s.png' % title, bbox_inches='tight')
        plt.clf()

    def categorical_split_expansion_heatmap_clustermap(self,
                                                       n_expansions: int,
                                                       **kwargs):
        """
        Categorically split expansion heatmap

        :param n_expansions: int, Number of expansions to run
        :return:
        """

        parent_dir = "%s/Categorical Expansion Heatmaps & Clustermaps" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        assert self.config.primary_categorical_splitter is not None, "No categorical splitter selected!"

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            expansion_features = feed_features.loc[idx[:, :, :n_expansions, :], :]

            heatmaps_dir = "%s/Categorical Split Expansion Heatmaps" % feed_dir
            mkdir_p(heatmaps_dir)

            cluster_dir = "%s/Marker Clusters" % heatmaps_dir
            mkdir_p(cluster_dir)

            markers_dir = "%s/Individual Markers" % heatmaps_dir
            mkdir_p(markers_dir)

            for cluster in self.config.marker_clusters.keys():
                self._categorical_split_expansion_heatmap_helper(n_expansions,
                                                                 expansion_features,
                                                                 cluster_dir,
                                                                 self.config.primary_categorical_splitter,
                                                                 self.config.secondary_categorical_splitter,
                                                                 marker=cluster,
                                                                 cluster=True)

                for marker in self.config.marker_clusters[cluster]:
                    self._categorical_split_expansion_heatmap_helper(n_expansions,
                                                                     expansion_features,
                                                                     markers_dir,
                                                                     self.config.primary_categorical_splitter,
                                                                     self.config.secondary_categorical_splitter,
                                                                     marker=marker,
                                                                     cluster=False)

    def _heatmap_clustermap_generator(self,
                                      data,
                                      x_tick_labels,
                                      x_label,
                                      cmap,
                                      marker_clusters,
                                      output_dir,
                                      map_name,
                                      cluster=False,
                                      y_tick_labels=False):

        """
        Helper method to save heatmap and clustermap output

        :param data: array_like, data to plot
        :param x_tick_labels: list[str], x tick labels
        :param x_label: str, x-axis label
        :param cmap: cmap, Colour map
        :param marker_clusters: dict, Marker cluster names
        :param output_dir: str, Output directory
        :param map_name: str, Heatmap/Clustermap name
        :param cluster: bool, should use clustermap?
        :return:
        """

        if y_tick_labels is None:
            y_tick_labels = self.markers_names

        if not cluster:

            plt.figure(figsize=(22, 10))

            ax = sns.heatmap(data,
                             cmap=cmap,
                             xticklabels=x_tick_labels,
                             yticklabels=y_tick_labels,
                             linewidths=0,
                             vmin=0,
                             vmax=1
                             )

            ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

            if axis_ticklabels_overlap(ax.get_xticklabels()):
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            if len(x_tick_labels) > 100:
                xticks = ax.get_xticklabels()

                x_tick_values = [0.00, 0.25, 0.5, 0.75, 1.0]

                for xtick in xticks:
                    xtick.set_visible(False)

                for i, val in enumerate(x_tick_labels):
                    if val in x_tick_values:
                        xticks[i].set_visible(True)
                        x_tick_values.remove(val)

            plt.xlabel("%s" % x_label)

            h_line_idx = 0

            for key in marker_clusters.keys():
                if h_line_idx != 0:
                    ax.axhline(h_line_idx, 0, len(self.markers_names), linewidth=3, c='w')

                for _ in marker_clusters[key]:
                    h_line_idx += 1

            plt.savefig(output_dir + '/%s.png' % map_name, bbox_inches='tight')
            plt.clf()

        else:
            ax = sns.clustermap(data,
                                cmap=cmap,
                                row_cluster=True,
                                col_cluster=False,
                                linewidths=0,
                                xticklabels=x_tick_labels,
                                yticklabels=y_tick_labels,
                                figsize=(20, 10),
                                vmin=0,
                                vmax=1
                                )
            ax_ax = ax.ax_heatmap
            ax_ax.set_xlabel("%s" % x_label)

            ax_ax.set_xticklabels(ax_ax.get_xticklabels(), rotation="horizontal")

            if axis_ticklabels_overlap(ax_ax.get_xticklabels()):
                ax_ax.set_xticklabels(ax_ax.get_xticklabels(), rotation=45, ha="right")

            if len(x_tick_labels) > 100:
                xticks = ax_ax.get_xticklabels()

                x_tick_values = [0.00, 0.25, 0.5, 0.75, 1.0]

                for xtick in xticks:
                    xtick.set_visible(False)

                for i, val in enumerate(x_tick_labels):
                    if val in x_tick_values:
                        xticks[i].set_visible(True)
                        x_tick_values.remove(val)

            ax.savefig(output_dir + '/%s.png' % map_name)
            plt.clf()

    def _brain_region_expansion_heatmap(self,
                                        n_expansions: int,
                                        expansion_features: pd.DataFrame,
                                        heatmaps_dir: str,
                                        clustermaps_dir: str
                                        ):
        """
        Brain region expansion heatmaps helper method

        :param n_expansions: int, Number of expansions to run
        :return:
        """
        idx = pd.IndexSlice

        all_mask_data = []
        mfg_mask_data = []
        hip_mask_data = []
        caud_mask_data = []
        pixel_interval = round_to_nearest_half(abs(self.config.pixel_interval) * self.config.pixels_to_distance)

        brain_regions = self.config.brain_region_point_ranges
        marker_clusters = self.config.marker_clusters

        for i in sorted(expansion_features.index.unique("Expansion").tolist()):
            try:
                current_expansion_all = expansion_features.loc[idx[:, :,
                                                               i,
                                                               "Data"], self.markers_names].to_numpy()
            except KeyError:
                current_expansion_all = np.array([])

            try:
                current_expansion_mfg = expansion_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                               :,
                                                               i,
                                                               "Data"], self.markers_names].to_numpy()
            except KeyError:
                current_expansion_mfg = np.array([])

            try:
                current_expansion_hip = expansion_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                               :,
                                                               i,
                                                               "Data"], self.markers_names].to_numpy()
            except KeyError:
                current_expansion_hip = np.array([])

            try:
                current_expansion_caud = expansion_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                                :,
                                                                i,
                                                                "Data"], self.markers_names].to_numpy()
            except KeyError:
                current_expansion_caud = np.array([])

            if current_expansion_all.size > 0:
                all_mask_data.append(np.mean(np.array(current_expansion_all), axis=0))
            else:
                all_mask_data.append(np.zeros((self.config.n_markers,), np.uint8))

            if current_expansion_mfg.size > 0:
                mfg_mask_data.append(np.mean(np.array(current_expansion_mfg), axis=0))
            else:
                mfg_mask_data.append(np.zeros((self.config.n_markers,), np.uint8))

            if current_expansion_hip.size > 0:
                hip_mask_data.append(np.mean(np.array(current_expansion_hip), axis=0))
            else:
                hip_mask_data.append(np.zeros((self.config.n_markers,), np.uint8))

            if current_expansion_caud.size > 0:
                caud_mask_data.append(np.mean(np.array(current_expansion_caud), axis=0))
            else:
                caud_mask_data.append(np.zeros((self.config.n_markers,), np.uint8))

        all_mask_data = np.array(all_mask_data)
        mfg_mask_data = np.array(mfg_mask_data)
        hip_mask_data = np.array(hip_mask_data)
        caud_mask_data = np.array(caud_mask_data)

        idx = pd.IndexSlice
        all_nonmask_data = expansion_features.loc[idx[:, :,
                                                  i,
                                                  "Non-Vascular Space"], self.markers_names].to_numpy()
        mfg_nonmask_data = expansion_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                  :,
                                                  i,
                                                  "Non-Vascular Space"], self.markers_names].to_numpy()
        hip_nonmask_data = expansion_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                  :,
                                                  i,
                                                  "Non-Vascular Space"], self.markers_names].to_numpy()
        caud_nonmask_data = expansion_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                   :,
                                                   i,
                                                   "Non-Vascular Space"], self.markers_names].to_numpy()

        mean_nonmask_data = np.mean(all_nonmask_data, axis=0)

        mfg_nonmask_data = np.mean(mfg_nonmask_data, axis=0)

        hip_nonmask_data = np.mean(hip_nonmask_data, axis=0)

        caud_nonmask_data = np.mean(caud_nonmask_data, axis=0)

        all_mask_data = np.append(all_mask_data, [mean_nonmask_data], axis=0)
        mfg_mask_data = np.append(mfg_mask_data, [mfg_nonmask_data], axis=0)
        hip_mask_data = np.append(hip_mask_data, [hip_nonmask_data], axis=0)
        caud_mask_data = np.append(caud_mask_data, [caud_nonmask_data], axis=0)

        all_mask_data = np.transpose(all_mask_data)
        mfg_mask_data = np.transpose(mfg_mask_data)
        hip_mask_data = np.transpose(hip_mask_data)
        caud_mask_data = np.transpose(caud_mask_data)

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

        output_dir = "%s/%s%s Expansion" % (heatmaps_dir,
                                            str(round_to_nearest_half(n_expansions *
                                                                      self.config.pixel_interval *
                                                                      self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        self._heatmap_clustermap_generator(data=all_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="All_Points",
                                           cluster=False)

        self._heatmap_clustermap_generator(data=mfg_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="MFG_Region",
                                           cluster=False)

        self._heatmap_clustermap_generator(data=hip_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="HIP_Region",
                                           cluster=False)

        self._heatmap_clustermap_generator(data=caud_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="CAUD_Region",
                                           cluster=False)

        # Clustermaps Outputs

        output_dir = "%s/%s%s Expansion" % (clustermaps_dir, str(round_to_nearest_half(n_expansions *
                                                                                       self.config.pixel_interval *
                                                                                       self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(output_dir)

        self._heatmap_clustermap_generator(data=all_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="All_Points",
                                           cluster=True)

        self._heatmap_clustermap_generator(data=mfg_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="MFG_Region",
                                           cluster=True)

        self._heatmap_clustermap_generator(data=hip_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="HIP_Region",
                                           cluster=True)

        self._heatmap_clustermap_generator(data=caud_mask_data,
                                           x_tick_labels=x_tick_labels,
                                           x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                           cmap=cmap,
                                           marker_clusters=marker_clusters,
                                           output_dir=output_dir,
                                           map_name="CAUD_Region",
                                           cluster=True)

    def brain_region_expansion_heatmap(self, n_expansions: int, **kwargs):
        """
        Brain Region Expansion Heatmap

        :param n_expansions: int, Number of expansions
        """
        parent_dir = "%s/Expansion Heatmaps & Clustermaps" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            expansion_features = feed_features.loc[idx[:, :, :n_expansions, :], :]

            if self.config.primary_categorical_splitter is None:
                heatmaps_dir = "%s/Expansion Heatmaps" % feed_dir
                clustermaps_dir = "%s/Expansion Clustermaps" % feed_dir

                mkdir_p(heatmaps_dir)
                mkdir_p(clustermaps_dir)

                self._brain_region_expansion_heatmap(n_expansions, expansion_features, heatmaps_dir, clustermaps_dir)
            else:
                for i in feed_features[self.config.primary_categorical_splitter].unique():
                    split_dir = "%s/%s: %s" % (feed_dir, self.config.primary_categorical_splitter, i)
                    mkdir_p(split_dir)

                    heatmaps_dir = "%s/Expansion Heatmaps" % split_dir
                    clustermaps_dir = "%s/Expansion Clustermaps" % split_dir

                    mkdir_p(heatmaps_dir)
                    mkdir_p(clustermaps_dir)

                    split_features = feed_features.loc[feed_features[self.config.primary_categorical_splitter] == i]

                    self._brain_region_expansion_heatmap(n_expansions, split_features, heatmaps_dir, clustermaps_dir)

    def marker_expression_masks(self, **kwargs):
        """
        Marker Expression Overlay Masks

        :return:
        """

        parent_dir = "%s/Expression Masks" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():

            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            for i in range(len(self.config.n_points)):
                point_contours = feed_data.loc[i, "Contours"]

                point_dir = output_dir + "/Point %s" % str(i + 1)
                mkdir_p(point_dir)

                contours = point_contours.contours
                contour_areas = point_contours.areas
                marker_data = self.all_feeds_data[feed_idx, i]

                img_shape = marker_data[0].shape

                expression_img = np.zeros(img_shape, np.uint8)
                expression_img = cv.cvtColor(expression_img, cv.COLOR_GRAY2BGR)

                data = calculate_composition_marker_expression(self.config, marker_data, contours, contour_areas,
                                                               self.markers_names)

                for marker_idx, marker_name in enumerate(self.markers_names):
                    for idx, vessel_vec in data.iterrows():
                        color = plt.get_cmap('hot')(vessel_vec[marker_idx])
                        color = (255 * color[0], 255 * color[1], 255 * color[2])

                        cv.drawContours(expression_img, contours, idx[1], color, cv.FILLED)

                    plt.imshow(expression_img)
                    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('hot'))
                    plt.colorbar(sm)
                    plt.savefig(os.path.join(point_dir, "%s.png" % marker_name))
                    plt.clf()

    def removed_vessel_expression_boxplot(self, **kwargs):
        """
        Create kept vs. removed vessel expression comparison using Box Plots
        """
        n_points = self.config.n_points

        all_points_vessels_expression = []
        all_points_removed_vessels_expression = []

        parent_dir = "%s/Kept Vs. Removed Vessel Boxplots" % self.config.visualization_results_dir
        mkdir_p(parent_dir)

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():

            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            output_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(output_dir)

            # Iterate through each point
            for i in range(n_points):
                contours = feed_data.loc[i, "Contours"].contours
                contour_areas = feed_data.loc[i, "Contours"].areas
                removed_contours = feed_data.loc[i, "Contours"].removed_contours
                removed_areas = feed_data.loc[i, "Contours"].removed_areas
                marker_data = self.all_feeds_data[feed_idx, i]

                start_expression = datetime.datetime.now()

                vessel_expression_data = calculate_composition_marker_expression(self.config,
                                                                                 marker_data,
                                                                                 contours,
                                                                                 contour_areas,
                                                                                 self.markers_names,
                                                                                 point_num=i + 1)
                removed_vessel_expression_data = calculate_composition_marker_expression(self.config,
                                                                                         marker_data,
                                                                                         removed_contours,
                                                                                         removed_areas,
                                                                                         self.markers_names,
                                                                                         point_num=i + 1)

                all_points_vessels_expression.append(vessel_expression_data)
                all_points_removed_vessels_expression.append(removed_vessel_expression_data)

                end_expression = datetime.datetime.now()

                logging.debug(
                    "Finished calculating expression for Point %s in %s" % (
                        str(i + 1), end_expression - start_expression))

            kept_vessel_expression = pd.concat(all_points_vessels_expression).fillna(0)
            kept_vessel_expression.index = map(lambda a: (a[0], a[1], a[2], a[3], "Kept"), kept_vessel_expression.index)
            removed_vessel_expression = pd.concat(all_points_removed_vessels_expression).fillna(0)
            removed_vessel_expression.index = map(lambda a: (a[0], a[1], a[2], a[3], "Removed"),
                                                  removed_vessel_expression.index)

            kept_removed_features = [kept_vessel_expression, removed_vessel_expression]
            kept_removed_features = pd.concat(kept_removed_features).fillna(0)
            kept_removed_features.index = pd.MultiIndex.from_tuples(kept_removed_features.index)
            kept_removed_features = kept_removed_features.sort_index()

            scaling_factor = self.config.scaling_factor
            transformation = self.config.transformation_type
            normalization = self.config.normalization_type
            n_markers = self.config.n_markers

            kept_removed_features = normalize_expression_data(self.config,
                                                              kept_removed_features,
                                                              self.markers_names,
                                                              transformation=transformation,
                                                              normalization=normalization,
                                                              scaling_factor=scaling_factor,
                                                              n_markers=n_markers)

            brain_region_names = self.config.brain_region_names
            brain_region_point_ranges = self.config.brain_region_point_ranges
            markers_to_show = self.config.marker_clusters["Vessels"]

            all_points_per_brain_region_dir = "%s/All Points Per Region" % parent_dir
            mkdir_p(all_points_per_brain_region_dir)

            average_points_dir = "%s/Average Per Region" % parent_dir
            mkdir_p(average_points_dir)

            all_points = "%s/All Points" % parent_dir
            mkdir_p(all_points)

            all_kept_removed_vessel_expression_data_collapsed = []

            for idx, brain_region in enumerate(brain_region_names):
                brain_region_range = brain_region_point_ranges[idx]

                vessel_removed_vessel_expression_data_collapsed = []

                idx = pd.IndexSlice
                per_point__vessel_data = kept_removed_features.loc[idx[brain_region_range[0]:brain_region_range[1], :,
                                                                   :,
                                                                   "Data",
                                                                   "Kept"],
                                                                   markers_to_show]

                per_point__removed_data = kept_removed_features.loc[idx[brain_region_range[0]:brain_region_range[1], :,
                                                                    :,
                                                                    "Data",
                                                                    "Removed"],
                                                                    markers_to_show]

                for index, vessel_data in per_point__vessel_data.iterrows():
                    data = np.mean(vessel_data)
                    vessel_removed_vessel_expression_data_collapsed.append(
                        [data, "Kept", index[0]])

                    all_kept_removed_vessel_expression_data_collapsed.append(
                        [data, "Kept", index[0]])

                for index, vessel_data in per_point__removed_data.iterrows():
                    data = np.mean(vessel_data)
                    vessel_removed_vessel_expression_data_collapsed.append([data, "Removed",
                                                                            index[0]])

                    all_kept_removed_vessel_expression_data_collapsed.append([data, "Removed",
                                                                              index[0]])

                df = pd.DataFrame(vessel_removed_vessel_expression_data_collapsed,
                                  columns=["Expression", "Vessel", "Point"])

                plt.title("Kept vs Removed Vessel Marker Expression - %s" % brain_region)
                ax = sns.boxplot(x="Point", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
                plt.savefig(os.path.join(all_points_per_brain_region_dir, "%s.png" % brain_region))
                plt.clf()

                plt.title("Kept vs Removed Vessel Marker Expression - %s: Average Points" % brain_region)
                ax = sns.boxplot(x="Vessel", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
                plt.savefig(os.path.join(average_points_dir, "%s.png" % brain_region))
                plt.clf()

            df = pd.DataFrame(all_kept_removed_vessel_expression_data_collapsed,
                              columns=["Expression", "Vessel", "Point"])

            plt.title("Kept vs Removed Vessel Marker Expression - All Points")
            ax = sns.boxplot(x="Vessel", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
            plt.savefig(os.path.join(all_points, "All_Points.png"))
            plt.clf()

    def vessel_areas_histogram(self, **kwargs):
        """
        Create visualizations of vessel areas
        """
        output_dir = "%s/Vessel Areas Histogram" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        masks = self.config.all_masks
        region_names = self.config.brain_region_names

        show_outliers = self.config.show_boxplot_outliers

        total_areas = [[], [], []]

        brain_regions = self.config.brain_region_point_ranges

        mibi_reader = MIBIReader(self.config)
        object_extractor = ObjectExtractor(self.config)

        for segmentation_type in masks:
            current_point = 1
            current_region = 0

            marker_segmentation_masks, markers_data, self.markers_names = mibi_reader.get_all_point_data()

            contour_images_multiple_points = []
            contour_data_multiple_points = []

            for segmentation_mask in marker_segmentation_masks:
                contour_images, contours, removed_contours = object_extractor.extract(segmentation_mask)
                contour_images_multiple_points.append(contour_images)
                contour_data_multiple_points.append(contours)

            vessel_areas = self.plot_vessel_areas(contour_data_multiple_points,
                                                  segmentation_type=segmentation_type,
                                                  show_outliers=show_outliers)

            for point_vessel_areas in vessel_areas:
                current_point += 1

                if not (brain_regions[current_region][0] <= current_point <= brain_regions[current_region][1]):
                    current_region += 1

                if current_region < len(total_areas):
                    total_areas[current_region].extend(sorted(point_vessel_areas))

        colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']

        fig = plt.figure(1, figsize=(9, 6))
        plt.title("All Objects Across Brain Regions - All Vessels")

        # Create an axes instance
        ax = fig.add_subplot(111)

        # Create the boxplot
        bp = ax.boxplot(total_areas, showfliers=show_outliers, patch_artist=True, labels=region_names)

        for w, region in enumerate(brain_regions):
            patch = bp['boxes'][w]
            patch.set(facecolor=colors[w])

        plt.savefig(os.path.join(output_dir, "Vessel_Areas_Histogram.png"))
        plt.clf()

    def pseudo_time_heatmap(self,
                            cmap=None,
                            **kwargs):
        """
        Pseudo-Time Heatmap

        :param cmap: Matplotlib.Colormap, Colormap to be used for plot
        :param kwargs: Keyword arguments
        :return:
        """

        output_dir = "%s/Pseudo-Time Heatmaps" % self.config.visualization_results_dir
        mkdir_p(output_dir)

        if cmap is None:
            norm = matplotlib.colors.Normalize(-1, 1)
            colors = [[norm(-1.0), "black"],
                      [norm(-0.5), "indigo"],
                      [norm(0), "firebrick"],
                      [norm(0.5), "orange"],
                      [norm(1.0), "khaki"]]

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        analysis_variable = kwargs.get('analysis_variable', "Asymmetry Score")
        n_x_ticks = kwargs.get('n_x_ticks', 9)
        mask_type = kwargs.get('mask_type', "expansion_only")

        allowance = 0.1

        assert self.config.primary_categorical_splitter is not None, "Must have a primary categorical variable"
        assert self.config.secondary_categorical_splitter is not None, "Must have a secondary categorical variable"

        parent_dir = "%s/%s Scatter Plots" % (self.config.visualization_results_dir, analysis_variable)
        mkdir_p(parent_dir)

        for feed_idx in range(self.all_feeds_data.shape[0]):
            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            by_primary_analysis_variable_dir = "%s/By %s" % (feed_dir, self.config.primary_categorical_splitter)
            mkdir_p(by_primary_analysis_variable_dir)

            by_secondary_analysis_variable_dir = "%s/By %s" % (feed_dir, self.config.secondary_categorical_splitter)
            mkdir_p(by_secondary_analysis_variable_dir)

            with_vessel_id_dir = "%s/%s" % (feed_dir, "With Vessel ID")
            mkdir_p(with_vessel_id_dir)

            secondary_separate_dir = "%s/Separate By %s" % (feed_dir, self.config.secondary_categorical_splitter)
            mkdir_p(secondary_separate_dir)

            feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

            if mask_type == "mask_only":
                feed_features = feed_features.loc[idx[:, :, :-1, "Data"], :]
            elif mask_type == "mask_and_expansion":
                feed_features = feed_features.loc[idx[:, :, :, "Data"], :]
            elif mask_type == "expansion_only":
                feed_features = feed_features.loc[idx[:, :, 0:, "Data"], :]

            feed_features = pd.melt(feed_features,
                                    id_vars=self.config.non_marker_vars,
                                    ignore_index=False)

            feed_features = feed_features.rename(columns={'variable': 'Marker',
                                                          'value': 'Expression'})

            feed_features['Mean Expression'] = \
                feed_features.groupby(['Vessel', 'Point', 'Marker'])['Expression'].transform('mean')

            feed_features.reset_index(level=['Vessel', 'Point', 'Expansion'], inplace=True)

            feed_features = feed_features[feed_features["Expansion"] == feed_features["Expansion"].max()]

            all_mask_data = []
            x_tick_labels = []

            for val in sorted(feed_features[analysis_variable].unique()):

                y_tick_labels = feed_features[feed_features[analysis_variable] == val].groupby(['Marker'],
                                                                                               sort=False)[
                    'Mean Expression'].mean().reset_index()["Marker"].values

                current_expansion_all = feed_features[feed_features[analysis_variable] == val].groupby(['Marker'],
                                                                                                       sort=False)[
                    'Mean Expression'].mean().to_numpy()

                x_tick_labels.append(round(val, 2))

                if current_expansion_all.size > 0:
                    all_mask_data.append(current_expansion_all)
                else:
                    all_mask_data.append(np.zeros((self.config.n_markers,), np.uint8))

            all_mask_data = np.array(all_mask_data)

            all_mask_data = all_mask_data.T

            self._heatmap_clustermap_generator(data=all_mask_data,
                                               x_tick_labels=x_tick_labels,
                                               x_label=analysis_variable,
                                               cmap=cmap,
                                               marker_clusters=self.config.marker_clusters,
                                               output_dir=output_dir,
                                               map_name="Pseudo_Time_Analysis",
                                               cluster=False,
                                               y_tick_labels=y_tick_labels)

            all_mask_data = []
            x_tick_labels = []

            for val in np.linspace(0, 1, 5):

                current_expansion_all = feed_features[(feed_features[analysis_variable] >= val) &
                                                      (feed_features[analysis_variable] < val + 0.25)].groupby(
                    ['Marker'])[
                    'Mean Expression'].mean().to_numpy()

                x_tick_labels.append(round(val, 2))

                if current_expansion_all.size > 0:
                    all_mask_data.append(current_expansion_all)
                else:
                    all_mask_data.append(np.zeros((self.config.n_markers,), np.uint8))

            all_mask_data = np.array(all_mask_data)

            all_mask_data = all_mask_data.T

            self._heatmap_clustermap_generator(data=all_mask_data,
                                               x_tick_labels=x_tick_labels,
                                               x_label=analysis_variable,
                                               cmap=cmap,
                                               marker_clusters=self.config.marker_clusters,
                                               output_dir=output_dir,
                                               map_name="Pseudo_Time_Analysis_Binned",
                                               cluster=False,
                                               y_tick_labels=y_tick_labels)

    def vessel_asymmetry_area_spread_plot(self, **kwargs):
        """
        Vessel Asymmetry Area Spread Plot
        :return:
        """

        assert "Asymmetry" in self.all_samples_features, "Asymmetry analysis has not been performed, please run " \
                                                         "vessel_contiguity_analysis() first!"

        output_dir = "%s/Vessel Areas Spread Boxplot" % self.config.visualization_results_dir
        mkdir_p(output_dir)
        idx = pd.IndexSlice

        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      0,
                                                      "Data"], :]

        asymmetry_features = plot_features.loc[self.all_samples_features["Asymmetry"] != "NA"]

        plot_features['Size'] = pd.cut(plot_features['Contour Area'],
                                       bins=[self.config.small_vessel_threshold,
                                             self.config.medium_vessel_threshold,
                                             self.config.large_vessel_threshold,
                                             float('Inf')],
                                       labels=["Small",
                                               "Medium",
                                               "Large"])

        asymmetry_features['Size'] = pd.cut(asymmetry_features['Contour Area'],
                                            bins=[self.config.small_vessel_threshold,
                                                  self.config.medium_vessel_threshold,
                                                  self.config.large_vessel_threshold,
                                                  float('Inf')],
                                            labels=["Small",
                                                    "Medium",
                                                    "Large"])

        asymmetry_features = asymmetry_features.rename(columns={'Contour Area': 'Pixel Area'})
        plot_features = plot_features.rename(columns={'Contour Area': 'Pixel Area'})

        nobs_split = asymmetry_features.groupby(['Size', 'Asymmetry']).apply(lambda x: 'n: {}'.format(len(x)))
        nobs = plot_features.groupby(['Size']).apply(lambda x: 'n: {}'.format(len(x)))

        ax = sns.boxplot(x="Size",
                         y="Pixel Area",
                         hue="Asymmetry",
                         data=asymmetry_features,
                         showfliers=False)

        for tick, label in enumerate(ax.get_xticklabels()):
            ax_size = label.get_text()

            for j, ax_asymmetry in enumerate(ax.get_legend_handles_labels()[1]):
                x_offset = (j - 0.5) * 2 / 5
                num = nobs_split[ax_size, ax_asymmetry]

                point_data_transform = (tick + x_offset, 0)

                axis_to_data = ax.transAxes + ax.transData.inverted()
                data_to_axis = axis_to_data.inverted()

                point_axis_transform = data_to_axis.transform(point_data_transform)

                ax.text(point_axis_transform[0], 0.005, str(num),
                        horizontalalignment='center', size='x-small', color='k', weight='semibold',
                        transform=ax.transAxes)

        plt.savefig(output_dir + '/vessel_area_spread_asymmetry_split.png',
                    bbox_inches='tight')
        plt.clf()

        ax = sns.boxplot(x="Size",
                         y="Pixel Area",
                         data=plot_features,
                         showfliers=False)

        for tick, label in enumerate(ax.get_xticklabels()):
            ax_size = label.get_text()

            num = nobs[ax_size]

            point_data_transform = (tick, 0)

            axis_to_data = ax.transAxes + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()

            point_axis_transform = data_to_axis.transform(point_data_transform)

            ax.text(point_axis_transform[0], 0.005, str(num),
                    horizontalalignment='center', size='x-small', color='k', weight='semibold',
                    transform=ax.transAxes)

        plt.savefig(output_dir + '/vessel_area_spread.png',
                    bbox_inches='tight')
        plt.clf()

    def plot_vessel_areas(self,
                          all_points_vessel_contours: list,
                          segmentation_type: str = 'allvessels',
                          show_outliers: bool = False) -> list:
        """
        Plot box plot vessel areas

        :param all_points_vessel_contours: array_like, [n_points, n_vessels] -> list of vessel contours for each point
        :param save_csv: bool, Save csv of vessel areas
        :param segmentation_type: str, Segmentation mask type
        :param show_outliers: bool, Include outliers in box plots
        :return: list, [n_points, n_vessels] -> All points vessel areas
        """

        brain_regions = self.config.brain_region_point_ranges
        region_data = []
        current_point = 1
        current_region = 0
        areas = []
        per_point_areas = []
        total_per_point_areas = []

        all_points_vessel_areas = []

        for idx, contours in enumerate(all_points_vessel_contours):
            current_per_point_area = []

            for cnt in contours:
                contour_area = cv.contourArea(cnt)
                areas.append(contour_area)
                current_per_point_area.append(contour_area)

            current_point += 1
            per_point_areas.append(current_per_point_area)
            all_points_vessel_areas.append(current_per_point_area)

            if not (brain_regions[current_region][0] <= current_point <= brain_regions[current_region][1]):
                current_region += 1
                region_data.append(sorted(areas))
                total_per_point_areas.append(per_point_areas)
                areas = []
                per_point_areas = []

        for i, area in enumerate(region_data):
            area = sorted(area)
            plt.hist(area, bins=200)
            plt.title("Points %s to %s" % (str(brain_regions[i][0]), str(brain_regions[i][1])))
            plt.xlabel("Pixel Area")
            plt.ylabel("Count")
            plt.show()

        colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']

        fig = plt.figure(1, figsize=(9, 6))
        plt.title("%s Mask Points 1 to 48" % segmentation_type)

        # Create an axes instance
        ax = fig.add_subplot(111)

        # Create the boxplot
        bp = ax.boxplot(all_points_vessel_areas, showfliers=show_outliers, patch_artist=True)

        for w, region in enumerate(brain_regions):
            patches = bp['boxes'][region[0] - 1:region[1]]

            for patch in patches:
                patch.set(facecolor=colors[w])

        plt.show()

        return all_points_vessel_areas

    def vessel_nonvessel_masks(self,
                               n_expansions: int = 5,
                               **kwargs
                               ):
        """
        Get Vessel nonvessel masks

        :param n_expansions: int, Number of expansions
        """

        img_shape = self.config.segmentation_mask_size

        example_img = np.zeros(img_shape, np.uint8)
        example_img = cv.cvtColor(example_img, cv.COLOR_GRAY2BGR)

        parent_dir = "%s/Associated Area Masks" % self.config.visualization_results_dir

        for feed_idx in self.all_feeds_contour_data.index.get_level_values('Feed Index').unique():

            feed_data = self.all_feeds_contour_data.loc[feed_idx]

            idx = pd.IndexSlice
            feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]

            feed_dir = "%s/%s" % (parent_dir, feed_name)
            mkdir_p(feed_dir)

            output_dir = "%s/%s%s Expansion" % (feed_dir,
                                                str(round_to_nearest_half((n_expansions) *
                                                                          self.config.pixel_interval *
                                                                          self.config.pixels_to_distance)),
                                                self.config.data_resolution_units)
            mkdir_p(output_dir)

            for point_num in range(self.config.n_points):
                per_point_vessel_contours = feed_data.loc[point_num, "Contours"].contours

                regions = get_assigned_regions(per_point_vessel_contours, img_shape)

                for idx, cnt in enumerate(per_point_vessel_contours):
                    mask_expanded = expand_vessel_region(cnt, img_shape,
                                                         upper_bound=self.config.pixel_interval * n_expansions)
                    mask_expanded = cv.bitwise_and(mask_expanded, regions[idx].astype(np.uint8))
                    dark_space_mask = regions[idx].astype(np.uint8) - mask_expanded

                    example_img[np.where(dark_space_mask == 1)] = self.config.nonvessel_mask_colour  # red
                    example_img[np.where(mask_expanded == 1)] = self.config.vessel_space_colour  # green
                    cv.drawContours(example_img, [cnt], -1, self.config.vessel_mask_colour, cv.FILLED)  # blue

                    vesselnonvessel_label = "Point %s" % str(point_num + 1)

                    cv.imwrite(os.path.join(output_dir, "%s.png" % vesselnonvessel_label),
                               example_img)
