import datetime
import math
import random
from collections import Collection
from functools import partial

import cv2
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
from src.data_preprocessing.transforms import melt_markers, loc_by_expansion
from src.utils.iterators import feed_features_iterator
from src.utils.utils_functions import mkdir_p, round_to_nearest_half, save_fig_or_show

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


class Visualizer:

    def __init__(self,
                 config: Config,
                 all_samples_features: pd.DataFrame,
                 markers_names: list,
                 all_feeds_contour_data: pd.DataFrame,
                 all_feeds_metadata: pd.DataFrame,
                 all_points_marker_data: np.array,
                 results_dir: str
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
        self.results_dir = results_dir

    def expansion_line_plots_per_vessel(self, n_expansions: int, **kwargs):
        """
        Create vessel region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """
        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        style = kwargs.get("style", None)
        size = kwargs.get("size", None)
        vessel_line_plots_points = kwargs.get("vessel_line_plots_points", self.config.vessel_line_plots_points)
        save_fig = kwargs.get("save_fig", True)

        output_dir = "%s/Line Plots Per Vessel" % self.results_dir

        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir, str(round_to_nearest_half(n_expansions *
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

        plot_features = melt_markers(plot_features,
                                     non_id_vars=self.markers_names,
                                     reset_index=['Expansion'],
                                     add_marker_group=True,
                                     marker_groups=marker_clusters)

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        for point in vessel_line_plots_points:
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
                                 hue="Marker Group",
                                 style=style,
                                 size=size,
                                 palette=self.config.line_plots_bin_colors,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=average_bins_dir + '/Vessel_ID_%s.png' % str(vessel))

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

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=all_bins_dir + '/Vessel_ID_%s.png' % str(vessel))

                for key in marker_clusters.keys():
                    bin_dir = "%s/%s" % (per_bin_dir, key)
                    mkdir_p(bin_dir)

                    bin_features = vessel_features.loc[vessel_features["Marker Group"] == key]
                    # Per Bin
                    g = sns.lineplot(data=bin_features,
                                     x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                     y="Expression",
                                     hue="Marker",
                                     style=style,
                                     size=size,
                                     palette=perbin_marker_color_dict,
                                     ci=None)

                    box = g.get_position()
                    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=bin_dir + '/Vessel_ID_%s.png' % str(vessel))

    def expansion_line_plots_per_point(self, n_expansions: int, **kwargs):
        """
        Create point region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        style = kwargs.get("style", None)
        size = kwargs.get("size", None)
        save_fig = kwargs.get("save_fig", True)

        n_points = len(self.all_samples_features.index.get_level_values("Point").unique())

        output_dir = "%s/Line Plots Per Point" % self.results_dir

        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir,
                                            str(round_to_nearest_half(n_expansions *
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

        plot_features = melt_markers(plot_features,
                                     non_id_vars=self.markers_names,
                                     reset_index=['Expansion'],
                                     add_marker_group=True,
                                     marker_groups=marker_clusters)

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        for point in range(n_points):
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
                             hue="Marker Group",
                             style=style,
                             palette=self.config.line_plots_bin_colors,
                             ci=None)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=point_dir + '/Average_Bins.png')

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

            save_fig_or_show(save_fig=save_fig,
                             figure_path=point_dir + '/All_Bins.png')

            for key in marker_clusters.keys():
                bin_features = point_features.loc[point_features["Marker Group"] == key]
                # Per Bin
                g = sns.lineplot(data=bin_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker",
                                 style=style,
                                 palette=perbin_marker_color_dict,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=point_dir + '/%s.png' % str(key))

                for marker in marker_clusters[key]:
                    marker_features = plot_features.loc[plot_features["Marker"] == marker]
                    g = sns.lineplot(data=marker_features,
                                     x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                     y="Expression",
                                     hue="Marker",
                                     style=style,
                                     size=size,
                                     palette=perbin_marker_color_dict,
                                     ci=None)

                    box = g.get_position()
                    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=per_marker_dir + '/%s.png' % str(marker))

    def expansion_line_plots_all_points(self, n_expansions: int, **kwargs):
        """
        Create all points average region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        style = kwargs.get("style", None)
        size = kwargs.get("size", None)
        save_fig = kwargs.get("save_fig", True)

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

        plot_features = melt_markers(plot_features,
                                     non_id_vars=self.markers_names,
                                     reset_index=['Expansion'],
                                     add_marker_group=True,
                                     marker_groups=marker_clusters)

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

        output_dir = "%s/Line Plots All Points Average" % self.results_dir

        mkdir_p(output_dir)

        output_dir = "%s/%s%s Expansion" % (output_dir,
                                            str(round_to_nearest_half(n_expansions *
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
                         hue="Marker Group",
                         style=style,
                         palette=self.config.line_plots_bin_colors,
                         ci=None)

        box = g.get_position()
        g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/Average_Bins.png')

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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/All_Bins.png')

        for key in marker_clusters.keys():
            bin_features = plot_features.loc[plot_features["Marker Group"] == key]
            # Per Bin
            g = sns.lineplot(data=bin_features,
                             x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue="Marker",
                             style=style,
                             palette=perbin_marker_color_dict,
                             ci=None)

            box = g.get_position()
            g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=per_bin_dir + '/%s.png' % str(key))

            for marker in marker_clusters[key]:
                marker_features = plot_features.loc[plot_features["Marker"] == marker]
                g = sns.lineplot(data=marker_features,
                                 x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                 y="Expression",
                                 hue="Marker",
                                 style=style,
                                 size=size,
                                 palette=perbin_marker_color_dict,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=per_marker_dir + '/%s.png' % str(marker))

    def expansion_line_plots_per_region(self, n_expansions: int,
                                        **kwargs):
        """
        Create brain region average region line plots for all marker bins, average marker bins and per marker bins

        :param n_expansions: int, Number of expansions
        :return:
        """

        marker_clusters = self.config.marker_clusters
        color_maps = self.config.line_plots_color_maps
        colors = self.config.line_plots_bin_colors

        style = kwargs.get("style", None)
        size = kwargs.get("size", None)
        save_fig = kwargs.get("save_fig", True)

        parent_dir = "%s/Line Plots Per Region" % self.results_dir

        mkdir_p(parent_dir)

        parent_dir = "%s/%s%s Expansion" % (parent_dir,
                                            str(round_to_nearest_half(n_expansions *
                                                                      self.config.pixel_interval *
                                                                      self.config.pixels_to_distance)),
                                            self.config.data_resolution_units)
        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

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
            plot_features = feed_features.loc[idx[:,
                                              :,
                                              :n_expansions,
                                              "Data"], :]

            plot_features = melt_markers(plot_features,
                                         non_id_vars=self.markers_names,
                                         reset_index=['Expansion'],
                                         add_marker_group=True,
                                         marker_groups=marker_clusters)

            plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                          round_to_nearest_half(
                                                                              x * self.config.pixel_interval
                                                                              * self.config.pixels_to_distance))
            plot_features = plot_features.rename(
                columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

            plot_features.reset_index(level=['Point'], inplace=True)

            bins = [brain_region_point_ranges[i][0] - 1 for i in range(len(brain_region_point_ranges))]
            bins.append(float('Inf'))

            plot_features['Region'] = pd.cut(plot_features['Point'],
                                             bins=bins,
                                             labels=brain_region_names)

            for region in self.config.brain_region_names:
                region_dir = "%s/%s" % (feed_dir, region)
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
                                 hue="Marker Group",
                                 style=style,
                                 palette=self.config.line_plots_bin_colors,
                                 ci=None)

                box = g.get_position()
                g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=region_dir + '/Average_Bins.png')

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

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=region_dir + '/All_Bins.png')

                for key in marker_clusters.keys():
                    bin_features = region_features.loc[region_features["Marker Group"] == key]
                    # Per Bin
                    g = sns.lineplot(data=bin_features,
                                     x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                     y="Expression",
                                     hue="Marker",
                                     style=style,
                                     palette=perbin_marker_color_dict,
                                     ci=None)

                    box = g.get_position()
                    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=per_bin_dir + '/%s.png' % str(key))

                    for marker in marker_clusters[key]:
                        marker_features = plot_features.loc[plot_features["Marker"] == marker]
                        g = sns.lineplot(data=marker_features,
                                         x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                         y="Expression",
                                         hue="Marker",
                                         style=style,
                                         size=size,
                                         palette=perbin_marker_color_dict,
                                         ci=None)

                        box = g.get_position()
                        g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
                        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

                        save_fig_or_show(save_fig=save_fig,
                                         figure_path=per_marker_dir + '/%s.png' % str(marker))

    def obtain_expanded_vessel_masks(self, **kwargs):
        """
        Create expanded region vessel masks

        :return:
        """

        expansion_upper_bound = kwargs.get('expansion_upper_bound', 60)

        n_points = len(self.all_samples_features.index.get_level_values("Point").unique())

        parent_dir = "%s/Expanded Vessel Masks" % self.results_dir

        mkdir_p(parent_dir)

        parent_dir = "%s/%s %s" % (parent_dir,
                                   str(round_to_nearest_half(expansion_upper_bound * self.config.pixels_to_distance)),
                                   self.config.data_resolution_units)
        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            distinct_dir = "%s/Original Mask Excluded" % feed_dir
            mkdir_p(distinct_dir)

            nondistinct_dir = "%s/Original Mask Included" % feed_dir
            mkdir_p(nondistinct_dir)

            for idx in range(n_points):
                point_contours = feed_contours.loc[idx, "Contours"].contours

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
        n_points = len(self.all_samples_features.index.get_level_values("Point").unique())

        parent_dir = "%s/Embedded Vessel Masks" % self.results_dir

        mkdir_p(parent_dir)

        parent_dir = "%s/%s %s" % (parent_dir,
                                   str(round_to_nearest_half(expansion_upper_bound * self.config.pixels_to_distance)),
                                   self.config.data_resolution_units)
        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            distinct_dir = "%s/Original Mask Excluded" % feed_dir
            mkdir_p(distinct_dir)

            nondistinct_dir = "%s/Original Mask Included" % feed_dir
            mkdir_p(nondistinct_dir)

            for idx in range(n_points):
                original_not_included_point_mask = np.zeros(self.config.segmentation_mask_size, np.uint8)
                original_included_point_mask = np.zeros(self.config.segmentation_mask_size, np.uint8)

                point_contours = feed_contours.loc[idx, "Contours"].contours

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

        n_expansions = kwargs.get("n_expansions", 10)
        interval = self.config.pixel_interval
        n_points = len(self.all_samples_features.index.get_level_values("Point").unique())
        expansions = kwargs.get("expansions", self.config.expansion_to_run)

        parent_dir = "%s/Ring Plots" % self.results_dir

        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            for point_num in range(n_points):
                current_interval = interval

                expansion_image = np.zeros(self.config.segmentation_mask_size, np.uint8)

                colors = pl.cm.Greys(np.linspace(0, 1, n_expansions + 10))

                for x in range(n_expansions):
                    per_point_vessel_contours = feed_contours.loc[point_num, "Contours"].contours
                    expansion_ring_plots(per_point_vessel_contours,
                                         expansion_image,
                                         pixel_expansion_upper_bound=current_interval,
                                         pixel_expansion_lower_bound=current_interval - interval,
                                         color=colors[x + 5] * 255)

                    if x + 1 in expansions:
                        child_dir = parent_dir + "/%s%s Expansion" % (str(round_to_nearest_half(n_expansions *
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
        save_fig = kwargs.get("save_fig", True)

        output_dir = "%s/Expression Histograms" % self.results_dir

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
        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/SMA.png')

    def biaxial_scatter_plot(self, **kwargs):
        """
        Biaxial Scatter Plot for visualizing Marker Expressions

        :return:
        """
        save_fig = kwargs.get("save_fig", True)

        output_dir = "%s/Biaxial Scatter Plots" % self.results_dir

        mkdir_p(output_dir)

        idx = pd.IndexSlice

        x = "SMA"
        y = "GLUT1"

        visualization_features = loc_by_expansion(self.all_samples_features.copy(),
                                                  columns_to_keep=self.markers_names,
                                                  expansion_type="mask_only",
                                                  average=True)

        x_data = visualization_features[x].values
        y_data = visualization_features[y].values

        positive_sma = len(self.all_samples_features.loc[self.all_samples_features[x] > 0.1].values)
        all_vess = len(self.all_samples_features.values)

        logging.debug("There are %s / %s vessels which are positive for SMA" % (positive_sma, all_vess))

        # Calculate the point density
        xy = np.vstack([x_data, y_data])

        z = gaussian_kde(xy)(xy)

        x_data = np.expand_dims(x_data, axis=0)
        y_data = np.expand_dims(y_data, axis=0)

        plt.scatter(x_data, y_data, c=z, s=35, edgecolor='')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title("%s vs %s" % (x, y))

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/%s_%s.png' % (x, y))

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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/%s_%s.png' % (x, y))

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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/%s_%s.png' % (x, y), )

    def average_quartile_violin_plot_subplots(self, **kwargs):
        """
        Pseduotime Heatmap and Violin Plot subplot
        """
        save_fig = kwargs.get("save_fig", True)
        mask_type = kwargs.get('mask_type', "expansion_only")
        primary_categorical_analysis_variable = kwargs.get('primary_categorical_analysis_variable', "Solidity")
        order = kwargs.get("order", None)
        parent_dir = "%s/Average Quartile Violin Plots" % self.results_dir
        marker_clusters = self.config.marker_clusters
        selected_clusters = ["Vessels", "Astrocytes"]

        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1.0), "black"],
                  [norm(-0.5), "indigo"],
                  [norm(0), "firebrick"],
                  [norm(0.5), "orange"],
                  [norm(1.0), "khaki"]]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            feed_features = loc_by_expansion(feed_features,
                                             expansion_type=mask_type,
                                             average=False)

            feed_features = melt_markers(feed_features,
                                         non_id_vars=self.markers_names,
                                         )

            n_rows = 2
            n_cols = 8
            row = 0
            column = 0
            i = 0

            fig = plt.figure(constrained_layout=True, figsize=(40, 15))

            ax = fig.add_gridspec(n_rows, n_cols)

            for key in selected_clusters:

                for marker, marker_name in enumerate(marker_clusters[key]):
                    marker_features = feed_features[(feed_features["Marker"] == marker_name)]

                    ax2 = fig.add_subplot(ax[row, column])

                    sns.violinplot(x=primary_categorical_analysis_variable,
                                   y="Expression",
                                   order=order,
                                   inner="quartile",
                                   data=marker_features,
                                   bw=0.2,
                                   ax=ax2
                                   )
                    ax2.set_ylim(-0.15, 1.75)
                    ax2.set_title(marker_name)

                    i += 1

                    column += 1

                    if i == n_cols - 1:
                        row += 1
                        column = 0

            save_fig_or_show(save_fig=save_fig,
                             figure_path=feed_dir + '/average_quartile_violins.png',
                             fig=fig)

    def categorical_violin_plot_with_images(self, **kwargs):
        """
        Categorical Violin Plot with Images
        """
        save_fig = kwargs.get("save_fig", True)
        mask_type = kwargs.get('mask_type', "mask_only")
        outward_expansion = kwargs.get("outward_expansion", 0)
        inward_expansion = kwargs.get("inward_expansion", 0)
        mask_size = kwargs.get("mask_size", self.config.segmentation_mask_size)

        primary_categorical_analysis_variable = kwargs.get('primary_categorical_analysis_variable',
                                                           "Solidity")
        order = kwargs.get("order", ["25%", "50%", "75%", "100%"])

        # random.seed(10)  # 568, 570

        parent_dir = "%s/Categorical Violin Plots with Images" % self.results_dir

        mkdir_p(parent_dir)

        marker_clusters = self.config.marker_clusters

        cmap = matplotlib.cm.get_cmap('viridis')

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            feed_features = loc_by_expansion(feed_features,
                                             expansion_type=mask_type,
                                             average=False)

            feed_features = melt_markers(feed_features,
                                         non_id_vars=self.markers_names,
                                         )

            quartile_25 = np.percentile(feed_features['Contour Area'], 25)
            quartile_50 = np.percentile(feed_features['Contour Area'], 50)
            quartile_75 = np.percentile(feed_features['Contour Area'], 75)

            feed_features['Size'] = pd.cut(feed_features['Contour Area'],
                                           bins=[0,
                                                 quartile_25,
                                                 quartile_50,
                                                 quartile_75,
                                                 float('Inf')],
                                           labels=["25%",
                                                   "50%",
                                                   "75%",
                                                   "100%"])

            split_dir = "%s/By %s" % (feed_dir, primary_categorical_analysis_variable)
            mkdir_p(split_dir)

            vessel_images = {}
            for size in feed_features['Size'].unique():

                vessel_images[size] = {}

                for val in feed_features[primary_categorical_analysis_variable].unique():

                    split_features = feed_features[(feed_features[primary_categorical_analysis_variable] == val) &
                                                   (feed_features['Size'] == size)]

                    vessel_images[size][val] = []

                    if len(split_features.index) == 0:
                        continue

                    sample_index = random.sample(list(split_features.index), 1)[0]

                    point_idx = sample_index[0]
                    cnt_idx = sample_index[1]

                    cnt = feed_contours.loc[point_idx - 1, "Contours"].contours[cnt_idx]

                    if mask_type == "mask_and_expansion":
                        mask = expand_vessel_region(cnt,
                                                    mask_size,
                                                    upper_bound=outward_expansion,
                                                    lower_bound=0)
                    elif mask_type == "expansion_only":
                        original_mask = np.zeros(mask_size, np.uint8)
                        cv.drawContours(original_mask, [cnt], -1, (1, 1, 1), cv.FILLED)

                        mask_expanded = expand_vessel_region(cnt,
                                                             mask_size,
                                                             upper_bound=outward_expansion,
                                                             lower_bound=0)
                        mask = mask_expanded - original_mask
                    elif mask_type == "mask_only":
                        mask = np.zeros(mask_size, np.uint8)
                        cv.drawContours(mask, [cnt], -1, (1, 1, 1), cv.FILLED)

                    marker_data = self.all_feeds_data[feed_idx, point_idx - 1]

                    marker_dict = dict(zip(self.markers_names, marker_data))

                    for marker in self.markers_names:
                        marker_dict[marker] = gaussian_filter(marker_dict[marker], sigma=4)

                        normed_data = (marker_dict[marker] - np.min(marker_dict[marker])) / (
                                np.max(marker_dict[marker]) - np.min(marker_dict[marker]))

                        mapped_data = cmap(normed_data)

                        result = (cv.bitwise_and(mapped_data, mapped_data, mask=mask) * 255).astype("uint8")
                        result = cv.cvtColor(result, cv.COLOR_RGBA2RGB)

                        x, y, w, h = cv.boundingRect(cnt)
                        marker_dict[marker] = result[
                                              max(0, y - outward_expansion):y + h + outward_expansion,
                                              max(0, x - outward_expansion):x + w + outward_expansion]

                    vessel_images[size][val].append({
                        "Index": (point_idx, cnt_idx),
                        "Image": marker_dict
                    })

            for key in marker_clusters.keys():
                marker_cluster_dir = "%s/%s" % (split_dir, key)
                mkdir_p(marker_cluster_dir)

                for marker, marker_name in enumerate(marker_clusters[key]):
                    marker_features = feed_features[(feed_features["Marker"] == marker_name)]

                    fig = plt.figure(constrained_layout=True, figsize=(40, 15))

                    ax = fig.add_gridspec(4, 9)

                    row = 0

                    for size_key in order:
                        column = 0

                        for key_val in order:

                            for img_pair in vessel_images[size_key][key_val]:
                                img = img_pair["Image"][marker_name]

                                ax1 = fig.add_subplot(ax[row, column])
                                color_map = ax1.imshow(img, origin='lower')

                                divider = make_axes_locatable(ax1)
                                cax = divider.append_axes('right', size='5%', pad=0.05)
                                fig.colorbar(color_map, cax=cax, orientation='vertical')

                                ax1.set_title(key_val + ", " + size_key)

                            column += 1

                        row += 1

                    violin_ax = fig.add_subplot(ax[0:, 5:])

                    sns.violinplot(x="Size",
                                   y="Expression",
                                   hue=primary_categorical_analysis_variable,
                                   hue_order=order,
                                   inner="quartile",
                                   data=marker_features,
                                   bw=0.2,
                                   ax=violin_ax
                                   )
                    violin_ax.set_ylim(-0.15, 1.75)

                    violin_ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                                     title=primary_categorical_analysis_variable)

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=marker_cluster_dir + '/%s.png' % str(marker_name),
                                     fig=fig)

    def categorical_violin_plot(self, **kwargs):
        """
        Categorical Violin Plot
        """

        mask_type = kwargs.get('mask_type', "expansion_only")
        primary_categorical_analysis_variable = kwargs.get('primary_categorical_analysis_variable', "Solidity")
        order = kwargs.get("order", None)
        save_fig = kwargs.get("save_fig", True)

        parent_dir = "%s/Categorical Violin Plots" % self.results_dir

        mkdir_p(parent_dir)

        marker_clusters = self.config.marker_clusters

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            feed_features = loc_by_expansion(feed_features,
                                             expansion_type=mask_type,
                                             average=False)

            feed_features = melt_markers(feed_features,
                                         non_id_vars=self.markers_names,
                                         )

            feed_features['Size'] = pd.cut(feed_features['Contour Area'],
                                           bins=[self.config.small_vessel_threshold,
                                                 self.config.medium_vessel_threshold,
                                                 self.config.large_vessel_threshold,
                                                 float('Inf')],
                                           labels=["Small",
                                                   "Medium",
                                                   "Large"])

            split_dir = "%s/By %s" % (feed_dir, primary_categorical_analysis_variable)
            mkdir_p(split_dir)

            for key in marker_clusters.keys():
                marker_cluster_dir = "%s/%s" % (split_dir, key)
                mkdir_p(marker_cluster_dir)

                for marker, marker_name in enumerate(marker_clusters[key]):
                    marker_features = feed_features[(feed_features["Marker"] == marker_name)]

                    plt.figure(figsize=(22, 10))

                    ax = sns.violinplot(x="Size",
                                        y="Expression",
                                        hue=primary_categorical_analysis_variable,
                                        hue_order=order,
                                        inner="box",
                                        data=marker_features,
                                        bw=0.2,
                                        # palette=cmap
                                        )

                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                               title=primary_categorical_analysis_variable)

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=marker_cluster_dir + '/%s.png' % str(marker_name))

            non_split_dir = "%s/Without Split" % feed_dir
            mkdir_p(non_split_dir)

            for key in marker_clusters.keys():
                marker_cluster_dir = "%s/%s" % (non_split_dir, key)
                mkdir_p(marker_cluster_dir)

                for marker, marker_name in enumerate(marker_clusters[key]):
                    marker_features = feed_features[(feed_features["Marker"] == marker_name)]

                    plt.figure(figsize=(10, 10))

                    ax = sns.violinplot(x=primary_categorical_analysis_variable,
                                        y="Expression",
                                        order=order,
                                        inner="box",
                                        data=marker_features,
                                        bw=0.2,
                                        # palette=cmap
                                        )

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=marker_cluster_dir + '/%s.png' % str(marker_name))

    def violin_plot_brain_expansion(self, n_expansions: int, **kwargs):
        """
        Violin Plots for Expansion Data

        :param n_expansions: int, Number of expansions

        :return:
        """

        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Vessel Size")
        save_fig = kwargs.get("save_fig", True)

        dist_upper_end = 1.75

        output_dir = "%s/Expansion Violin Plots" % self.results_dir

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

        plot_features = melt_markers(plot_features,
                                     non_id_vars=self.markers_names,
                                     reset_index=['Expansion', 'Point'],
                                     add_marker_group=True,
                                     marker_groups=marker_clusters)

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

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
                                    hue=primary_categorical_analysis_variable,
                                    palette=colors_clusters,
                                    inner=None,
                                    data=marker_features,
                                    bw=0.2)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=per_marker_expansions_dir + '/%s.png' % str(marker_name))

        for key in marker_clusters.keys():
            marker_features = plot_features.loc[plot_features["Marker Group"] == key]

            colors_clusters = color_maps[key](np.linspace(0, 1, 6))[3:]

            plt.figure(figsize=(22, 10))

            if max_expression < dist_upper_end:
                plt.ylim(-0.15, dist_upper_end)
            else:
                plt.ylim(-0.15, max_expression)

            ax = sns.violinplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                                y="Expression",
                                hue=primary_categorical_analysis_variable,
                                palette=colors_clusters,
                                inner=None,
                                data=marker_features,
                                bw=0.2)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=per_bin_expansions_dir + '/%s.png' % str(key))

    def box_plot_brain_expansions(self, n_expansions: int, **kwargs):
        """
        Box Plots for Expansion Data

        :param n_expansions: int, Number of expansions

        :return:
        """

        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Vessel Size")
        save_fig = kwargs.get("save_fig", True)

        dist_upper_end = 1.75

        output_dir = "%s/Expansion Box Plots" % self.results_dir

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

        plot_features = melt_markers(plot_features,
                                     non_id_vars=self.markers_names,
                                     reset_index=['Expansion', 'Point'],
                                     add_marker_group=True,
                                     marker_groups=marker_clusters)

        plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                      round_to_nearest_half(
                                                                          x * self.config.pixel_interval
                                                                          * self.config.pixels_to_distance))
        plot_features = plot_features.rename(
            columns={'Expansion': "Distance Expanded (%s)" % self.config.data_resolution_units})

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
                                 hue=primary_categorical_analysis_variable,
                                 palette=colors_clusters,
                                 data=marker_features)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=per_marker_expansions_dir + '/%s.png' % str(marker_name))

        for key in marker_clusters.keys():
            marker_features = plot_features.loc[plot_features["Marker Group"] == key]

            colors_clusters = color_maps[key](np.linspace(0, 1, 6))[3:]

            plt.figure(figsize=(22, 10))

            if max_expression < dist_upper_end:
                plt.ylim(-0.15, dist_upper_end)
            else:
                plt.ylim(-0.15, max_expression)

            ax = sns.boxplot(x="Distance Expanded (%s)" % self.config.data_resolution_units,
                             y="Expression",
                             hue=primary_categorical_analysis_variable,
                             palette=colors_clusters,
                             data=marker_features)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=per_bin_expansions_dir + '/%s.png' % str(key))

    def categorical_spatial_probability_maps(self, **kwargs):
        """
        Spatial Probability Maps

        :return:
        """

        mask_size = kwargs.get("mask_size", self.config.segmentation_mask_size)
        save_fig = kwargs.get("save_fig", True)
        n_samples = kwargs.get("n_samples", 3)

        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Solidity")

        parent_dir = "%s/Pixel Expression Spatial Maps" % self.results_dir
        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            n_points = len(feed_features.index.get_level_values("Point").unique())

            all_markers_dir = "%s/All Markers" % feed_dir
            mkdir_p(all_markers_dir)

            for point_idx in range(n_points):

                marker_data = self.all_feeds_data[feed_idx, point_idx]

                marker_dict = dict(zip(self.markers_names, marker_data))

                point_dir = "%s/Point %s" % (all_markers_dir, str(point_idx + 1))
                mkdir_p(point_dir)

                for marker in self.markers_names:
                    marker_dir = "%s/%s" % (point_dir, marker)
                    mkdir_p(marker_dir)

                    data = marker_dict[marker]
                    blurred_data = gaussian_filter(data, sigma=4)

                    for category in feed_features[primary_categorical_analysis_variable].unique():
                        category_dir = "%s/%s" % (marker_dir, category)
                        mkdir_p(category_dir)

                        point_features = feed_features.loc[pd.IndexSlice[point_idx + 1, :, :, "Data"]]
                        contour_indices = point_features[point_features[primary_categorical_analysis_variable] == category].index.get_level_values("Vessel").unique().tolist()

                        point_contours = feed_contours.loc[point_idx, "Contours"].contours
                        point_contours = [point_contours[i] for i in contour_indices]

                        if len(point_contours) > n_samples:
                            point_contours = point_contours[:n_samples]

                        for contour_idx, c in enumerate(point_contours):
                            mask = np.zeros(mask_size, np.uint8)
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

                            save_fig_or_show(save_fig=save_fig,
                                             figure_path=os.path.join(category_dir, "Vessel_ID_%s.png" % str(contour_idx + 1)))

    def spatial_probability_maps(self, **kwargs):
        """
        Spatial Probability Maps

        :return:
        """

        mask_size = kwargs.get("mask_size", self.config.segmentation_mask_size)
        save_fig = kwargs.get("save_fig", True)

        parent_dir = "%s/Pixel Expression Spatial Maps" % self.results_dir

        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            n_points = len(feed_features.index.get_level_values("Point").unique())

            vessels_dir = "%s/Vessels" % feed_dir
            mkdir_p(vessels_dir)

            astrocytes_dir = "%s/Astrocytes" % feed_dir
            mkdir_p(astrocytes_dir)

            all_markers_dir = "%s/All Markers" % feed_dir
            mkdir_p(all_markers_dir)

            for point_idx in range(n_points):

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
                save_fig_or_show(save_fig=save_fig,
                                 figure_path="%s/Point%s" % (vessels_dir, str(point_idx + 1)))

                point_contours = feed_contours.loc[point_idx, "Contours"].contours

                point_dir = "%s/Point %s" % (vessels_dir, str(point_idx + 1))
                mkdir_p(point_dir)

                for contour_idx, c in enumerate(point_contours):
                    mask = np.zeros(mask_size, np.uint8)
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

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=os.path.join(point_dir, "Vessel_ID_%s.png" % str(contour_idx + 1)))

            for point_idx in range(n_points):

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

                save_fig_or_show(save_fig=save_fig,
                                 figure_path="%s/Point%s" % (astrocytes_dir, str(point_idx + 1)))

                point_dir = "%s/Point %s" % (astrocytes_dir, str(point_idx + 1))
                mkdir_p(point_dir)

                for contour_idx, c in enumerate(point_contours):
                    mask = np.zeros(mask_size, np.uint8)
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

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=os.path.join(point_dir, "Vessel_ID_%s.png" % str(contour_idx + 1)))

            for point_idx in range(n_points):

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

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path="%s/%s" % (point_dir, marker_name))

    def vessel_images_by_categorical_variable(self, **kwargs):
        """
        Vessel Images by Categorical Variable
        :return:
        """
        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Solidity")
        n_examples = kwargs.get("n_examples", 10)
        random.seed(42)

        assert primary_categorical_analysis_variable is not None, "There must be a primary categorical splitter"

        parent_dir = "%s/%s Vessel Images" % (self.results_dir,
                                              primary_categorical_analysis_variable)

        img_shape = self.config.segmentation_mask_size

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            for val in feed_features[primary_categorical_analysis_variable].unique():

                output_dir = "%s/%s" % (feed_dir, val)
                mkdir_p(output_dir)

                split_features = feed_features[feed_features[primary_categorical_analysis_variable] == val]

                for i in random.sample(list(split_features.index), min(n_examples, len(list(split_features.index)))):
                    point_idx = i[0]
                    cnt_idx = i[1]

                    cnt = feed_contours.loc[point_idx - 1, "Contours"].contours[cnt_idx]

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
                                max_val=None,
                                save_fig=True):
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
        columns = data.columns

        assert x in columns and y in columns and hue in columns, "Analysis columns are not in dataframe!"

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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/%s.png' % figure_name)

    def scatter_plot_umap_marker_projection(self, **kwargs):
        """
        UMAP projection with marker channel colorbar

        :return:
        """

        mask_type = kwargs.get('mask_type', "mask_only")
        primary_continuous_analysis_variable = kwargs.get('primary_continuous_analysis_variable', "Solidity Score")
        save_fig = kwargs.get("save_fig", True)

        parent_dir = self.results_dir + "/UMAP Scatter Plot Projection"

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            plot_features = loc_by_expansion(feed_features,
                                             expansion_type=mask_type,
                                             average=False)

            for marker_cluster in self.config.marker_clusters.keys():
                plot_features[marker_cluster] = \
                    plot_features.loc[pd.IndexSlice[:,
                                      :,
                                      :,
                                      :], self.config.marker_clusters[marker_cluster]].mean(axis=1)

            for marker_cluster in self.config.marker_clusters.keys():

                self._scatter_plot_color_bar(marker_cluster,
                                             feed_dir + "/Marker Clusters",
                                             plot_features,
                                             hue=marker_cluster,
                                             min_val=0,
                                             max_val=1,
                                             save_fig=save_fig)

                for marker in self.config.marker_clusters[marker_cluster]:
                    self._scatter_plot_color_bar(marker,
                                                 feed_dir + "/Individual Markers",
                                                 plot_features,
                                                 hue=marker,
                                                 save_fig=save_fig)

            self._scatter_plot_color_bar("umap_projection_by_size",
                                         feed_dir + "/Size",
                                         plot_features,
                                         hue="Contour Area",
                                         min_val=self.config.small_vessel_threshold,
                                         max_val=1000,
                                         save_fig=save_fig)

            self._scatter_plot_color_bar("umap_projection",
                                         feed_dir + "/%s" % primary_continuous_analysis_variable,
                                         plot_features,
                                         hue=primary_continuous_analysis_variable,
                                         min_val=plot_features[primary_continuous_analysis_variable].min(),
                                         max_val=1.25,
                                         save_fig=save_fig)

            plot_features.reset_index(level=['Point'], inplace=True)

            bins = [brain_region_point_ranges[i][0] - 1 for i in range(len(brain_region_point_ranges))]
            bins.append(float('Inf'))

            plot_features['Region'] = pd.cut(plot_features['Point'],
                                             bins=bins,
                                             labels=brain_region_names)

            region_dir = feed_dir + "/Region"
            mkdir_p(region_dir)

            g = sns.scatterplot(data=plot_features,
                                x="UMAP0",
                                y="UMAP1",
                                hue="Region",
                                ci=None,
                                palette="tab20")

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                       borderaxespad=0.)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=region_dir + '/umap_projection_by_region.png')

    def _average_expression_across_expansions(self,
                                              n_expansions: int,
                                              mibi_features: pd.DataFrame,
                                              marker_names: list,
                                              region: list = None,
                                              data_type: str = "Data"):
        """
        Helper function to calculate the average marker expression across several expansions
        """
        idx = pd.IndexSlice

        try:

            if region is None:
                expression_matrix = mibi_features.loc[idx[:, :,
                                                      :n_expansions,
                                                      data_type], marker_names].to_numpy()
            else:
                expression_matrix = mibi_features.loc[idx[region[0]:region[1],
                                                      :,
                                                      :n_expansions,
                                                      data_type], marker_names].to_numpy()

            average_expression = np.mean(expression_matrix, axis=0)
        except KeyError:
            average_expression = np.zeros((len(marker_names),), np.uint8)

        return average_expression

    def _vessel_non_vessel_heatmap_block(self,
                                         n_expansions: int,
                                         mibi_features: pd.DataFrame,
                                         marker_names: list,
                                         brain_regions: list,
                                         heatmap_data_list: list,
                                         data_type: str = "Data"):
        """
        Computes the data for the vessel-non-vessel heatmap for a n+1 row block, n = # of regions
        """

        average_expression_all_points = self._average_expression_across_expansions(n_expansions,
                                                                                   mibi_features,
                                                                                   marker_names,
                                                                                   data_type=data_type)

        heatmap_data_list.append(average_expression_all_points)

        for region in brain_regions:
            average_expression_per_region = self._average_expression_across_expansions(n_expansions,
                                                                                       mibi_features,
                                                                                       marker_names,
                                                                                       region=region,
                                                                                       data_type=data_type)

            heatmap_data_list.append(average_expression_per_region)

    def _vessel_non_vessel_heatmap_block(self,
                                         n_expansions: int,
                                         mibi_features: pd.DataFrame,
                                         marker_names: list,
                                         brain_regions: list,
                                         heatmap_data_list: list,
                                         data_type: str = "Data"):
        """
        Computes the data for the vessel-non-vessel heatmap for a n+1 row block, n = # of regions
        """

        average_expression_all_points = self._average_expression_across_expansions(n_expansions,
                                                                                   mibi_features,
                                                                                   marker_names,
                                                                                   data_type=data_type)

        heatmap_data_list.append(average_expression_all_points)

        for region in brain_regions:
            average_expression_per_region = self._average_expression_across_expansions(n_expansions,
                                                                                       mibi_features,
                                                                                       marker_names,
                                                                                       region=region,
                                                                                       data_type=data_type)

            heatmap_data_list.append(average_expression_per_region)

    def _vessel_nonvessel_heatmap(self,
                                  n_expansions: int,
                                  mibi_features: pd.DataFrame,
                                  brain_regions: list,
                                  marker_clusters: list,
                                  feed_dir: str,
                                  save_fig: bool = True):
        """
        Vessel/Non-vessel heatmaps helper method

        :param n_expansions:
        :param feed_features:
        :return:
        """

        # Vessel Space (SMA Positive)
        positve_sma = mibi_features.loc[
            mibi_features["SMA"] >= self.config.SMA_positive_threshold]

        # Vessel Space (SMA Negative)
        negative_sma = mibi_features.loc[
            mibi_features["SMA"] < self.config.SMA_positive_threshold]

        heatmap_data = []

        self._vessel_non_vessel_heatmap_block(n_expansions,
                                              positve_sma,
                                              self.markers_names,
                                              brain_regions,
                                              heatmap_data,
                                              data_type="Data")

        self._vessel_non_vessel_heatmap_block(n_expansions,
                                              negative_sma,
                                              self.markers_names,
                                              brain_regions,
                                              heatmap_data,
                                              data_type="Data")

        # Non-vessel Space

        self._vessel_non_vessel_heatmap_block(n_expansions,
                                              positve_sma,
                                              self.markers_names,
                                              brain_regions,
                                              heatmap_data,
                                              data_type="Non-Vascular Space")

        self._vessel_non_vessel_heatmap_block(n_expansions,
                                              negative_sma,
                                              self.markers_names,
                                              brain_regions,
                                              heatmap_data,
                                              data_type="Non-Vascular Space")

        # Vessel environment space

        self._vessel_non_vessel_heatmap_block(n_expansions,
                                              positve_sma,
                                              self.markers_names,
                                              brain_regions,
                                              heatmap_data,
                                              data_type="Vascular Space")

        self._vessel_non_vessel_heatmap_block(n_expansions,
                                              negative_sma,
                                              self.markers_names,
                                              brain_regions,
                                              heatmap_data,
                                              data_type="Vascular Space")

        yticklabels = ["Vascular Space (SMA+) - All Points",
                       "Vascular Space (SMA+) - MFG",
                       "Vascular Space (SMA+) - HIP",
                       "Vascular Space (SMA+) - CAUD",
                       "Vascular Space (SMA-) - All Points",
                       "Vascular Space (SMA-) - MFG",
                       "Vascular Space (SMA-) - HIP",
                       "Vascular Space (SMA-) - CAUD",
                       "Non-Vascular Space (SMA+) - All Points",
                       "Non-Vascular Space (SMA-) - All Points",
                       "Non-Vascular Space (SMA+) - MFG",
                       "Non-Vascular Space (SMA-) - MFG",
                       "Non-Vascular Space (SMA+) - HIP",
                       "Non-Vascular Space (SMA-) - HIP",
                       "Non-Vascular Space (SMA+) - CAUD",
                       "Non-Vascular Space (SMA-) - CAUD",
                       "Vascular Expansion Space (SMA+) - All Points",
                       "Vascular Expansion Space (SMA+) - MFG",
                       "Vascular Expansion Space (SMA+) - HIP",
                       "Vascular Expansion Space (SMA+) - CAUD",
                       "Vascular Expansion Space (SMA-) - All Points",
                       "Vascular Expansion Space (SMA-) - MFG",
                       "Vascular Expansion Space (SMA-) - HIP",
                       "Vascular Expansion Space (SMA-) - CAUD",
                       ]

        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1.0), "black"],
                  [norm(-0.5), "indigo"],
                  [norm(0), "firebrick"],
                  [norm(0.5), "orange"],
                  [norm(1.0), "khaki"]]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        plt.figure(figsize=(22, 10))

        ax = sns.heatmap(heatmap_data,
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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/Expansion_%s.png' % str(n_expansions))

        ax = sns.clustermap(heatmap_data,
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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/Expansion_%s.png' % str(n_expansions))

    def continuous_scatter_plot(self, **kwargs):
        """
        Plot marker expressions as a function of some continuous variable

        :return:
        """

        primary_categorical_analysis_variable = kwargs.get('primary_categorical_analysis_variable', "Solidity")
        secondary_categorical_analysis_variable = kwargs.get('secondary_categorical_analysis_variable', "Vessel Size")
        save_fig = kwargs.get("save_fig", True)

        mask_type = kwargs.get('mask_type', "expansion_only")
        primary_continuous_analysis_variable = kwargs.get('primary_continuous_analysis_variable', "Solidity Score")

        assert primary_categorical_analysis_variable is not None, "Must have a primary categorical variable"
        assert secondary_categorical_analysis_variable is not None, "Must have a secondary categorical variable"

        parent_dir = "%s/%s Scatter Plots" % (self.results_dir
                                              , primary_continuous_analysis_variable)
        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            by_primary_analysis_variable_dir = "%s/By %s" % (feed_dir, primary_categorical_analysis_variable)
            mkdir_p(by_primary_analysis_variable_dir)

            by_secondary_analysis_variable_dir = "%s/By %s" % (feed_dir, secondary_categorical_analysis_variable)
            mkdir_p(by_secondary_analysis_variable_dir)

            with_vessel_id_dir = "%s/%s" % (feed_dir, "With Vessel ID")
            mkdir_p(with_vessel_id_dir)

            secondary_separate_dir = "%s/Separate By %s" % (feed_dir, secondary_categorical_analysis_variable)
            mkdir_p(secondary_separate_dir)

            feed_features = loc_by_expansion(feed_features,
                                             expansion_type=mask_type,
                                             average=False)

            feed_features = melt_markers(feed_features,
                                         non_id_vars=self.markers_names,
                                         add_marker_group=False)

            feed_features['Mean Expression'] = \
                feed_features.groupby(['Vessel', 'Point', 'Marker'])['Expression'].transform('mean')

            feed_features.reset_index(level=['Vessel', 'Point', 'Expansion'], inplace=True)

            feed_features = feed_features[feed_features["Expansion"] == feed_features["Expansion"].max()]

            if len(feed_features[primary_categorical_analysis_variable].unique()) <= 2:
                cmap = colors.ListedColormap(['blue', 'red'])(np.linspace(0, 1, 2))
            else:
                cmap = matplotlib.cm.get_cmap('Set1')(np.linspace(0,
                                                                  1,
                                                                  len(feed_features[
                                                                          primary_categorical_analysis_variable].unique())))

            for marker in feed_features["Marker"].unique():
                marker_features = feed_features[feed_features["Marker"] == marker]

                g = sns.scatterplot(data=marker_features,
                                    x=primary_continuous_analysis_variable,
                                    y="Mean Expression",
                                    hue=primary_categorical_analysis_variable,
                                    ci=None,
                                    palette=cmap)

                plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                           borderaxespad=0.,
                           title=primary_categorical_analysis_variable)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=by_primary_analysis_variable_dir + '/%s.png' % str(marker))

            if len(feed_features[primary_categorical_analysis_variable].unique()) <= 3:
                cmap = colors.ListedColormap(['cyan', 'pink', 'yellow'])(np.linspace(0, 1, 3))
            else:
                cmap = matplotlib.cm.get_cmap('Set1')(
                    np.linspace(0,
                                1,
                                len(feed_features[
                                        primary_categorical_analysis_variable].unique())))

            for marker in feed_features["Marker"].unique():
                marker_features = feed_features[feed_features["Marker"] == marker]

                g = sns.scatterplot(data=marker_features,
                                    x=primary_continuous_analysis_variable,
                                    y="Mean Expression",
                                    hue=secondary_categorical_analysis_variable,
                                    ci=None,
                                    palette=cmap,
                                    edgecolor='k',
                                    linewidth=1)

                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                           title=secondary_categorical_analysis_variable)

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=by_secondary_analysis_variable_dir + '/%s.png' % str(marker))

            for marker in feed_features["Marker"].unique():
                marker_features = feed_features[feed_features["Marker"] == marker]

                g = sns.scatterplot(data=marker_features,
                                    x=primary_continuous_analysis_variable,
                                    y="Mean Expression",
                                    ci=None,
                                    palette="tab20")

                for line in range(0, min(50, len(marker_features["Point"].values))):
                    point_label = str(marker_features["Point"].values[line]) + ":" \
                                  + str(marker_features["Vessel"].values[line])

                    if not math.isnan(marker_features[primary_continuous_analysis_variable].values[line]) \
                            and not math.isnan(marker_features["Mean Expression"].values[line]):
                        g.text(marker_features[primary_continuous_analysis_variable].values[line],
                               marker_features["Mean Expression"].values[line],
                               point_label, horizontalalignment='left',
                               size='medium', color='black', weight='semibold')

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=with_vessel_id_dir + '/%s.png' % str(marker))

            for split_val in feed_features[secondary_categorical_analysis_variable].unique():
                out_dir = "%s/%s" % (secondary_separate_dir, split_val)
                mkdir_p(out_dir)

                split_features = feed_features[feed_features[secondary_categorical_analysis_variable] == split_val]

                for marker in split_features["Marker"].unique():
                    marker_features = split_features[split_features["Marker"] == marker]

                    plt.xlim([-0.05, 1.05])

                    g = sns.scatterplot(data=marker_features,
                                        x=primary_continuous_analysis_variable,
                                        y="Mean Expression",
                                        hue=primary_categorical_analysis_variable,
                                        ci=None,
                                        palette="tab20")

                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                               title=primary_categorical_analysis_variable)

                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=out_dir + '/%s.png' % str(marker))

    def vessel_nonvessel_heatmap(self, n_expansions: int, **kwargs):
        """
        Vessel/Non-vessel heatmaps for marker expression

        :param n_expansions: int, Number of expansions
        :return:
        """
        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Vessel Size")
        save_fig = kwargs.get("save_fig", True)

        marker_clusters = self.config.marker_clusters

        parent_dir = "%s/Heatmaps & Clustermaps" % self.results_dir

        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            if primary_categorical_analysis_variable is None:
                self._vessel_nonvessel_heatmap(n_expansions, feed_features, brain_region_point_ranges, marker_clusters,
                                               feed_dir,
                                               save_fig=save_fig)
            else:
                for i in feed_features[primary_categorical_analysis_variable].unique():
                    split_dir = "%s/%s: %s" % (feed_dir, primary_categorical_analysis_variable, i)
                    mkdir_p(split_dir)

                    split_features = feed_features.loc[feed_features[primary_categorical_analysis_variable] == i]

                    self._vessel_nonvessel_heatmap(n_expansions, split_features, brain_region_point_ranges,
                                                   marker_clusters,
                                                   split_dir,
                                                   save_fig=save_fig)

    def _categorical_split_expansion_heatmap_helper(self,
                                                    n_expansions: int,
                                                    expansion_features: pd.DataFrame,
                                                    heatmaps_dir: str,
                                                    primary_splitter: str,
                                                    secondary_splitter: str = None,
                                                    marker: str = "Astrocytes",
                                                    cluster=True,
                                                    save_fig=True
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

        for split_val in expansion_features[primary_splitter].unique():

            if split_val in classes_to_ignore:
                continue

            y_lab_split = "%s : %s" % (primary_splitter, split_val)
            y_tick_labels.append(y_lab_split)

            split_features = expansion_features.loc[expansion_features[
                                                        primary_splitter]
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
                    enumerate(expansion_features[secondary_splitter].unique()):
                secondary_split_features = expansion_features.loc[expansion_features[
                                                                      secondary_splitter]
                                                                  == secondary_split_val]

                for split_val in expansion_features[primary_splitter].unique():

                    if split_val in classes_to_ignore:
                        continue

                    y_lab_split_first_level = "%s : %s" % (primary_splitter, split_val)

                    split_features = secondary_split_features.loc[expansion_features[
                                                                      primary_splitter]
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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/%s.png' % title)

    def categorical_split_expansion_heatmap_clustermap(self,
                                                       n_expansions: int,
                                                       **kwargs):
        """
        Categorically split expansion heatmap

        :param n_expansions: int, Number of expansions to run
        :return:
        """
        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Vessel Size")
        secondary_categorical_analysis_variable = kwargs.get("secondary_categorical_analysis_variable", None)
        save_fig = kwargs.get("save_fig", True)

        parent_dir = "%s/Categorical Expansion Heatmaps & Clustermaps" % self.results_dir

        mkdir_p(parent_dir)

        assert primary_categorical_analysis_variable is not None, "No categorical splitter selected!"

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            expansion_features = feed_features.loc[pd.IndexSlice[:, :, :n_expansions, :], :]

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
                                                                 primary_categorical_analysis_variable,
                                                                 secondary_categorical_analysis_variable,
                                                                 marker=cluster,
                                                                 cluster=True,
                                                                 save_fig=save_fig)

                for marker in self.config.marker_clusters[cluster]:
                    self._categorical_split_expansion_heatmap_helper(n_expansions,
                                                                     expansion_features,
                                                                     markers_dir,
                                                                     primary_categorical_analysis_variable,
                                                                     secondary_categorical_analysis_variable,
                                                                     marker=marker,
                                                                     cluster=False,
                                                                     save_fig=save_fig)

    def marker_covariance_heatmap(self, **kwargs):
        """
        Marker Covariance Heatmap
        """

        degree = kwargs.get("deg", 4)
        save_fig = kwargs.get("save_fig", True)

        marker_clusters = self.config.marker_clusters

        parent_dir = "%s/Covariance Heatmap" % self.results_dir
        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            idx = pd.IndexSlice
            plot_features = self.all_samples_features.loc[idx[:,
                                                          :,
                                                          :,
                                                          "Data"], :]

            plot_features = melt_markers(plot_features,
                                         non_id_vars=self.markers_names,
                                         reset_index=['Expansion'],
                                         add_marker_group=True,
                                         marker_groups=marker_clusters)

            plot_features['Expansion'] = plot_features['Expansion'].apply(lambda x:
                                                                          round_to_nearest_half(
                                                                              x * self.config.pixel_interval
                                                                              * self.config.pixels_to_distance))

            plot_features = plot_features[["Marker", "Expansion", "Expression"]]
            plot_features = plot_features.groupby(["Marker", "Expansion"]).mean()
            plot_features.reset_index(level=['Marker'], inplace=True)

            marker_poly = {}
            n_markers = len(self.markers_names)
            n_expansions = len(plot_features.index.unique("Expansion"))

            heatmap_data = np.zeros((n_markers, n_expansions))

            for marker_name in self.markers_names:
                y = plot_features[plot_features["Marker"] == marker_name]["Expression"].values
                x = np.arange(len(y))

                p = np.poly1d(np.polyfit(x, y, degree))
                marker_poly[marker_name] = np.polyder(p)

            for m, marker_name in enumerate(self.markers_names):
                for n, expansion in enumerate(plot_features.index.unique("Expansion")):
                    p = marker_poly[marker_name]
                    slope = p(n)

                    heatmap_data[m, n] = slope

            norm = matplotlib.colors.Normalize(-1, 1)
            colors = [[norm(-1.0), "black"],
                      [norm(-0.5), "indigo"],
                      [norm(0), "firebrick"],
                      [norm(0.5), "orange"],
                      [norm(1.0), "khaki"]]

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

            plt.figure(figsize=(22, 10))

            xticklabels = plot_features.index.unique("Expansion")

            ax = sns.heatmap(heatmap_data,
                             cmap=cmap,
                             xticklabels=xticklabels,
                             yticklabels=self.markers_names,
                             linewidths=0,
                             )

            save_fig_or_show(save_fig=save_fig,
                             figure_path=feed_dir + '/Expansion_%s.png' % str(n_expansions))

    def _heatmap_clustermap_generator(self,
                                      data,
                                      x_tick_labels,
                                      x_label,
                                      cmap,
                                      marker_clusters,
                                      output_dir,
                                      map_name,
                                      cluster=False,
                                      y_tick_labels=False,
                                      x_tick_values=None,
                                      x_tick_indices=None,
                                      vmin=None,
                                      vmax=None,
                                      ax=None,
                                      save_fig=True):

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

            if ax is None:
                plt.figure(figsize=(22, 10))

                ax = sns.heatmap(data,
                                 cmap=cmap,
                                 xticklabels=x_tick_labels,
                                 yticklabels=y_tick_labels,
                                 linewidths=0,
                                 vmin=vmin,
                                 vmax=vmax
                                 )

                ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

                if axis_ticklabels_overlap(ax.get_xticklabels()):
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                if len(x_tick_labels) != data.shape[1] or x_tick_values is not None or x_tick_indices is not None:
                    xticks = ax.get_xticklabels()

                    for xtick in xticks:
                        xtick.set_visible(False)

                    if x_tick_values is not None:
                        for i, val in enumerate(x_tick_labels):
                            if val in x_tick_values:
                                xticks[i].set_visible(True)
                                x_tick_values.remove(val)

                    if x_tick_indices is not None:
                        for i in x_tick_indices:
                            xticks[i].set_visible(True)

                plt.xlabel("%s" % x_label)

                h_line_idx = 0

                for key in marker_clusters.keys():
                    if h_line_idx != 0:
                        ax.axhline(h_line_idx, 0, len(self.markers_names), linewidth=3, c='w')

                    for _ in marker_clusters[key]:
                        h_line_idx += 1

                save_fig_or_show(save_fig=save_fig,
                                 figure_path=output_dir + '/%s.png' % map_name)
            else:
                ax = sns.heatmap(data,
                                 cmap=cmap,
                                 xticklabels=x_tick_labels,
                                 yticklabels=y_tick_labels,
                                 linewidths=0,
                                 vmin=vmin,
                                 vmax=vmax,
                                 ax=ax
                                 )

                ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")

                if axis_ticklabels_overlap(ax.get_xticklabels()):
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                if len(x_tick_labels) != data.shape[1] or x_tick_values is not None or x_tick_indices is not None:
                    xticks = ax.get_xticklabels()

                    for xtick in xticks:
                        xtick.set_visible(False)

                    if x_tick_values is not None:
                        for i, val in enumerate(x_tick_labels):
                            if val in x_tick_values:
                                xticks[i].set_visible(True)
                                x_tick_values.remove(val)

                    if x_tick_indices is not None:
                        for i in x_tick_indices:
                            xticks[i].set_visible(True)

                plt.xlabel("%s" % x_label)

                h_line_idx = 0

                for key in marker_clusters.keys():
                    if h_line_idx != 0:
                        ax.axhline(h_line_idx, 0, len(self.markers_names), linewidth=3, c='w')

                    for _ in marker_clusters[key]:
                        h_line_idx += 1

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

            if len(x_tick_labels) != data.shape[1] or x_tick_values is not None or x_tick_indices is not None:
                xticks = ax_ax.get_xticklabels()

                for xtick in xticks:
                    xtick.set_visible(False)

                if x_tick_values is not None:
                    for i, val in enumerate(x_tick_labels):
                        if val in x_tick_values:
                            xticks[i].set_visible(True)
                            x_tick_values.remove(val)

                if x_tick_indices is not None:
                    for i in x_tick_indices:
                        xticks[i].set_visible(True)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=output_dir + '/%s.png' % map_name)

    def _marker_expression_at_expansion(self,
                                        expansion: int,
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

    def _brain_region_expansion_heatmap(self,
                                        n_expansions: int,
                                        mibi_features: pd.DataFrame,
                                        heatmaps_dir: str,
                                        clustermaps_dir: str,
                                        brain_regions: list,
                                        brain_region_names: list,
                                        save_fig: bool = True
                                        ):
        """
        Brain region expansion heatmaps helper method

        :param n_expansions: int, Number of expansions to run
        :return:
        """
        regions_mask_data = []

        marker_names = self.markers_names
        pixel_interval = self.config.pixel_interval
        marker_clusters = self.config.marker_clusters

        marker_mask_expression_map_fn_all_regions = partial(self._marker_expression_at_expansion,
                                                            mibi_features=mibi_features,
                                                            marker_names=marker_names,
                                                            data_type="Data")

        all_regions_mask_data = np.array(list(map(marker_mask_expression_map_fn_all_regions,
                                                  sorted(mibi_features.index.unique("Expansion").tolist()))))

        all_nonmask_data = mibi_features.loc[pd.IndexSlice[:, :,
                                             max(mibi_features.index.unique("Expansion").tolist()),
                                             "Non-Vascular Space"], marker_names].to_numpy()

        mean_nonmask_data = np.mean(all_nonmask_data, axis=0)

        all_mask_data = np.append(all_regions_mask_data, [mean_nonmask_data], axis=0)

        all_mask_data = np.transpose(all_mask_data)

        for region in brain_regions:
            region_marker_mask_expression_map_fn = partial(self._marker_expression_at_expansion,
                                                           mibi_features=mibi_features,
                                                           marker_names=marker_names,
                                                           region=region,
                                                           data_type="Data")

            region_mask_data = np.array(list(map(region_marker_mask_expression_map_fn,
                                                 sorted(mibi_features.index.unique("Expansion").tolist()))))

            region_non_mask_data = mibi_features.loc[pd.IndexSlice[region[0]:region[1],
                                                     :,
                                                     max(mibi_features.index.unique("Expansion").tolist()),
                                                     "Non-Vascular Space"], marker_names].to_numpy()

            region_mean_non_mask_data = np.mean(region_non_mask_data, axis=0)

            region_mask_data = np.append(region_mask_data, [region_mean_non_mask_data], axis=0)

            region_mask_data = np.transpose(region_mask_data)

            regions_mask_data.append(region_mask_data)

        x_tick_labels = np.array(sorted(mibi_features.index.unique("Expansion").tolist())) * pixel_interval
        x_tick_labels = x_tick_labels.tolist()
        x_tick_labels = [str(round_to_nearest_half(x)) for x in x_tick_labels]
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
                                           cluster=False,
                                           save_fig=save_fig)

        for idx, region_data in enumerate(regions_mask_data):
            region_name = brain_region_names[idx]

            self._heatmap_clustermap_generator(data=region_data,
                                               x_tick_labels=x_tick_labels,
                                               x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                               cmap=cmap,
                                               marker_clusters=marker_clusters,
                                               output_dir=output_dir,
                                               map_name="%s_Region" % region_name,
                                               cluster=False,
                                               save_fig=save_fig)

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
                                           cluster=True,
                                           save_fig=save_fig)

        for idx, region_data in enumerate(regions_mask_data):
            region_name = brain_region_names[idx]

            self._heatmap_clustermap_generator(data=region_data,
                                               x_tick_labels=x_tick_labels,
                                               x_label="Distance Expanded (%s)" % self.config.data_resolution_units,
                                               cmap=cmap,
                                               marker_clusters=marker_clusters,
                                               output_dir=output_dir,
                                               map_name="%s_Region" % region_name,
                                               cluster=True,
                                               save_fig=save_fig)

    def brain_region_expansion_heatmap(self, n_expansions: int, **kwargs):
        """
        Brain Region Expansion Heatmap

        :param n_expansions: int, Number of expansions
        """
        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Vessel Size")
        save_fig = kwargs.get("save_fig", True)

        parent_dir = "%s/Expansion Heatmaps & Clustermaps" % self.results_dir

        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            expansion_features = feed_features.loc[pd.IndexSlice[:, :, :n_expansions, :], :]

            if primary_categorical_analysis_variable is None:
                heatmaps_dir = "%s/Expansion Heatmaps" % feed_dir
                clustermaps_dir = "%s/Expansion Clustermaps" % feed_dir

                mkdir_p(heatmaps_dir)
                mkdir_p(clustermaps_dir)

                self._brain_region_expansion_heatmap(n_expansions, expansion_features, heatmaps_dir, clustermaps_dir,
                                                     brain_region_point_ranges, brain_region_names, save_fig=save_fig)
            else:
                for i in feed_features[primary_categorical_analysis_variable].unique():
                    split_dir = "%s/%s: %s" % (feed_dir, primary_categorical_analysis_variable, i)
                    mkdir_p(split_dir)

                    heatmaps_dir = "%s/Expansion Heatmaps" % split_dir
                    clustermaps_dir = "%s/Expansion Clustermaps" % split_dir

                    mkdir_p(heatmaps_dir)
                    mkdir_p(clustermaps_dir)

                    split_features = feed_features.loc[feed_features[primary_categorical_analysis_variable] == i]

                    self._brain_region_expansion_heatmap(n_expansions, split_features, heatmaps_dir, clustermaps_dir,
                                                         brain_region_point_ranges, brain_region_names,
                                                         save_fig=save_fig)

    def marker_expression_masks(self, **kwargs):
        """
        Marker Expression Overlay Masks

        :return:
        """
        save_fig = kwargs.get("save_fig", True)

        n_points = len(self.all_samples_features.index.get_level_values("Point").unique())

        parent_dir = "%s/Expression Masks" % self.results_dir

        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            for i in range(n_points):
                point_contours = feed_contours.loc[i, "Contours"]

                point_dir = feed_dir + "/Point %s" % str(i + 1)
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
                    save_fig_or_show(save_fig=save_fig,
                                     figure_path=os.path.join(point_dir, "%s.png" % marker_name))

    def _pseudo_time_heatmap(self,
                             mibi_features,
                             ax=None,
                             save_dir=None,
                             binned=False,
                             cmap=None,
                             map_name="Example",
                             **kwargs):
        """
        Pseudo Time Heatmap
        """

        mask_type = kwargs.get("mask_type", "mask_only")
        primary_continuous_analysis_variable = kwargs.get("primary_continuous_analysis_variable", "Eccentricity Score")
        save_fig = kwargs.get("save_fig", True)

        mibi_features = loc_by_expansion(mibi_features,
                                         expansion_type=mask_type,
                                         average=False)

        mibi_features = melt_markers(mibi_features,
                                     non_id_vars=self.markers_names,
                                     add_marker_group=False)

        mibi_features["Expression"] = pd.to_numeric(mibi_features["Expression"])

        mibi_features['Mean Expression'] = \
            mibi_features.groupby(['Vessel', 'Point', 'Marker'])['Expression'].transform('mean')

        mibi_features.reset_index(level=['Vessel', 'Point', 'Expansion'], inplace=True)

        mibi_features = mibi_features[mibi_features["Expansion"] == mibi_features["Expansion"].max()]

        all_mask_data = []
        x_tick_labels = []

        for val in sorted(mibi_features[primary_continuous_analysis_variable].unique()):
            y_tick_labels = \
                mibi_features[mibi_features[primary_continuous_analysis_variable] == val].groupby(['Marker'],
                                                                                                  sort=False)[
                    'Mean Expression'].mean().reset_index()["Marker"].values

        if binned:
            for val in np.linspace(mibi_features[primary_continuous_analysis_variable].min(),
                                   mibi_features[primary_continuous_analysis_variable].max(), 5):

                current_expansion_all = mibi_features[(mibi_features[primary_continuous_analysis_variable] >= val) &
                                                      (mibi_features[primary_continuous_analysis_variable] < val +
                                                       mibi_features[
                                                           primary_continuous_analysis_variable].max() / 5)].groupby(
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
                                               x_label=primary_continuous_analysis_variable,
                                               x_tick_indices=np.linspace(0,
                                                                          all_mask_data.shape[1] - 1,
                                                                          5,
                                                                          dtype=int),
                                               cmap=cmap,
                                               marker_clusters=self.config.marker_clusters,
                                               output_dir=save_dir,
                                               map_name="%s_pseudo_time_heatmap_binned_%s" % (
                                                   primary_continuous_analysis_variable,
                                                   map_name),
                                               cluster=False,
                                               y_tick_labels=y_tick_labels,
                                               vmin=0,
                                               vmax=1.25,
                                               save_fig=save_fig
                                               )
        else:

            for val in sorted(mibi_features[primary_continuous_analysis_variable].unique()):

                current_expansion_all = \
                    mibi_features[mibi_features[primary_continuous_analysis_variable] == val].groupby(['Marker'],
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
                                               x_label=primary_continuous_analysis_variable,
                                               x_tick_indices=np.linspace(0,
                                                                          all_mask_data.shape[1] - 1,
                                                                          5,
                                                                          dtype=int),
                                               cmap=cmap,
                                               marker_clusters=self.config.marker_clusters,
                                               output_dir=save_dir,
                                               map_name="%s_pseudo_time_heatmap_%s" % (
                                                   primary_continuous_analysis_variable,
                                                   map_name),
                                               cluster=False,
                                               y_tick_labels=y_tick_labels,
                                               ax=ax,
                                               vmin=0,
                                               vmax=1.25,
                                               save_fig=save_fig
                                               )

        return ax

    def pseudo_time_heatmap(self,
                            cmap=None,
                            **kwargs):
        """
        Pseudo-Time Heatmap

        :param cmap: Matplotlib.Colormap, Colormap to be used for plot
        :param kwargs: Keyword arguments
        :return:
        """

        parent_dir = "%s/Pseudo-Time Heatmaps" % self.results_dir
        save_fig = kwargs.get("save_fig", True)

        mkdir_p(parent_dir)

        if cmap is None:
            norm = matplotlib.colors.Normalize(-1, 1)
            colors = [[norm(-1.0), "black"],
                      [norm(-0.5), "indigo"],
                      [norm(0), "firebrick"],
                      [norm(0.5), "orange"],
                      [norm(1.0), "khaki"]]

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        primary_continuous_analysis_variable = kwargs.get('primary_continuous_analysis_variable', "Solidity Score")
        mask_type = kwargs.get('mask_type', "expansion_only")

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            feed_features.reset_index(level=['Point', 'Vessel', 'Expansion', 'Expansion Type'], inplace=True)

            bins = [brain_region_point_ranges[i][0] - 1 for i in range(len(brain_region_point_ranges))]
            bins.append(float('Inf'))

            feed_features['Region'] = pd.cut(feed_features['Point'],
                                             bins=bins,
                                             labels=brain_region_names)

            feed_features.set_index(['Point', 'Vessel', 'Expansion', 'Expansion Type'], inplace=True)

            for region in feed_features['Region'].unique():
                region_dir = "%s/%s" % (feed_dir, region)
                mkdir_p(region_dir)

                region_features = feed_features[feed_features['Region'] == region]

                self._pseudo_time_heatmap(region_features,
                                          save_dir=region_dir,
                                          primary_continuous_analysis_variable=primary_continuous_analysis_variable,
                                          mask_type=mask_type,
                                          map_name=region,
                                          cmap=cmap,
                                          save_fig=save_fig)

                self._pseudo_time_heatmap(region_features,
                                          save_dir=region_dir,
                                          binned=True,
                                          primary_continuous_analysis_variable=primary_continuous_analysis_variable,
                                          mask_type=mask_type,
                                          map_name=region,
                                          cmap=cmap,
                                          save_fig=save_fig)

            self._pseudo_time_heatmap(feed_features,
                                      save_dir=feed_dir,
                                      binned=False,
                                      primary_continuous_analysis_variable=primary_continuous_analysis_variable,
                                      mask_type=mask_type,
                                      map_name="all_points",
                                      cmap=cmap,
                                      save_fig=save_fig)

    def vessel_shape_area_spread_plot(self, **kwargs):
        """
        Vessel Shape Area Spread Plot
        :return:
        """

        output_dir = "%s/Vessel Areas Spread Boxplot" % self.results_dir

        mkdir_p(output_dir)
        idx = pd.IndexSlice

        primary_categorical_analysis_variable = kwargs.get("primary_categorical_analysis_variable", "Solidity")
        save_fig = kwargs.get("save_fig", True)

        plot_features = self.all_samples_features.loc[idx[:,
                                                      :,
                                                      0,
                                                      "Data"], :]

        shape_quantification_features = plot_features.loc[
            self.all_samples_features[primary_categorical_analysis_variable] != "NA"]

        plot_features['Size'] = pd.cut(plot_features['Contour Area'],
                                       bins=[self.config.small_vessel_threshold,
                                             self.config.medium_vessel_threshold,
                                             self.config.large_vessel_threshold,
                                             float('Inf')],
                                       labels=["Small",
                                               "Medium",
                                               "Large"])

        shape_quantification_features['Size'] = pd.cut(shape_quantification_features['Contour Area'],
                                                       bins=[self.config.small_vessel_threshold,
                                                             self.config.medium_vessel_threshold,
                                                             self.config.large_vessel_threshold,
                                                             float('Inf')],
                                                       labels=["Small",
                                                               "Medium",
                                                               "Large"])

        shape_quantification_features = shape_quantification_features.rename(columns={'Contour Area': 'Pixel Area'})
        plot_features = plot_features.rename(columns={'Contour Area': 'Pixel Area'})

        nobs_split = shape_quantification_features.groupby(['Size', primary_categorical_analysis_variable]).apply(
            lambda x: 'n: {}'.format(len(x)))
        nobs = plot_features.groupby(['Size']).apply(lambda x: 'n: {}'.format(len(x)))

        ax = sns.boxplot(x="Size",
                         y="Pixel Area",
                         hue=primary_categorical_analysis_variable,
                         data=shape_quantification_features,
                         showfliers=False)

        for tick, label in enumerate(ax.get_xticklabels()):
            ax_size = label.get_text()

            for j, ax_shape_quantification in enumerate(ax.get_legend_handles_labels()[1]):
                x_offset = (j - 0.5) * 2 / 5
                num = nobs_split[ax_size, ax_shape_quantification]

                point_data_transform = (tick + x_offset, 0)

                axis_to_data = ax.transAxes + ax.transData.inverted()
                data_to_axis = axis_to_data.inverted()

                point_axis_transform = data_to_axis.transform(point_data_transform)

                ax.text(point_axis_transform[0], 0.005, str(num),
                        horizontalalignment='center', size='x-small', color='k', weight='semibold',
                        transform=ax.transAxes)

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/vessel_area_spread_%s_split.png'
                                     % primary_categorical_analysis_variable)

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

        save_fig_or_show(save_fig=save_fig,
                         figure_path=output_dir + '/vessel_area_spread.png')

    def vessel_nonvessel_masks(self,
                               n_expansions: int = 5,
                               **kwargs
                               ):
        """
        Get Vessel nonvessel masks

        :param n_expansions: int, Number of expansions
        """

        mask_size = kwargs.get("mask_size", self.config.segmentation_mask_size)

        example_img = np.zeros(mask_size, np.uint8)
        example_img = cv.cvtColor(example_img, cv.COLOR_GRAY2BGR)

        parent_dir = "%s/Associated Area Masks" % self.results_dir

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            n_points = len(feed_features.index.get_level_values("Point").unique())

            output_dir = "%s/%s%s Expansion" % (feed_dir,
                                                str(round_to_nearest_half(n_expansions *
                                                                          self.config.pixel_interval *
                                                                          self.config.pixels_to_distance)),
                                                self.config.data_resolution_units)
            mkdir_p(output_dir)

            for point_num in range(n_points):
                per_point_vessel_contours = feed_contours.loc[point_num, "Contours"].contours

                regions = get_assigned_regions(per_point_vessel_contours, mask_size)

                for idx, cnt in enumerate(per_point_vessel_contours):
                    mask_expanded = expand_vessel_region(cnt, mask_size,
                                                         upper_bound=self.config.pixel_interval * n_expansions)
                    mask_expanded = cv.bitwise_and(mask_expanded, regions[idx].astype(np.uint8))
                    dark_space_mask = regions[idx].astype(np.uint8) - mask_expanded

                    example_img[np.where(dark_space_mask == 1)] = (0, 0, 255)  # red
                    example_img[np.where(mask_expanded == 1)] = (0, 255, 0)  # green
                    cv.drawContours(example_img, [cnt], -1, (255, 0, 0), cv.FILLED)  # blue

                    vesselnonvessel_label = "Point %s" % str(point_num + 1)

                    cv.imwrite(os.path.join(output_dir, "%s.png" % vesselnonvessel_label),
                               example_img)

    def removed_vessel_expression_boxplot(self, **kwargs):
        """
        Create kept vs. removed vessel expression comparison using Box Plots
        """
        save_fig = kwargs.get("save_fig", True)

        all_points_vessels_expression = []
        all_points_removed_vessels_expression = []

        parent_dir = "%s/Kept Vs. Removed Vessel Boxplots" % self.results_dir

        mkdir_p(parent_dir)

        for feed_idx, feed_contours, feed_features, feed_dir, brain_region_point_ranges, brain_region_names in feed_features_iterator(
                self.all_samples_features,
                self.all_feeds_data,
                self.all_feeds_contour_data,
                self.all_feeds_metadata,
                save_to_dir=True,
                parent_dir=parent_dir):

            n_points = len(feed_features.index.get_level_values("Point").unique())

            # Iterate through each point
            for i in range(n_points):
                contours = feed_contours.loc[i, "Contours"].contours
                contour_areas = feed_contours.loc[i, "Contours"].areas

                removed_contours = feed_contours.loc[i, "Contours"].removed_contours
                removed_areas = feed_contours.loc[i, "Contours"].removed_areas
                marker_data = self.all_feeds_data[feed_idx, i]

                start_expression = datetime.datetime.now()

                vessel_expression_data = calculate_composition_marker_expression(self.config,
                                                                                 marker_data,
                                                                                 contours,
                                                                                 contour_areas,
                                                                                 self.markers_names,
                                                                                 point_num=i + 1)

                if len(removed_contours) == 0:
                    continue

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

            if len(all_points_removed_vessels_expression) == 0:
                return

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

            markers_to_show = self.config.marker_clusters["Vessels"]

            all_points_per_brain_region_dir = "%s/All Points Per Region" % feed_dir
            mkdir_p(all_points_per_brain_region_dir)

            average_points_dir = "%s/Average Per Region" % feed_dir
            mkdir_p(average_points_dir)

            all_points = "%s/All Points" % feed_dir
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
                save_fig_or_show(save_fig=save_fig,
                                 figure_path=os.path.join(all_points_per_brain_region_dir, "%s.png" % brain_region))

                plt.title("Kept vs Removed Vessel Marker Expression - %s: Average Points" % brain_region)
                ax = sns.boxplot(x="Vessel", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
                save_fig_or_show(save_fig=save_fig,
                                 figure_path=os.path.join(average_points_dir, "%s.png" % brain_region))

            df = pd.DataFrame(all_kept_removed_vessel_expression_data_collapsed,
                              columns=["Expression", "Vessel", "Point"])

            plt.title("Kept vs Removed Vessel Marker Expression - All Points")
            ax = sns.boxplot(x="Vessel", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)

            save_fig_or_show(save_fig=save_fig,
                             figure_path=os.path.join(all_points, "All_Points.png"))
