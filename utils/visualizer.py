import datetime
import random
from collections import Counter
import os

import cv2 as cv
import matplotlib

from utils.mibi_reader import read, get_all_point_data
from image_segmentation.extract_cell_events import *
from feature_extraction.markers_feature_gen import *
from utils.utils_functions import mkdir_p


def vessel_region_plots(n_expansions, interval, markers_names, expansion_data):
    """
    Vessel region line plots

    :param n_expansions:
    :param interval:
    :param markers_names:
    :param expansion_data:
    :return:
    """
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

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    points = [1, 7, 26, 30, 43, 48]
    interval = abs(interval)

    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    # Change in Marker Expression w.r.t pixel expansion per vessel (All bins)
    for point in points:
        output_dir = "results/mean_per_vessel_per_point_per_brain_region/point_%s_vessels_%s_interval_" \
                     "%s_expansions_allbins" % (
                         str(point), str(interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        n_vessels = len(expansion_data[0][point - 1])

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():
                plt.plot([], [], color=colors[key], label=key)

            for marker, marker_name in enumerate(markers_names):
                y = []
                color = None

                for key in marker_clusters.keys():
                    if marker_name in marker_clusters[key]:
                        color = colors[key]
                        break

                for i in range(n_expansions):
                    vessel_data = expansion_data[i][point - 1][vessel]
                    y.append(vessel_data[marker])

                plt.plot(pixel_expansions, y, color=color)
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Vessel ID: %s, Point %s All Bins" % (str(vessel), str(point)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/vessel_%s_allbins.png' % str(vessel), bbox_inches='tight')
            plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per vessel (Average bins)
    for point in points:
        output_dir = "results/mean_per_vessel_per_point_per_brain_region/point_%s_vessels_%s_interval_" \
                     "%s_expansions_averagebins" % (
                         str(point), str(interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        n_vessels = len(expansion_data[0][point - 1])

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():
                plt.plot([], [], color=colors[key], label=key)

            for key in marker_clusters.keys():
                color = colors[key]
                y_tot = []
                for marker, marker_name in enumerate(markers_names):
                    if marker_name not in marker_clusters[key]:
                        continue

                    y = []

                    for i in range(n_expansions):
                        vessel_data = expansion_data[i][point - 1][vessel]
                        y.append(vessel_data[marker])

                    y_tot.append(y)

                plt.plot(pixel_expansions, np.mean(np.array(y_tot), axis=0), color=color)
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Vessel ID: %s, Point %s Average Bins" % (str(vessel), str(point)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/vessel_%s_averagebins.png' % str(vessel), bbox_inches='tight')
            plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per vessel (One plot per bin)
    for point in points:
        output_dir = "results/mean_per_vessel_per_point_per_brain_region/point_%s_vessels_%s_interval_" \
                     "%s_expansions_perbin" % (
                         str(point), str(interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        n_vessels = len(expansion_data[0][point - 1])

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():

                for marker, marker_name in enumerate(markers_names):
                    if marker_name not in marker_clusters[key]:
                        continue

                    y = []

                    for i in range(n_expansions):
                        vessel_data = expansion_data[i][point - 1][vessel]
                        y.append(vessel_data[marker])

                    plt.plot(pixel_expansions, y, label=marker_name)
                plt.xticks(pixel_expansions, fontsize=7)
                plt.xlabel("# of Pixels Expanded")
                plt.ylabel("Mean Pixel Expression")
                plt.title("Vessel ID: %s, Point %s %s" % (str(vessel), str(point), key))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(output_dir + '/vessel_%s_%s.png' % (str(vessel), key), bbox_inches='tight')
                plt.clf()


def point_region_plots(n_expansions, interval, markers_names, expansion_data):
    """
    Point region line plots

    :param n_expansions:
    :param interval:
    :param markers_names:
    :param expansion_data:
    :return:
    """
    n_points = 48
    interval = abs(interval)

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

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    # Points Plots

    output_dir = "results/mean_per_point_per_brain_region/points_%s_interval_%s_expansions" % (
        str(interval), str(n_expansions - 1))
    mkdir_p(output_dir)

    # Change in Marker Expression w.r.t pixel expansion per point (All bins)
    for point in range(n_points):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for marker, marker_name in enumerate(markers_names):
            y = []
            color = None

            for key in marker_clusters.keys():
                if marker_name in marker_clusters[key]:
                    color = colors[key]
                    break

            for i in range(n_expansions):
                current = []
                point_data = expansion_data[i][point]

                for vessel in point_data:
                    current.append(vessel[marker])

                average_expression = sum(current) / len(current)
                y.append(average_expression)
            plt.plot(pixel_expansions, y, color=color)
        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Point %s" % str(point + 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/point_%s_allbins.png' % str(point + 1), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per point (Average bins)
    for point in range(n_points):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for key in marker_clusters.keys():
            color = colors[key]
            y_tot = []
            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    point_data = expansion_data[i][point]

                    for vessel in point_data:
                        current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)
                y_tot.append(y)

            plt.plot(pixel_expansions, np.mean(np.array(y_tot), axis=0), color=color)
        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Point %s" % str(point + 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/point_%s_averagebins.png' % str(point + 1), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per point (One bin per plot)
    for point in range(n_points):

        for key in marker_clusters.keys():

            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    point_data = expansion_data[i][point]

                    for vessel in point_data:
                        current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)

                plt.plot(pixel_expansions, y, label=marker_name)
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Point %s" % str(point + 1))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/point_%s_%s.png' % (str(point + 1), str(key)), bbox_inches='tight')
            plt.clf()


def brain_region_plots(n_expansions, interval, markers_names, expansion_data):
    """
    Brain region expansion line plots
    :param n_expansions:
    :param interval:
    :param markers_names:
    :param expansion_data:
    :return:
    """
    brain_regions = [(1, 16), (17, 32), (33, 48)]
    interval = abs(interval)

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

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    region_names = [
        "MFG",
        "HIP",
        "CAUD"
    ]

    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    output_dir = "results/mean_per_brain_region/brain_regions_%s_interval_%s_expansions" % (
        str(interval), str(n_expansions - 1))
    mkdir_p(output_dir)

    # Brain Region Plots

    # Change in Marker Expression w.r.t pixel expansion per brain region (All bins in one graph)
    for idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for marker, marker_name in enumerate(markers_names):
            y = []
            color = None
            for key in marker_clusters.keys():
                if marker_name in marker_clusters[key]:
                    color = colors[key]
                    break

            for i in range(n_expansions):
                current = []

                for point_data in expansion_data[i][region[0] - 1: region[1] - 1]:
                    for vessel in point_data:
                        current.append(vessel[marker])

                average_expression = sum(current) / len(current)
                y.append(average_expression)

            plt.plot(pixel_expansions, y, color=color)

        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s - All Bins" % str(region_names[idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/region_%s_allbins.png' % str(region_names[idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (Bins averaged into one line)
    for idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for key in marker_clusters.keys():
            color = colors[key]
            y_tot = []
            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    for point_data in expansion_data[i][region[0] - 1: region[1] - 1]:
                        for vessel in point_data:
                            current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)
                y_tot.append(y)

            plt.plot(pixel_expansions, np.mean(np.array(y_tot), axis=0), color=color)
        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s - Average Bins" % str(region_names[idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir + '/region_%s_averagebins.png' % str(region_names[idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (One bin per plot)
    for idx, region in enumerate(brain_regions):
        x = np.arange(n_expansions)

        for key in marker_clusters.keys():
            for marker, marker_name in enumerate(markers_names):
                if marker_name not in marker_clusters[key]:
                    continue

                y = []

                for i in range(n_expansions):
                    current = []
                    for point_data in expansion_data[i][region[0] - 1: region[1] - 1]:
                        for vessel in point_data:
                            current.append(vessel[marker])

                    average_expression = sum(current) / len(current)
                    y.append(average_expression)

                plt.plot(pixel_expansions, y, label=marker_name)
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Brain Region - %s - %s" % (str(region_names[idx]), str(key)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(output_dir + '/region_%s_%s.png' % (str(region_names[idx]), str(key)), bbox_inches='tight')
            plt.clf()


def pixel_expansion_ring_plots(n_expansions=11, interval=-5, mask="allvessels", point_num=33):
    """
    Pixel expansion "Ring" plots

    :param n_expansions: # of expansions
    :param interval: Pixel expansion
    :param mask: Mask type ex. allvessels, largevessels etc.
    :param point_num: Point number to visualize
    :return:
    """

    # Convert point number to index from 0
    point_num -= 1

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data(segmentation_type=mask)

    contour_data_multiple_points = []
    contour_images_multiple_points = []
    expansions = [2, 4, 6, 8, 10]

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    current_interval = interval
    expansion_image = np.zeros(markers_data[0][0].shape, np.uint8)

    for x in range(n_expansions):

        contours = contour_data_multiple_points[point_num]
        expansion_ring_plots(contours,
                             expansion_image,
                             pixel_expansion_amount=current_interval,
                             prev_pixel_expansion_amount=current_interval - interval)
        print("Current interval %s, previous interval %s" % (str(current_interval), str(current_interval - interval)))

        if x + 1 in expansions:
            cv.imwrite("results/ring_plots/expansion_plot_%s_interval_%s_expansion.png" % (str(interval), str(x + 1)),
                       expansion_image)

        current_interval += interval


def mask_nonmask_heatmap(mask_data, nonmask_data, markers_names, n_expansions, interval):
    pixel_expansions = np.array(range(0, n_expansions)) * interval
    pixel_expansions = pixel_expansions.tolist()

    brain_regions = [(1, 16), (17, 32), (33, 48)]

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

    colors = {
        "Nucleus": "b",
        "Microglia": "g",
        "Disease": "r",
        "Vessels": "c",
        "Astrocytes": "m",
        "Synapse": "y",
        "Oligodendrocytes": "k",
        "Neurons": "#ffbb33"
    }

    region_names = [
        "MFG",
        "HIP",
        "CAUD"
    ]

    all_mask_data = []
    mfg_mask_data = []
    hip_mask_data = []
    caud_mask_data = []

    for expansion in mask_data[0:n_expansions]:
        for point_idx, point in enumerate(expansion):
            for vessel in point:
                all_mask_data.append(vessel)

                if brain_regions[0][0] <= point_idx <= brain_regions[0][1]:
                    mfg_mask_data.append(vessel)
                elif brain_regions[1][0] <= point_idx <= brain_regions[1][1]:
                    hip_mask_data.append(vessel)
                elif brain_regions[2][0] <= point_idx <= brain_regions[2][1]:
                    caud_mask_data.append(vessel)

    all_mask_data = np.array(all_mask_data)
    mean_mask_data = np.mean(all_mask_data, axis=0)

    mfg_mask_data = np.array(mfg_mask_data)
    mfg_mask_data = np.mean(mfg_mask_data, axis=0)

    hip_mask_data = np.array(hip_mask_data)
    hip_mask_data = np.mean(hip_mask_data, axis=0)

    caud_mask_data = np.array(caud_mask_data)
    caud_mask_data = np.mean(caud_mask_data, axis=0)

    all_nonmask_data = []
    mfg_nonmask_data = []
    hip_nonmask_data = []
    caud_nonmask_data = []

    for point_idx, point in enumerate(nonmask_data[n_expansions - 1]):
        for vessel in point:
            all_nonmask_data.append(vessel)

            if brain_regions[0][0] <= point_idx <= brain_regions[0][1]:
                mfg_nonmask_data.append(vessel)
            elif brain_regions[1][0] <= point_idx <= brain_regions[1][1]:
                hip_nonmask_data.append(vessel)
            elif brain_regions[2][0] <= point_idx <= brain_regions[2][1]:
                caud_nonmask_data.append(vessel)

    all_nonmask_data = np.array(all_nonmask_data)
    mean_nonmask_data = np.mean(all_nonmask_data, axis=0)

    mfg_nonmask_data = np.array(mfg_nonmask_data)
    mfg_nonmask_data = np.mean(mfg_nonmask_data, axis=0)

    hip_nonmask_data = np.array(hip_nonmask_data)
    hip_nonmask_data = np.mean(hip_nonmask_data, axis=0)

    caud_nonmask_data = np.array(caud_nonmask_data)
    caud_nonmask_data = np.mean(caud_nonmask_data, axis=0)

    all_data = [mean_mask_data,
                mean_nonmask_data,
                mfg_mask_data,
                mfg_nonmask_data,
                hip_mask_data,
                hip_nonmask_data,
                caud_mask_data,
                caud_nonmask_data]

    yticklabels = ["Vessel Space - All Points",
                   "Non-Vessel Space - All Points",
                   "Vessel Space - MFG",
                   "Non-Vessel Space - MFG",
                   "Vessel Space - HIP",
                   "Non-Vessel Space - HIP",
                   "Vessel Space - CAUD",
                   "Non-Vessel Space - CAUD"]

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
                     xticklabels=markers_names,
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
        h_line_idx += 2
        ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

    output_dir = "results/heatmaps"
    mkdir_p(output_dir)

    plt.savefig(output_dir + '/mask_nonmask_heatmap_%s_expansions.png' % str(n_expansions - 1), bbox_inches='tight')
    plt.clf()

    ax = sns.clustermap(all_data,
                        cmap=cmap,
                        row_cluster=False,
                        col_cluster=True,
                        linewidths=0,
                        xticklabels=markers_names,
                        yticklabels=yticklabels,
                        figsize=(22, 10)
                        )

    output_dir = "results/clustermaps"
    mkdir_p(output_dir)

    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    ax.ax_heatmap.yaxis.tick_left()
    ax.ax_heatmap.yaxis.set_label_position("left")

    ax.savefig(output_dir + '/mask_nonmask_heatmap_%s_expansions.png' % str(n_expansions - 1))
    plt.clf()


def vessel_areas_histogram(show_outliers=False):
    '''
    Create visualizations of vessel areas

    :return: None
    '''

    masks = [
        'astrocytes',
        'BBB',
        'largevessels',
        'microglia',
        'myelin',
        'plaques',
        'tangles',
        'allvessels'
    ]

    region_names = [
        "MFG",
        "HIP",
        "CAUD"
    ]

    total_areas = [[], [], []]

    brain_regions = [(1, 16),
                     (17, 32),
                     (33, 48)]

    for segmentation_type in masks:
        current_point = 1
        current_region = 0

        marker_segmentation_masks, markers_data, markers_names = get_all_point_data(segmentation_type=segmentation_type)

        contour_images_multiple_points = []
        contour_data_multiple_points = []

        for segmentation_mask in marker_segmentation_masks:
            contour_images, contours = extract_cell_contours(segmentation_mask, show=False)
            contour_images_multiple_points.append(contour_images)
            contour_data_multiple_points.append(contours)

        vessel_areas = plot_vessel_areas(contour_data_multiple_points, marker_segmentation_masks,
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

    plt.show()
