import matplotlib
import matplotlib.pylab as pl
import seaborn as sns

from utils.mibi_reader import get_all_point_data
from utils.extract_vessel_contours import *
from utils.markers_feature_gen import *
from utils.utils_functions import mkdir_p

'''
Authors: Aswin Visva, John-Paul Oliveria, Ph.D
'''


def vessel_region_plots(n_expansions: int,
                        pixel_interval: int,
                        markers_names: list,
                        all_samples_features: pd.DataFrame):
    """
    Create vessel region line plots for all marker bins, average marker bins and per marker bins

    :param n_expansions: int, Number of expansions
    :param pixel_interval: int, Pixel interval
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param expansion_data: list, [n_expansions, n_points, n_vessels, n_markers] -> Microenvironment expansion data
    :return:
    """
    marker_clusters = config.marker_clusters
    color_maps = config.line_plots_color_maps
    colors = config.line_plots_bin_colors
    points = config.vessel_line_plots_points

    pixel_interval = abs(pixel_interval)

    pixel_expansions = np.array(range(0, n_expansions)) * pixel_interval
    pixel_expansions = pixel_expansions.tolist()

    marker_color_dict = {}

    for marker_cluster in marker_clusters.keys():
        for marker in marker_clusters[marker_cluster]:
            marker_color_dict[marker] = colors[marker_cluster]

    # Change in Marker Expression w.r.t pixel expansion per vessel (All bins)
    for point in points:
        output_dir = "%s/mean_per_vessel_per_point_per_brain_region/point_%s_vessels_%s_interval_" \
                     "%s_expansions_allbins" % (config.visualization_results_dir,
                                                str(point), str(pixel_interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        idx = pd.IndexSlice
        n_vessels = len(all_samples_features.loc[idx[point,
                                                 :,
                                                 0,
                                                 "Data"], :].to_numpy())

        for vessel in range(n_vessels):
            for key in marker_clusters.keys():
                plt.plot([], [], color=colors[key], label=key)

            for marker, marker_name in enumerate(markers_names):
                y = []
                color = marker_color_dict[marker_name]

                for i in range(n_expansions):
                    idx = pd.IndexSlice
                    vessel_data = all_samples_features.loc[idx[point,
                                                               vessel,
                                                               i,
                                                               "Data"], marker_name]
                    y.append(vessel_data)

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
        output_dir = "%s/mean_per_vessel_per_point_per_brain_region/point_%s_vessels_%s_interval_" \
                     "%s_expansions_averagebins" % (config.visualization_results_dir,
                                                    str(point), str(pixel_interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        idx = pd.IndexSlice
        n_vessels = len(all_samples_features.loc[idx[point,
                                                 :,
                                                 0,
                                                 "Data"], :].to_numpy())
        for vessel in range(n_vessels):
            for key in marker_clusters.keys():
                plt.plot([], [], color=colors[key], label=key)

            for key in marker_clusters.keys():
                color = colors[key]
                y_tot = []

                for i in range(n_expansions):
                    idx = pd.IndexSlice
                    all_vessels_data = all_samples_features.loc[idx[point,
                                                                    vessel,
                                                                    i,
                                                                    "Data"], marker_clusters[key]]
                    average_expression = np.mean(all_vessels_data)
                    y_tot.append(average_expression)

                plt.plot(pixel_expansions, y_tot, color=color)
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Vessel ID: %s, Point %s Average Bins" % (str(vessel), str(point)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.savefig(output_dir + '/vessel_%s_averagebins.png' % str(vessel), bbox_inches='tight')
            plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per vessel (One plot per bin)
    for point in points:
        output_dir = "%s/mean_per_vessel_per_point_per_brain_region/point_%s_vessels_%s_interval_" \
                     "%s_expansions_perbin" % (config.visualization_results_dir,
                                               str(point), str(pixel_interval), str(n_expansions - 1))
        mkdir_p(output_dir)

        idx = pd.IndexSlice
        n_vessels = len(all_samples_features.loc[idx[point,
                                                 :,
                                                 0,
                                                 "Data"], :].to_numpy())
        for vessel in range(n_vessels):
            for key in marker_clusters.keys():

                colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
                color_idx = 2

                for marker, marker_name in enumerate(marker_clusters[key]):
                    y = []

                    for i in range(n_expansions):
                        idx = pd.IndexSlice
                        all_vessels_data = all_samples_features.loc[idx[point,
                                                                        vessel,
                                                                        i,
                                                                        "Data"], marker_name]
                        average_expression = np.mean(all_vessels_data)
                        y.append(average_expression)

                    plt.plot(pixel_expansions, y, label=marker_name, color=colors_clusters[color_idx])
                    color_idx += 1
                plt.xticks(pixel_expansions, fontsize=7)
                plt.xlabel("# of Pixels Expanded")
                plt.ylabel("Mean Pixel Expression")
                plt.title("Vessel ID: %s, Point %s %s" % (str(vessel), str(point), key))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                plt.savefig(output_dir + '/vessel_%s_%s.png' % (str(vessel), key), bbox_inches='tight')
                plt.clf()


def point_region_plots(n_expansions: int, pixel_interval: int, markers_names: list, all_samples_features: pd.DataFrame):
    """
    Create point region line plots for all marker bins, average marker bins and per marker bins

    :param n_expansions: int, Number of expansions
    :param pixel_interval: int, Pixel interval
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param expansion_data: list, [n_expansions, n_points, n_vessels, n_markers] -> Microenvironment expansion data
    :return:
    """

    n_points = config.n_points
    pixel_interval = abs(pixel_interval)

    marker_clusters = config.marker_clusters
    color_maps = config.line_plots_color_maps
    colors = config.line_plots_bin_colors

    pixel_expansions = np.array(range(0, n_expansions)) * pixel_interval
    pixel_expansions = pixel_expansions.tolist()

    # Points Plots
    output_dir = "%s/mean_per_point_per_brain_region/points_%s_interval_%s_expansions" % (
        config.visualization_results_dir,
        str(pixel_interval), str(n_expansions - 1))
    mkdir_p(output_dir)

    marker_color_dict = {}

    for marker_cluster in marker_clusters.keys():
        for marker in marker_clusters[marker_cluster]:
            marker_color_dict[marker] = colors[marker_cluster]

    # Change in Marker Expression w.r.t pixel expansion per point (All bins)
    for point in range(n_points):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for marker, marker_name in enumerate(markers_names):
            y = []
            color = marker_color_dict[marker_name]

            for i in range(n_expansions):
                idx = pd.IndexSlice
                all_vessels_data = all_samples_features.loc[idx[point + 1,
                                                            :,
                                                            i,
                                                            "Data"], marker_name].to_numpy()
                average_expression = np.mean(all_vessels_data)

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

            for i in range(n_expansions):
                idx = pd.IndexSlice
                all_vessels_data = all_samples_features.loc[idx[point + 1,
                                                            :,
                                                            i,
                                                            "Data"], marker_clusters[key]].to_numpy()
                average_expression = np.mean(all_vessels_data)
                y_tot.append(average_expression)

            plt.plot(pixel_expansions, y_tot, color=color)
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

            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2
            for marker, marker_name in enumerate(marker_clusters[key]):
                y = []

                for i in range(n_expansions):
                    idx = pd.IndexSlice
                    all_vessels_data = all_samples_features.loc[idx[point + 1,
                                                                :,
                                                                i,
                                                                "Data"], marker_name].to_numpy()
                    average_expression = np.mean(all_vessels_data)
                    y.append(average_expression)

                plt.plot(pixel_expansions, y, label=marker_name, color=colors_clusters[color_idx])
                color_idx += 1
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Point %s" % str(point + 1))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.savefig(output_dir + '/point_%s_%s.png' % (str(point + 1), str(key)), bbox_inches='tight')
            plt.clf()


def all_points_plots(n_expansions: int, pixel_interval: int, markers_names: list, all_samples_features: pd.DataFrame):
    """
    Create all points average region line plots for all marker bins, average marker bins and per marker bins

    :param n_expansions: int, Number of expansions
    :param pixel_interval: int, Pixel interval
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param expansion_data: list, [n_expansions, n_points, n_vessels, n_markers] -> Microenvironment expansion data
    :return:
    """

    marker_clusters = config.marker_clusters
    colors = config.line_plots_bin_colors
    color_maps = config.line_plots_color_maps

    pixel_expansions = np.array(range(0, n_expansions)) * pixel_interval
    pixel_expansions = pixel_expansions.tolist()

    output_dir = "%s/all_points/%s_expansions_%s_pix_interval" % (
        config.visualization_results_dir, str(n_expansions), str(pixel_interval))
    mkdir_p(output_dir)

    marker_color_dict = {}

    for marker_cluster in marker_clusters.keys():
        for marker in marker_clusters[marker_cluster]:
            marker_color_dict[marker] = colors[marker_cluster]

    # All Points Averages

    # Change in Marker Expression w.r.t pixel expansion per point (All bins)
    for key in marker_clusters.keys():
        plt.plot([], [], color=colors[key], label=key)

    for marker, marker_name in enumerate(markers_names):
        y = []
        color = marker_color_dict[marker_name]

        for i in range(n_expansions):
            idx = pd.IndexSlice
            all_vessels_data = all_samples_features.loc[idx[:,
                                                        :,
                                                        i,
                                                        "Data"], marker_name].to_numpy()
            average_expression = np.mean(all_vessels_data)

            y.append(average_expression)
        plt.plot(pixel_expansions, y, color=color)
    plt.xticks(pixel_expansions, fontsize=7)
    plt.xlabel("# of Pixels Expanded")
    plt.ylabel("Mean Pixel Expression")
    plt.title("All Points")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(output_dir + '/all_points_allbins.png', bbox_inches='tight')
    plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per point (Average bins)
    for key in marker_clusters.keys():
        plt.plot([], [], color=colors[key], label=key)

    for key in marker_clusters.keys():
        color = colors[key]
        y_tot = []

        for i in range(n_expansions):
            idx = pd.IndexSlice
            all_vessels_data = all_samples_features.loc[idx[:,
                                                        :,
                                                        i,
                                                        "Data"], marker_clusters[key]].to_numpy()
            average_expression = np.mean(all_vessels_data)
            y_tot.append(average_expression)

        plt.plot(pixel_expansions, y_tot, color=color)
    plt.xticks(pixel_expansions, fontsize=7)
    plt.xlabel("# of Pixels Expanded")
    plt.ylabel("Mean Pixel Expression")
    plt.title("All Points")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(output_dir + '/all_points_averagebins.png', bbox_inches='tight')
    plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per point (One bin per plot)
    for key in marker_clusters.keys():

        colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
        color_idx = 2
        for marker, marker_name in enumerate(marker_clusters[key]):
            y = []

            for i in range(n_expansions):
                idx = pd.IndexSlice
                all_vessels_data = all_samples_features.loc[idx[:,
                                                            :,
                                                            i,
                                                            "Data"], marker_name].to_numpy()
                average_expression = np.mean(all_vessels_data)
                y.append(average_expression)

            plt.plot(pixel_expansions, y, label=marker_name, color=colors_clusters[color_idx])
            color_idx += 1

        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("All Points - %s" % str(key))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(output_dir + '/all_points_%s.png' % str(key), bbox_inches='tight')
        plt.clf()


def brain_region_plots(n_expansions: int, pixel_interval: int, markers_names: list, all_samples_features: pd.DataFrame):
    """
    Create brain region average region line plots for all marker bins, average marker bins and per marker bins

    :param n_expansions: int, Number of expansions
    :param pixel_interval: int, Pixel interval
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param expansion_data: list, [n_expansions, n_points, n_vessels, n_markers] -> Microenvironment expansion data
    :return:
    """

    brain_regions = config.brain_region_point_ranges
    pixel_interval = abs(pixel_interval)

    marker_clusters = config.marker_clusters
    colors = config.line_plots_bin_colors
    color_maps = config.line_plots_color_maps
    region_names = config.brain_region_names

    pixel_expansions = np.array(range(0, n_expansions)) * pixel_interval
    pixel_expansions = pixel_expansions.tolist()

    output_dir = "%s/mean_per_brain_region/brain_regions_%s_interval_%s_expansions" % (
        config.visualization_results_dir, str(pixel_interval), str(n_expansions - 1))
    mkdir_p(output_dir)

    marker_color_dict = {}

    for marker_cluster in marker_clusters.keys():
        for marker in marker_clusters[marker_cluster]:
            marker_color_dict[marker] = colors[marker_cluster]

    # Brain Region Plots

    # Change in Marker Expression w.r.t pixel expansion per brain region (All bins in one graph)
    for region_idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for marker, marker_name in enumerate(markers_names):
            y = []
            color = marker_color_dict[marker_name]

            for i in range(n_expansions):
                idx = pd.IndexSlice
                all_vessels_data = all_samples_features.loc[idx[region[0]:region[1],
                                                            :,
                                                            i,
                                                            "Data"], marker_name].to_numpy()

                average_expression = np.mean(all_vessels_data)

                y.append(average_expression)

            plt.plot(pixel_expansions, y, color=color)

        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s - All Bins" % str(region_names[region_idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(output_dir + '/region_%s_allbins.png' % str(region_names[region_idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (Bins averaged into one line)
    for region_idx, region in enumerate(brain_regions):
        for key in marker_clusters.keys():
            plt.plot([], [], color=colors[key], label=key)

        for key in marker_clusters.keys():
            color = colors[key]
            y_tot = []
            for i in range(n_expansions):
                idx = pd.IndexSlice
                marker_cluster_data = all_samples_features.loc[idx[region[0]:region[1],
                                                               :,
                                                               i,
                                                               "Data"], marker_clusters[key]].to_numpy()

                y_tot.append(np.mean(marker_cluster_data))

            plt.plot(pixel_expansions, y_tot, color=color)
        plt.xticks(pixel_expansions, fontsize=7)
        plt.xlabel("# of Pixels Expanded")
        plt.ylabel("Mean Pixel Expression")
        plt.title("Brain Region - %s - Average Bins" % str(region_names[region_idx]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(output_dir + '/region_%s_averagebins.png' % str(region_names[region_idx]), bbox_inches='tight')
        plt.clf()

    # Change in Marker Expression w.r.t pixel expansion per brain region (One bin per plot)
    for region_idx, region in enumerate(brain_regions):
        x = np.arange(n_expansions)

        for key in marker_clusters.keys():

            colors_clusters = color_maps[key](np.linspace(0, 1, len(marker_clusters[key]) + 2))
            color_idx = 2
            for marker, marker_name in enumerate(marker_clusters[key]):
                y = []
                for i in range(n_expansions):
                    idx = pd.IndexSlice
                    marker_data = all_samples_features.loc[idx[region[0]:region[1],
                                                           :,
                                                           i,
                                                           "Data"], marker_name].to_numpy()

                    average_expression = np.mean(marker_data)
                    y.append(average_expression)

                plt.plot(pixel_expansions, y, label=marker_name, color=colors_clusters[color_idx])
                color_idx += 1
            plt.xticks(pixel_expansions, fontsize=7)
            plt.xlabel("# of Pixels Expanded")
            plt.ylabel("Mean Pixel Expression")
            plt.title("Brain Region - %s - %s" % (str(region_names[region_idx]), str(key)))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.savefig(output_dir + '/region_%s_%s.png' % (str(region_names[region_idx]), str(key)),
                        bbox_inches='tight')
            plt.clf()


def pixel_expansion_ring_plots():
    """
    Pixel expansion "Ring" plots
    """

    n_expansions = config.max_expansions
    interval = config.pixel_interval
    mask = config.selected_segmentation_mask_type
    n_points = config.n_points
    expansions = config.expansion_to_run

    parent_dir = "%s/ring_plots" % config.visualization_results_dir
    mkdir_p(parent_dir)

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data()

    all_points_vessel_contours = []
    contour_images_multiple_points = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, per_point_vessel_contours, removed_contours = extract(segmentation_mask)
        all_points_vessel_contours.append(per_point_vessel_contours)
        contour_images_multiple_points.append(contour_images)

    for point_num in range(n_points):
        current_interval = interval

        expansion_image = np.zeros(markers_data[0][0].shape, np.uint8)

        colors = pl.cm.Greys(np.linspace(0, 1, n_expansions + 10))

        for x in range(n_expansions):
            per_point_vessel_contours = all_points_vessel_contours[point_num]
            expansion_ring_plots(per_point_vessel_contours,
                                 expansion_image,
                                 pixel_expansion_upper_bound=current_interval,
                                 pixel_expansion_lower_bound=current_interval - interval,
                                 color=colors[x + 5] * 255)
            print(
                "Current interval %s, previous interval %s" % (str(current_interval), str(current_interval - interval)))

            if x + 1 in expansions:
                child_dir = parent_dir + "/expansion_%s" % str(x + 1)
                mkdir_p(child_dir)

                cv.imwrite(child_dir + "/point_%s.png" % str(point_num + 1), expansion_image)

            current_interval += interval


def vessel_nonvessel_heatmap(all_samples_features: pd.DataFrame,
                             markers_names: list,
                             n_expansions: int):
    """
    Vessel/Non-vessel heatmaps for marker expression

    :param all_samples_features: pd.DataFrame, All features
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param n_expansions: int, Number of expansions
    :return:
    """
    brain_regions = config.brain_region_point_ranges
    marker_clusters = config.marker_clusters

    # Vessel Space

    idx = pd.IndexSlice
    all_vessels_data = all_samples_features.loc[idx[:, :, 0,
                                                "Data"], :].to_numpy()
    mfg_vessels_data = all_samples_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                :,
                                                0,
                                                "Data"], :].to_numpy()
    hip_vessels_data = all_samples_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                :,
                                                0,
                                                "Data"], :].to_numpy()
    caud_vessels_data = all_samples_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                 :,
                                                 0,
                                                 "Data"], :].to_numpy()

    all_vessels_data = np.mean(all_vessels_data, axis=0)
    mfg_vessels_data = np.mean(mfg_vessels_data, axis=0)
    hip_vessels_data = np.mean(hip_vessels_data, axis=0)
    caud_vessels_data = np.mean(caud_vessels_data, axis=0)

    # Non-vessel Space

    idx = pd.IndexSlice
    all_nonmask_data = all_samples_features.loc[idx[:, :, :,
                                                "Non-Vascular Space"], :].to_numpy()
    mfg_nonmask_data = all_samples_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                :,
                                                1:n_expansions,
                                                "Non-Vascular Space"], :].to_numpy()
    hip_nonmask_data = all_samples_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                :,
                                                1:n_expansions,
                                                "Non-Vascular Space"], :].to_numpy()
    caud_nonmask_data = all_samples_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                 :,
                                                 1:n_expansions,
                                                 "Non-Vascular Space"], :].to_numpy()

    all_nonmask_data = np.mean(all_nonmask_data, axis=0)
    mfg_nonmask_data = np.mean(mfg_nonmask_data, axis=0)
    hip_nonmask_data = np.mean(hip_nonmask_data, axis=0)
    caud_nonmask_data = np.mean(caud_nonmask_data, axis=0)

    # Vessel environment space

    idx = pd.IndexSlice
    all_vessels_environment_data = all_samples_features.loc[idx[:, :,
                                                            1:n_expansions,
                                                            "Vascular Space"], :].to_numpy()
    mfg_vessels_environment_data = all_samples_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                            :,
                                                            1:n_expansions,
                                                            "Vascular Space"], :].to_numpy()
    hip_vessels_environment_data = all_samples_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                            :,
                                                            1:n_expansions,
                                                            "Vascular Space"], :].to_numpy()
    caud_vessels_environment_data = all_samples_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                             :,
                                                             1:n_expansions,
                                                             "Vascular Space"], :].to_numpy()

    all_vessels_environment_data = np.mean(all_vessels_environment_data, axis=0)
    mfg_vessels_environment_data = np.mean(mfg_vessels_environment_data, axis=0)
    hip_vessels_environment_data = np.mean(hip_vessels_environment_data, axis=0)
    caud_vessels_environment_data = np.mean(caud_vessels_environment_data, axis=0)

    all_data = [all_vessels_data,
                all_vessels_environment_data,
                all_nonmask_data,
                mfg_vessels_data,
                mfg_vessels_environment_data,
                mfg_nonmask_data,
                hip_vessels_data,
                hip_vessels_environment_data,
                hip_nonmask_data,
                caud_vessels_data,
                caud_vessels_environment_data,
                caud_nonmask_data]

    yticklabels = ["Vascular Space - All Points",
                   "Vascular Expansion Space - All Points",
                   "Non-Vascular Space - All Points",
                   "Vascular Space - MFG",
                   "Vascular Expansion Space - MFG",
                   "Non-Vascular Space - MFG",
                   "Vascular Space - HIP",
                   "Vascular Expansion Space - HIP",
                   "Non-Vascular Space - HIP",
                   "Vascular Space - CAUD",
                   "Vascular Expansion Space - CAUD",
                   "Non-Vascular Space - CAUD"]

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
        h_line_idx += 3
        ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

    output_dir = "%s/heatmaps" % config.visualization_results_dir
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
                        figsize=(20, 10)
                        )

    output_dir = "%s/clustermaps" % config.visualization_results_dir
    mkdir_p(output_dir)

    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    ax.ax_heatmap.yaxis.tick_left()
    ax.ax_heatmap.yaxis.set_label_position("left")

    ax.savefig(output_dir + '/mask_nonmask_heatmap_%s_expansions.png' % str(n_expansions - 1))
    plt.clf()


def brain_region_expansion_heatmap(all_samples_features: pd.DataFrame,
                                   markers_names: list,
                                   n_expansions: int,
                                   pixel_interval: int):
    """
    Brain Region Expansion Heatmap

    :param all_samples_features: pd.DataFrame, All samples features
    :param pixel_interval: int, Pixel interval for expansion
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param n_expansions: int, Number of expansions
    """

    brain_regions = config.brain_region_point_ranges
    marker_clusters = config.marker_clusters

    all_mask_data = []
    mfg_mask_data = []
    hip_mask_data = []
    caud_mask_data = []

    for i in range(n_expansions):
        if i == 0:
            idx = pd.IndexSlice
            current_expansion_all = all_samples_features.loc[idx[:, :,
                                                             i,
                                                             "Data"], :].to_numpy()
            current_expansion_mfg = all_samples_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                             :,
                                                             i,
                                                             "Data"], :].to_numpy()
            current_expansion_hip = all_samples_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                             :,
                                                             i,
                                                             "Data"], :].to_numpy()
            current_expansion_caud = all_samples_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                              :,
                                                              i,
                                                              "Data"], :].to_numpy()
        else:
            idx = pd.IndexSlice
            current_expansion_all = all_samples_features.loc[idx[:, :,
                                                             i,
                                                             "Vascular Space"], :].to_numpy()
            current_expansion_mfg = all_samples_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                             :,
                                                             i,
                                                             "Vascular Space"], :].to_numpy()
            current_expansion_hip = all_samples_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                             :,
                                                             i,
                                                             "Vascular Space"], :].to_numpy()
            current_expansion_caud = all_samples_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                              :,
                                                              i,
                                                              "Vascular Space"], :].to_numpy()

        all_mask_data.append(np.mean(np.array(current_expansion_all), axis=0))
        mfg_mask_data.append(np.mean(np.array(current_expansion_mfg), axis=0))
        hip_mask_data.append(np.mean(np.array(current_expansion_hip), axis=0))
        caud_mask_data.append(np.mean(np.array(current_expansion_caud), axis=0))

    all_mask_data = np.array(all_mask_data)
    mfg_mask_data = np.array(mfg_mask_data)
    hip_mask_data = np.array(hip_mask_data)
    caud_mask_data = np.array(caud_mask_data)

    idx = pd.IndexSlice
    all_nonmask_data = all_samples_features.loc[idx[:, :,
                                                i,
                                                "Non-Vascular Space"], :].to_numpy()
    mfg_nonmask_data = all_samples_features.loc[idx[brain_regions[0][0]:brain_regions[0][1],
                                                :,
                                                i,
                                                "Non-Vascular Space"], :].to_numpy()
    hip_nonmask_data = all_samples_features.loc[idx[brain_regions[1][0]:brain_regions[1][1],
                                                :,
                                                i,
                                                "Non-Vascular Space"], :].to_numpy()
    caud_nonmask_data = all_samples_features.loc[idx[brain_regions[2][0]:brain_regions[2][1],
                                                 :,
                                                 i,
                                                 "Non-Vascular Space"], :].to_numpy()

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

    x_tick_labels = np.array(range(0, n_expansions)) * pixel_interval
    x_tick_labels = x_tick_labels.tolist()
    x_tick_labels = [str(x) + " Pixel" for x in x_tick_labels]
    x_tick_labels.append("Nonvessel Space")

    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "black"],
              [norm(-0.5), "indigo"],
              [norm(0), "firebrick"],
              [norm(0.5), "orange"],
              [norm(1.0), "khaki"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    mkdir_p("%s/brain_region_expansion_heatmaps" % config.visualization_results_dir)
    mkdir_p("%s/brain_region_expansion_clustermaps" % config.visualization_results_dir)

    # Heatmaps Output

    output_dir = "%s/brain_region_expansion_heatmaps/%s_expansions" % (
        config.visualization_results_dir, str(n_expansions - 1))
    mkdir_p(output_dir)

    plt.figure(figsize=(22, 10))

    ax = sns.heatmap(all_mask_data,
                     cmap=cmap,
                     xticklabels=x_tick_labels,
                     yticklabels=markers_names,
                     linewidths=0,
                     )

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    h_line_idx = 0

    for key in marker_clusters.keys():
        if h_line_idx != 0:
            ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

        for _ in marker_clusters[key]:
            h_line_idx += 1

    plt.savefig(output_dir + '/All_Points.png', bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(22, 10))

    ax = sns.heatmap(mfg_mask_data,
                     cmap=cmap,
                     xticklabels=x_tick_labels,
                     yticklabels=markers_names,
                     linewidths=0,
                     )

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    h_line_idx = 0

    for key in marker_clusters.keys():
        if h_line_idx != 0:
            ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

        for _ in marker_clusters[key]:
            h_line_idx += 1

    plt.savefig(output_dir + '/MFG_Region.png', bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(22, 10))

    ax = sns.heatmap(hip_mask_data,
                     cmap=cmap,
                     xticklabels=x_tick_labels,
                     yticklabels=markers_names,
                     linewidths=0,
                     )

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    h_line_idx = 0

    for key in marker_clusters.keys():
        if h_line_idx != 0:
            ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

        for _ in marker_clusters[key]:
            h_line_idx += 1

    plt.savefig(output_dir + '/HIP_Region.png', bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(22, 10))

    ax = sns.heatmap(caud_mask_data,
                     cmap=cmap,
                     xticklabels=x_tick_labels,
                     yticklabels=markers_names,
                     linewidths=0,
                     )

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    h_line_idx = 0

    for key in marker_clusters.keys():
        if h_line_idx != 0:
            ax.axhline(h_line_idx, 0, len(markers_names), linewidth=3, c='w')

        for _ in marker_clusters[key]:
            h_line_idx += 1

    plt.savefig(output_dir + '/CAUD_Region.png', bbox_inches='tight')
    plt.clf()

    # Clustermaps Outputs

    output_dir = "%s/brain_region_expansion_clustermaps/%s_expansions" % (
        config.visualization_results_dir, str(n_expansions - 1))
    mkdir_p(output_dir)

    ax = sns.clustermap(all_mask_data,
                        cmap=cmap,
                        row_cluster=True,
                        col_cluster=False,
                        linewidths=0,
                        xticklabels=x_tick_labels,
                        yticklabels=markers_names,
                        figsize=(20, 10)
                        )

    # ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    ax.savefig(output_dir + '/All_Points.png')
    plt.clf()

    ax = sns.clustermap(mfg_mask_data,
                        cmap=cmap,
                        row_cluster=True,
                        col_cluster=False,
                        linewidths=0,
                        xticklabels=x_tick_labels,
                        yticklabels=markers_names,
                        figsize=(20, 10)
                        )

    # ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    ax.savefig(output_dir + '/MFG_Region.png')
    plt.clf()

    ax = sns.clustermap(hip_mask_data,
                        cmap=cmap,
                        row_cluster=True,
                        col_cluster=False,
                        linewidths=0,
                        xticklabels=x_tick_labels,
                        yticklabels=markers_names,
                        figsize=(20, 10)
                        )

    # ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    ax.savefig(output_dir + '/HIP_Region.png')
    plt.clf()

    ax = sns.clustermap(caud_mask_data,
                        cmap=cmap,
                        row_cluster=True,
                        col_cluster=False,
                        linewidths=0,
                        xticklabels=x_tick_labels,
                        yticklabels=markers_names,
                        figsize=(20, 10)
                        )

    # ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    ax.savefig(output_dir + '/CAUD_Region.png')
    plt.clf()


def marker_expression_masks(all_points_vessel_contours: list,
                            all_points_marker_data: list,
                            markers_names: list):
    """
    Marker Expression Overlay Masks

    :param all_points_vessel_contours: array_like, [n_points, n_vessels] -> list of vessel contours for each point
    :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]] -> list of marker
    data for each point
    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :return:
    """

    parent_dir = "%s/expression_masks" % config.visualization_results_dir
    mkdir_p(parent_dir)

    for i in range(len(all_points_vessel_contours)):
        point_dir = parent_dir + "/Point_%s" % str(i + 1)
        mkdir_p(point_dir)

        contours = all_points_vessel_contours[i]
        marker_data = all_points_marker_data[i]

        img_shape = marker_data[0].shape

        expression_img = np.zeros(img_shape, np.uint8)
        expression_img = cv.cvtColor(expression_img, cv.COLOR_GRAY2BGR)

        data = calculate_composition_marker_expression(marker_data, contours, markers_names)

        for marker_idx, marker_name in enumerate(markers_names):
            for idx, vessel_vec in data.iterrows():
                color = plt.get_cmap('hot')(vessel_vec[marker_idx])
                color = (255 * color[0], 255 * color[1], 255 * color[2])

                cv.drawContours(expression_img, contours, idx[1], color, cv.FILLED)

            plt.imshow(expression_img)
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('hot'))
            plt.colorbar(sm)
            plt.savefig(os.path.join(point_dir, "%s.png" % marker_name))
            plt.clf()


def removed_vessel_expression_boxplot(all_points_vessel_contours: list,
                                      all_points_removed_vessel_contours: list,
                                      all_points_marker_data: list,
                                      markers_names: list):
    """
    Create kept vs. removed vessel expression comparison using Box Plots

    :param markers_names: array_like, [n_points, n_markers] -> Names of markers
    :param all_points_removed_vessel_contours: array_like, [n_points, n_vessels] -> list of removed vessel contours
    for each point
    :param all_points_vessel_contours: array_like, [n_points, n_vessels] -> list of vessel contours for each point
    :param all_points_marker_data: array_like, [n_points, n_markers, point_size[0], point_size[1]] ->
    list of marker data for each point
    """
    n_points = config.n_points

    all_points_vessels_expression = []
    all_points_removed_vessels_expression = []

    parent_dir = "%s/kept_removed_vessel_expression" % config.visualization_results_dir
    mkdir_p(parent_dir)

    # Iterate through each point
    for i in range(n_points):
        contours = all_points_vessel_contours[i]
        removed_contours = all_points_removed_vessel_contours[i]
        marker_data = all_points_marker_data[i]
        start_expression = datetime.datetime.now()

        vessel_expression_data = calculate_composition_marker_expression(marker_data,
                                                                         contours,
                                                                         markers_names,
                                                                         point_num=i + 1)
        removed_vessel_expression_data = calculate_composition_marker_expression(marker_data,
                                                                                 removed_contours,
                                                                                 markers_names,
                                                                                 point_num=i + 1)

        all_points_vessels_expression.append(vessel_expression_data)
        all_points_removed_vessels_expression.append(removed_vessel_expression_data)

        end_expression = datetime.datetime.now()

        print("Finished calculating expression for Point %s in %s" % (str(i + 1), end_expression - start_expression))

    kept_vessel_expression = pd.concat(all_points_vessels_expression).fillna(0)
    kept_vessel_expression.index = map(lambda a: (a[0], a[1], a[2], a[3], "Kept"), kept_vessel_expression.index)
    removed_vessel_expression = pd.concat(all_points_removed_vessels_expression).fillna(0)
    removed_vessel_expression.index = map(lambda a: (a[0], a[1], a[2], a[3], "Removed"),
                                          removed_vessel_expression.index)

    kept_removed_features = [kept_vessel_expression, removed_vessel_expression]
    kept_removed_features = pd.concat(kept_removed_features).fillna(0)
    kept_removed_features.index = pd.MultiIndex.from_tuples(kept_removed_features.index)
    kept_removed_features = kept_removed_features.sort_index()

    scaling_factor = config.scaling_factor
    transformation = config.transformation_type
    normalization = config.normalization_type
    n_markers = config.n_markers

    kept_removed_features = normalize_expression_data(kept_removed_features,
                                                      transformation=transformation,
                                                      normalization=normalization,
                                                      scaling_factor=scaling_factor,
                                                      n_markers=n_markers)

    print(kept_removed_features)

    brain_region_names = config.brain_region_names
    brain_region_point_ranges = config.brain_region_point_ranges
    markers_to_show = config.marker_clusters["Vessels"]

    all_points_per_brain_region_dir = "%s/all_points_per_brain_region" % parent_dir
    mkdir_p(all_points_per_brain_region_dir)

    average_points_dir = "%s/average_points_per_brain_region" % parent_dir
    mkdir_p(average_points_dir)

    all_points = "%s/all_points" % parent_dir
    mkdir_p(all_points)

    all_kept_removed_vessel_expression_data_collapsed = []

    for idx, brain_region in enumerate(brain_region_names):
        brain_region_range = brain_region_point_ranges[idx]

        vessel_removed_vessel_expression_data_collapsed = []

        idx = pd.IndexSlice
        per_point_point_vessel_data = kept_removed_features.loc[idx[brain_region_range[0]:brain_region_range[1], :,
                                                                :,
                                                                "Data",
                                                                "Kept"],
                                                                markers_to_show]

        per_point_point_removed_data = kept_removed_features.loc[idx[brain_region_range[0]:brain_region_range[1], :,
                                                                 :,
                                                                 "Data",
                                                                 "Removed"],
                                                                 markers_to_show]

        for index, vessel_data in per_point_point_vessel_data.iterrows():
            data = np.mean(vessel_data)
            vessel_removed_vessel_expression_data_collapsed.append(
                [data, "Kept", index[0]])

            all_kept_removed_vessel_expression_data_collapsed.append(
                [data, "Kept", index[0]])

        for index, vessel_data in per_point_point_removed_data.iterrows():
            data = np.mean(vessel_data)
            vessel_removed_vessel_expression_data_collapsed.append([data, "Removed",
                                                                    index[0]])

            all_kept_removed_vessel_expression_data_collapsed.append([data, "Removed",
                                                                      index[0]])

        df = pd.DataFrame(vessel_removed_vessel_expression_data_collapsed, columns=["Expression", "Vessel", "Point"])

        plt.title("Kept vs Removed Vessel Marker Expression - %s" % brain_region)
        ax = sns.boxplot(x="Point", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
        plt.savefig(os.path.join(all_points_per_brain_region_dir, "%s.png" % brain_region))
        plt.clf()

        plt.title("Kept vs Removed Vessel Marker Expression - %s: Average Points" % brain_region)
        ax = sns.boxplot(x="Vessel", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
        plt.savefig(os.path.join(average_points_dir, "%s.png" % brain_region))
        plt.clf()

    df = pd.DataFrame(all_kept_removed_vessel_expression_data_collapsed, columns=["Expression", "Vessel", "Point"])

    plt.title("Kept vs Removed Vessel Marker Expression - All Points")
    ax = sns.boxplot(x="Vessel", y="Expression", hue="Vessel", data=df, palette="Set3", showfliers=False)
    plt.savefig(os.path.join(all_points, "All_Points.png"))
    plt.clf()


def vessel_areas_histogram():
    """
    Create visualizations of vessel areas
    """

    masks = config.all_masks
    region_names = config.brain_region_names

    show_outliers = config.show_boxplot_outliers

    total_areas = [[], [], []]

    brain_regions = config.brain_region_point_ranges

    for segmentation_type in masks:
        current_point = 1
        current_region = 0

        marker_segmentation_masks, markers_data, markers_names = get_all_point_data()

        contour_images_multiple_points = []
        contour_data_multiple_points = []

        for segmentation_mask in marker_segmentation_masks:
            contour_images, contours, removed_contours = extract(segmentation_mask)
            contour_images_multiple_points.append(contour_images)
            contour_data_multiple_points.append(contours)

        vessel_areas = plot_vessel_areas(contour_data_multiple_points,
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


def plot_vessel_areas(all_points_vessel_contours: list,
                      save_csv: bool = False,
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

    brain_regions = config.brain_region_point_ranges
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

    if save_csv:
        area = pd.DataFrame(all_points_vessel_areas)
        area.to_csv('vessel_areas.csv')

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
