import torch

from dnn_vessel_heterogeneity.cnn_dataloader import VesselDataset, preprocess_input
from dnn_vessel_heterogeneity.feature_extraction.cnn_feature_gen import CNNFeatureGen
from dnn_vessel_heterogeneity.feature_extraction.sift_feature_gen import SIFTFeatureGen
from dnn_vessel_heterogeneity.models.clustering_helper import ClusteringHelper
from dnn_vessel_heterogeneity.models.vessel_net import VesselNet
from utils.mibi_reader import get_all_point_data
from utils.extract_vessel_contours import *
from utils.markers_feature_gen import *
from utils.visualizer import vessel_nonvessel_heatmap, point_region_plots, vessel_region_plots, brain_region_plots, \
    all_points_plots, brain_region_expansion_heatmap, marker_expression_masks, vessel_areas_histogram
import config.config_settings as config


def extract_vessel_heterogeneity(n=56,
                                 feature_extraction_method="vesselnet"):
    """
    Extract vessel heterogeneity
    :return:
    """

    marker_segmentation_masks, markers_data, markers_names = get_all_point_data()

    contour_data_multiple_points = []
    contour_images_multiple_points = []

    for segmentation_mask in marker_segmentation_masks:
        contour_images, contours = extract(segmentation_mask, show=False)
        contour_data_multiple_points.append(contours)
        contour_images_multiple_points.append(contour_images)

    for i in range(len(contour_data_multiple_points)):
        contours = contour_data_multiple_points[i]
        marker_data = markers_data[i]
        get_microenvironment_masks(marker_data, contours, pixel_expansion_upper_bound=10, pixel_expansion_lower_bound=0)

    if feature_extraction_method == "vesselnet":
        data_loader = VesselDataset(contour_data_multiple_points, markers_data, batch_size=16, n=n)

        # vn = VesselNet(n=n)
        # vn.fit(data_loader, epochs=150)
        # vn.visualize_filters()
        # torch.save(vn.state_dict(), "trained_models/vessel_net_100.pth")

        vn = VesselNet(n=n)
        vn.load_state_dict(torch.load("trained_models/vessel_net_100.pth"))
        vn.to(torch.device("cuda"))
        # vn.visualize_filters()

        encoder_output = []
        marker_expressions = []

        for i in range(len(contour_data_multiple_points)):
            for x in range(len(contour_data_multiple_points[i])):
                contours = [contour_data_multiple_points[i][x]]
                marker_data = markers_data[i]
                expression, expression_images, stopped_vessels, _ = calculate_microenvironment_marker_expression(
                    marker_data,
                    contours,
                    plot=False)

                expression_image = preprocess_input(expression_images, n)
                expression_image = torch.unsqueeze(expression_image, 0)

                reconstructed_img, output = vn.forward(expression_image)

                y_pred_numpy = reconstructed_img.cpu().data.numpy()
                y_true_numpy = expression_image.cpu().data.numpy()

                row_i = np.random.choice(y_pred_numpy.shape[0], 1)
                random_pic = y_pred_numpy[row_i, :, :, :]
                random_pic = random_pic.reshape(34, n, n)

                true = y_true_numpy[row_i, :, :, :]
                true = true.reshape(34, n, n)

                # for w in range(len(random_pic)):
                #     cv.imshow("Predicted", random_pic[w] * 255)
                #     cv.imshow("True", true[w] * 255)
                #     cv.waitKey(0)

                output = output.cpu().data.numpy()
                encoder_output.append(output.reshape(2048))
                marker_expressions.append(expression[0])

    elif feature_extraction_method == "sift":
        flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

        sift = SIFTFeatureGen()
        encoder_output = sift.generate(flat_list)
    elif feature_extraction_method == "resnet":
        flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

        cnn = CNNFeatureGen()
        encoder_output = cnn.generate(flat_list)

    km = ClusteringHelper(encoder_output, n_clusters=10, metric="cosine", method="kmeans")
    indices, frequency = km.fit_predict()

    print(len(indices))

    indices = [[i, indices[i]] for i in range(len(indices))]
    values = set(map(lambda y: y[1], indices))

    grouped_indices = [[y[0] for y in indices if y[1] == x] for x in values]

    marker_expressions_grouped = []

    for i, cluster in enumerate(grouped_indices):
        temp = []
        for idx in cluster:
            temp.append(marker_expressions[idx])

        average = np.mean(temp, axis=0)

        marker_expressions_grouped.append(average)

    marker_expressions_grouped = np.array(marker_expressions_grouped)
    print(marker_expressions_grouped.shape)

    km.plot(x=marker_expressions_grouped, labels=markers_names)

    k = 10

    sampled_indices = []

    for cluster in grouped_indices:
        sampled_indices.append(random.choices(cluster, k=k))

    flat_list = [item for sublist in contour_images_multiple_points for item in sublist]

    for i, cluster in enumerate(sampled_indices):
        for idx in cluster:
            cv.imshow("Cluster %s" % i, flat_list[idx])
            cv.waitKey(0)