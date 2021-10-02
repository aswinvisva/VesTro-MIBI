from datetime import datetime

from config.config_settings import Config
from src.data_analysis._shape_quantification_metrics import *
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_analysis.positive_vessel_summary_analyzer import PositiveVesselSummaryAnalyzer
from src.data_analysis.shape_quantification_analyzer import ShapeQuantificationAnalyzer
from src.data_analysis.linear_regression_analyzer import LinearRegressionAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.mibi_pipeline import MIBIPipeline
from src.utils.utils_functions import mkdir_p


def medres_example():
    conf = Config()

    medres_feed = MIBIDataFeed(
        feed_data_loc="/media/large_storage/oliveria_data/data/medres",
        feed_mask_loc="/media/large_storage/oliveria_data/masks/medres",
        feed_name="Medres",
        n_points=300,
        points_per_dir=100,
        brain_region_point_ranges=[(1, 100), (101, 200), (201, 300)]
    )

    loader = MIBILoader(conf)
    loader.add_feed(medres_feed)
    loader.read()


def hires_example():
    conf = Config()

    results_dir = "/Users/aswinvisva/Documents/oliveria_data/results/test/"
    mkdir_p(results_dir)

    hires_feed = MIBIDataFeed(
        feed_data_loc="/Users/aswinvisva/Documents/oliveria_data/data/",
        feed_mask_loc="/Users/aswinvisva/Documents/oliveria_data//masks/",
        feed_name="Hires",
        n_points=48,
        brain_region_point_ranges=[(1, 16), (17, 32), (33, 48)],
        brain_region_names=["MFG", "HIP", "CAUD"]
    )

    pipe = MIBIPipeline(conf, results_dir, csv_loc="/Users/aswinvisva/Documents/oliveria_data/results/5_um_in_out.csv")
    pipe.add_feed(hires_feed)
    pipe.load_preprocess_data()

    pipe.add_analyzer(ShapeQuantificationAnalyzer)
    pipe.add_analyzer(LinearRegressionAnalyzer)
    # pipe.add_analyzer(PositiveVesselSummaryAnalyzer)
    # pipe.add_analyzer(DimensionalityReductionClusteringAnalyzer)
    # #
    shape_quantification_method = {
        "Name": "Contour Area",
        "Metric": contour_area
    }

    pipe.analyze_data(marker_settings="all_markers",
                      shape_quantification_method=shape_quantification_method)

    # pipe.visualizer.expansion_line_plots_per_point(n_expansions=10)
    # pipe.visualizer.expansion_line_plots_all_points(n_expansions=10)
    # pipe.visualizer.expansion_line_plots_per_region(n_expansions=10)
    #
    # pipe.visualizer.obtain_expanded_vessel_masks()
    # pipe.visualizer.pixel_expansion_ring_plots()
    #
    # pipe.visualizer.expression_histogram()
    # # pipe.visualizer.biaxial_scatter_plot()
    # pipe.visualizer.average_quartile_violin_plot_subplots()
    # pipe.visualizer.categorical_violin_plot_with_images()
    # pipe.visualizer.vessel_nonvessel_masks(n_expansions=10)
    # pipe.visualizer.categorical_violin_plot(primary_categorical_analysis_variable=shape_quantification_method["Name"])
    # pipe.visualizer.violin_plot_brain_expansion(primary_categorical_analysis_variable=shape_quantification_method["Name"])
    # pipe.visualizer.violin_plot_brain_expansion(n_expansions=10)
    # pipe.visualizer.box_plot_brain_expansions(n_expansions=10)
    # pipe.visualizer.categorical_spatial_probability_maps()
    # pipe.visualizer.spatial_probability_maps()
    # pipe.visualizer.vessel_images_by_categorical_variable(primary_categorical_analysis_variable=shape_quantification_method["Name"])
    # pipe.visualizer.scatter_plot_umap_marker_projection()
    # pipe.visualizer.continuous_scatter_plot()
    # pipe.visualizer.vessel_nonvessel_heatmap(n_expansions=10)
    # pipe.visualizer.categorical_split_expansion_heatmap_clustermap(n_expansions=10)
    # pipe.visualizer.marker_covariance_heatmap()
    # pipe.visualizer.brain_region_expansion_heatmap(n_expansions=10)
    # # pipe.visualizer.marker_expression_masks()
    # pipe.visualizer.pseudo_time_heatmap()
    # pipe.visualizer.vessel_shape_area_spread_plot()
    # pipe.visualizer.vessel_nonvessel_masks(n_expansions=10)
    # pipe.visualizer.removed_vessel_expression_boxplot()


if __name__ == '__main__':
    hires_example()
