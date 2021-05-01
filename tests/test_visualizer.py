import tempfile
import unittest
from os import path

from config.config_settings import Config
from src.data_analysis.shape_quantification_analyzer import ShapeQuantificationAnalyzer
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data
from src.utils.utils_functions import round_to_nearest_half
from src.data_analysis._shape_quantification_metrics import *


class TestVisualizer(unittest.TestCase):

    def test_line_plots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.expansion_line_plots_per_vessel(n_expansions=1,
                                                            vessel_line_plots_points=[1])

            assert path.exists("%s/%s%s Expansion" % ("%s/Line Plots Per Vessel" % temp_dir,
                                                      str(round_to_nearest_half(config.pixel_interval *
                                                                                config.pixels_to_distance)),
                                                      config.data_resolution_units))

            pipe.visualizer.expansion_line_plots_per_point(n_expansions=1,
                                                           n_points=2)

            assert path.exists("%s/%s%s Expansion" % ("%s/Line Plots Per Point" % temp_dir,
                                                      str(round_to_nearest_half(1 *
                                                                                config.pixel_interval *
                                                                                config.pixels_to_distance)),
                                                      config.data_resolution_units))

            pipe.visualizer.expansion_line_plots_all_points(n_expansions=1)

            assert path.exists("%s/%s%s Expansion" % ("%s/Line Plots All Points Average" % temp_dir,
                                                      str(round_to_nearest_half(1 *
                                                                                config.pixel_interval *
                                                                                config.pixels_to_distance)),
                                                      config.data_resolution_units))

    def test_obtain_expanded_vessel_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.obtain_expanded_vessel_masks(n_points=2)

            test_path = "%s/Expanded Vessel Masks/29.5 μm/Test/Original Mask Excluded/Point1.tif" % temp_dir

            assert path.exists(test_path)

    def test_obtain_embedded_vessel_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.obtain_embedded_vessel_masks(n_points=2)

            test_path = "%s/Embedded Vessel Masks/29.5 μm/Test/Original Mask Excluded/Point1.tif" % temp_dir

            assert path.exists(test_path)

    def test_pixel_expansion_ring_plots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.pixel_expansion_ring_plots(n_expansions=1,
                                                       expansions=[1])

            test_path = "%s/Ring Plots/%s%s Expansion/Point%s.png" % (temp_dir,
                                                                      str(round_to_nearest_half(
                                                                          1 *
                                                                          config.pixel_interval *
                                                                          config.pixels_to_distance)),
                                                                      config.data_resolution_units,
                                                                      str(1))

            assert path.exists(test_path)

    def test_expression_histogram(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.expression_histogram()

            test_path = "%s/Expression Histograms/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_categorical_violin_plot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.categorical_violin_plot(primary_categorical_analysis_variable="Vessel Size",
                                                    mask_type="mask_and_expansion")

            test_path = "%s/Categorical Violin Plots/Test/By Vessel Size/Vessels/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_categorical_violin_plot_with_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = ShapeQuantificationAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir,
                             mask_type="expansion_only",
                             marker_settings="all_markers",
                             shape_quantification_method={
                                 "Name": "Solidity",
                                 "Metric": solidity
                             },
                             img_shape=(2048, 2048))

            pipe.visualizer.categorical_violin_plot_with_images(primary_categorical_analysis_variable="Solidity",
                                                                mask_size=(2048, 2048),
                                                                order=["25%", "50%", "75%", "100%"])

            test_path = "%s/Categorical Violin Plots with Images/Test/By Solidity/Vessels/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_average_quartile_violin_plot_subplots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = ShapeQuantificationAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir,
                             mask_type="expansion_only",
                             marker_settings="all_markers",
                             shape_quantification_method={
                                 "Name": "Solidity",
                                 "Metric": solidity
                             },
                             img_shape=(2048, 2048))

            pipe.visualizer.average_quartile_violin_plot_subplots(primary_categorical_analysis_variable="Solidity",
                                                                  order=["25%", "50%", "75%", "100%"])

            test_path = "%s/Average Quartile Violin Plots/Test/average_quartile_violins.png" % temp_dir

            assert path.exists(test_path)

    def test_violin_plot_brain_expansion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.violin_plot_brain_expansion(n_expansions=1)

            test_path = "%s/Expansion Violin Plots/Per Marker/0.5μm Expansion/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_box_plot_brain_expansions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.box_plot_brain_expansions(n_expansions=1)

            test_path = "%s/Expansion Box Plots/Per Marker/0.5μm Expansion/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_spatial_probability_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.spatial_probability_maps(mask_size=(2048, 2048))

            test_path = "%s/Pixel Expression Spatial Maps/Test/Vessels/Point1.png" % temp_dir

            assert path.exists(test_path)

    def test_vessel_images_by_categorical_variable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.vessel_images_by_categorical_variable(primary_categorical_analysis_variable="Vessel Size")

            test_path = "%s/Vessel Size Vessel Images/Test/Large/Point_Num_1_Vessel_ID_5.png" % temp_dir

            assert path.exists(test_path)

    def test_scatter_plot_umap_marker_projection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            umap_analyzer = DimensionalityReductionClusteringAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            umap_analyzer.analyze(temp_dir)

            shape_quant_analyzer = ShapeQuantificationAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            shape_quant_analyzer.analyze(temp_dir,
                                         mask_type="expansion_only",
                                         marker_settings="all_markers",
                                         shape_quantification_method={
                                             "Name": "Solidity",
                                             "Metric": solidity
                                         },
                                         img_shape=(2048, 2048))

            pipe.visualizer.scatter_plot_umap_marker_projection(mask_type="mask_only",
                                                                primary_continuous_analysis_variable="Solidity Score")

            test_path = "%s/UMAP Scatter Plot Projection/Test/Marker Clusters/Vessels.png" % temp_dir

            assert path.exists(test_path)

    def test_continuous_scatter_plot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = ShapeQuantificationAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir,
                             mask_type="expansion_only",
                             marker_settings="all_markers",
                             shape_quantification_method={
                                 "Name": "Solidity",
                                 "Metric": solidity
                             },
                             img_shape=(2048, 2048))

            pipe.visualizer.continuous_scatter_plot()

            test_path = "%s/Solidity Score Scatter Plots/Test/By Solidity/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_vessel_nonvessel_heatmap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"]
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.vessel_nonvessel_heatmap(n_expansions=1,
                                                     primary_categorical_analysis_variable=None)

            test_path = "%s/Heatmaps & Clustermaps/Test/Heatmaps/Expansion_1.png" % temp_dir

            assert path.exists(test_path)

    def test_categorical_split_expansion_heatmap_clustermap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"]
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.categorical_split_expansion_heatmap_clustermap(n_expansions=1)

            test_path = "%s/Categorical Expansion Heatmaps & Clustermaps/Test/" \
                        "Categorical Split Expansion Heatmaps/Marker Clusters/0.5μm Expansion/Vessels.png" % temp_dir

            assert path.exists(test_path)

    def test_brain_region_expansion_heatmap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"]
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.brain_region_expansion_heatmap(n_expansions=1,
                                                           primary_categorical_analysis_variable=None)

            test_path = "%s/Expansion Heatmaps & Clustermaps/Test/Expansion Clustermaps/0.5μm Expansion/All_Points.png" % temp_dir

            assert path.exists(test_path)

    def test_marker_expression_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"]
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.marker_expression_masks()

            test_path = "%s/Expression Masks/Test/Point 1/SMA.png" % temp_dir

            assert path.exists(test_path)

    def test_pseudo_time_heatmap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"]
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            analyzer = ShapeQuantificationAnalyzer(
                pipe.config,
                pipe.all_expansions_features,
                pipe.marker_names,
                pipe.all_feeds_contour_data,
                pipe.all_feeds_metadata,
                pipe.all_feeds_data
            )

            analyzer.analyze(temp_dir,
                             mask_type="expansion_only",
                             marker_settings="all_markers",
                             shape_quantification_method={
                                 "Name": "Solidity",
                                 "Metric": solidity
                             },
                             img_shape=(2048, 2048))

            pipe.visualizer.pseudo_time_heatmap()

            test_path1 = "%s/Pseudo-Time Heatmaps/Test/Test Region/" \
                         "Solidity Score_pseudo_time_heatmap_Test Region.png" % temp_dir
            test_path2 = "%s/Pseudo-Time Heatmaps/Test/Test Region/" \
                         "Solidity Score_pseudo_time_heatmap_binned_Test Region.png" % temp_dir
            test_path3 = "%s/Pseudo-Time Heatmaps/Test/Solidity Score_pseudo_time_heatmap_all_points.png" % temp_dir

            assert path.exists(test_path1)
            assert path.exists(test_path2)
            assert path.exists(test_path3)

    def test_vessel_nonvessel_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir, n_points=2, resolution=(2048, 2048))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"]
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.vessel_nonvessel_masks()

            test_path = "%s/Associated Area Masks/Test/2.5μm Expansion/Point 1.png" % temp_dir

            assert path.exists(test_path)

    def test_removed_vessel_expression_boxplot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_test_data(temp_dir,
                             n_points=2,
                             resolution=(2048, 2048),
                             n_pseudo_vessel_size=(1, 50))

            config = Config()
            example_feed = MIBIDataFeed(
                feed_data_loc="%s/data" % temp_dir,
                feed_mask_loc="%s/masks" % temp_dir,
                feed_name="Test",
                n_points=2,
                brain_region_point_ranges=[(1, 2)],
                brain_region_names=["Test Region"],
                data_resolution_size=(5000, 5000)
            )

            pipe = MIBIPipeline(config, temp_dir,
                                csv_loc="data/dummy_test_data.csv",
                                max_inward_expansions=1,
                                max_outward_expansions=1,
                                expansions=[1],
                                n_workers=1,
                                run_async=False
                                )
            pipe.add_feed(example_feed)
            pipe.load_preprocess_data()

            pipe.visualizer.removed_vessel_expression_boxplot()

            test_path = "%s/Kept Vs. Removed Vessel Boxplots/Test/All Points/All_Points.png" % temp_dir

            assert path.exists(test_path)
