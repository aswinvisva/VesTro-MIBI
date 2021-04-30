import tempfile
import unittest
from os import path

from config.config_settings import Config
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.mibi_pipeline import MIBIPipeline
from src.utils.test_utils import create_test_data
from src.utils.utils_functions import round_to_nearest_half


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

            pipe.visualizer.categorical_violin_plot(analysis_variable="Vessel Size",
                                                    mask_type="mask_and_expansion")

            test_path = "%s/Categorical Violin Plots/Test/By Vessel Size/Vessels/SMA.png" % temp_dir

            assert path.exists(test_path)
