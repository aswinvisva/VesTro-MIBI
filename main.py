from datetime import datetime

from config.config_settings import Config
from src.data_analysis._shape_quantification_metrics import *
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_analysis.positive_vessel_summary_analyzer import PositiveVesselSummaryAnalyzer
from src.data_analysis.shape_quantification_analyzer import ShapeQuantificationAnalyzer
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

    results_dir = "/Users/aswinvisva/Documents/oliveria_data/results/experiment_%s/" % datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
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
    # pipe.add_analyzer(PositiveVesselSummaryAnalyzer)
    # pipe.add_analyzer(DimensionalityReductionClusteringAnalyzer)

    shape_quantification_method = {
        "Name": "Circularity",
        "Metric": circularity
    }

    pipe.analyze_data(marker_settings="all_markers",
                      shape_quantification_method=shape_quantification_method)

    pipe.visualizer.categorical_spatial_probability_maps(primary_categorical_analysis_variable=shape_quantification_method["Name"])


if __name__ == '__main__':
    hires_example()
