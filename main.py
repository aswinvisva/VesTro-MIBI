from config.config_settings import Config
from src.data_analysis.dimensionality_reduction_clustering import DimensionalityReductionClusteringAnalyzer
from src.data_analysis.positive_vessel_summary_analyzer import PositiveVesselSummaryAnalyzer
from src.data_analysis.vessel_asymmetry_analyzer import VesselAsymmetryAnalyzer
from src.data_loading.mibi_data_feed import MIBIDataFeed
from src.data_loading.mibi_loader import MIBILoader
from src.mibi_pipeline import MIBIPipeline


def medres_example():
    conf = Config()

    medres_feed = MIBIDataFeed(
        feed_data_loc="/media/large_storage/oliveria_data/data/medres",
        feed_mask_loc="/media/large_storage/oliveria_data/masks/medres",
        feed_name="Medres",
        n_points=300,
        points_per_dir=100
    )

    loader = MIBILoader(conf)
    loader.add_feed(medres_feed)
    loader.read()


def hires_example():
    conf = Config()

    hires_feed = MIBIDataFeed(
        feed_data_loc="/media/aswin/large_storage/oliveria_data/data/hires",
        feed_mask_loc="/media/aswin/large_storage/oliveria_data/masks/hires",
        feed_name="Hires",
        n_points=48
    )

    pipe = MIBIPipeline(conf, csv_loc="results/5um_impansion_5um_expansion.csv")
    pipe.add_feed(hires_feed)
    pipe.load_preprocess_data()

    pipe.add_analyzer(VesselAsymmetryAnalyzer)
    pipe.add_analyzer(PositiveVesselSummaryAnalyzer)
    # pipe.add_analyzer(DimensionalityReductionClusteringAnalyzer)

    pipe.analyze_data(mask_type="expansion_only", marker_settings="all_markers", analysis_variable="Roughness")

    pipe.save_to_csv()

    pipe.generate_visualizations(mask_type="mask_only",
                                 analysis_variable="Roughness")


if __name__ == '__main__':
    hires_example()
