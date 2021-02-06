from config.config_settings import Config
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
        feed_data_loc="/media/large_storage/oliveria_data/data/hires",
        feed_mask_loc="/media/large_storage/oliveria_data/masks/hires",
        feed_name="Hires",
        n_points=48
    )

    pipe = MIBIPipeline(conf)
    pipe.add_feed(hires_feed)

    pipe.load_preprocess_data()
    pipe.analyze_data()
    pipe.generate_visualizations()


if __name__ == '__main__':
    hires_example()
