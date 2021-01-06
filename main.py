from config.config_settings import Config
from utils.mibi_pipeline import MIBIPipeline

if __name__ == '__main__':
    conf = Config()
    pipe = MIBIPipeline(conf)
    pipe.load_preprocess_data()
    pipe.generate_visualizations()
