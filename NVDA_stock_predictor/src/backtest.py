import wrds
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

from data_loader import load_data

class DataProcessor:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def load_data(self):
        print(f"Loading data for {self.cfg.data_loader.TICKER} from {self.cfg.data_loader.START_DATE} to {self.cfg.data_loader.END_DATE}")
        load_data(
            self.cfg,
            self.cfg.data_loader.TICKER,
            self.cfg.data_loader.PERMNO,
            self.cfg.data_loader.GVKEY,
            self.cfg.data_loader.START_DATE,
            self.cfg.data_loader.END_DATE,
            self.cfg.data_loader.raw_data_path
        )


@hydra.main(version_base=None, config_path="../configs", config_name="backtest")
def main(cfg: DictConfig):
    try:
        data_processor = DataProcessor(cfg)
        data_processor.load_data()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
