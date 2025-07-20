import yfinance as yf
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

def load_data(cfg: DictConfig):
    print(f"Loading data for {cfg.data_loader.TICKER} from {cfg.data_loader.START_DATE} to {cfg.data_loader.END_DATE}")
    dataFrame = yf.download(cfg.data_loader.TICKER, start=cfg.data_loader.START_DATE, end=cfg.data_loader.END_DATE)
    
    if dataFrame.empty:
        raise ValueError(f"No data found for {cfg.data_loader.TICKER} in the specified date range.")

    # Convert relative path to absolute path within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
    repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
    output_path = repo_root / cfg.data_loader.raw_data_path.lstrip('../')  # Remove leading ../
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(output_path, index=True)
    
    print(f"Data saved to {output_path.absolute()}")
    print("--------- Data Loading Statistics ---------")
    print(f"Total features created: {len(dataFrame.columns)}")
    print(f"Dataset shape: {dataFrame.shape}")
    print(f"All features: {list(dataFrame.columns)}")
    print("--------- Data Loading Completed ---------")

    return dataFrame

@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: DictConfig):
    try:
        load_data(cfg)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
