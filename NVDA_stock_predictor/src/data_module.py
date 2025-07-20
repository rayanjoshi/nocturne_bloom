import hydra
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig


def data_module(dataFrame, cfg: DictConfig):
    window_size = cfg.data_module.window_size
    target_col = cfg.data_module.target_col
    
    features = dataFrame.drop(columns=[target_col])
    target = dataFrame[target_col]

    x, y = [], []
    for i in range(window_size, len(dataFrame)):
        x.append(features.iloc[i-window_size:i].values)  # past window_size days features
        y.append(target.iloc[i])                         # target is the value at day i

    return np.array(x), np.array(y)


@hydra.main(version_base=None, config_path="../configs", config_name="data_module")
def main(cfg: DictConfig):
    try:
        # Convert relative path to absolute path within the repository
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        processed_data_path = repo_root / cfg.feature_engineering.processed_data_path.lstrip('../')

        print(f"Reading processed data from: {processed_data_path.absolute()}")
        dataFrame = pd.read_csv(processed_data_path, header=0, skiprows=[1,2], index_col=0, parse_dates=True)

        x,y = data_module(dataFrame, cfg)
        print("--------- Data Module Statistics ---------")
        print(f"Lookback window shape: {x.shape}")
        print(f"Target vector shape: {y.shape}")
        print("--------- Data Module Completed ---------")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    
if __name__ == "__main__":
    main()