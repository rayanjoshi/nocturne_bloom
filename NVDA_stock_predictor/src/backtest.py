import wrds
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import joblib
import torch

from data_loader import load_data
from feature_engineering import feature_engineering


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
    
    def engineer_features(self):
        print(f"Engineering features for {self.cfg.data_loader.TICKER}")
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        raw_data_path = repo_root / self.cfg.data_loader.raw_data_path.lstrip('../')
        save_data_path = repo_root / self.cfg.features.preprocessing_data_path.lstrip('../')
        dataFrame = pd.read_csv(
            raw_data_path, 
            header=0, 
            index_col=0, 
            parse_dates=True
        )
        feature_engineering(dataFrame, self.cfg, save_data_path)
    
    def data_module(self, dataFrame):
        print(f"Converting features to tensors for {self.cfg.data_loader.TICKER}")
        window_size = self.cfg.data_module.window_size
        target_col = self.cfg.data_module.target_col
        
        features = dataFrame.drop(columns=[target_col])
        target = dataFrame[target_col]
        
        x, y = [], []
        for i in range(window_size, len(dataFrame)):
            x.append(features.iloc[i-window_size:i].values)  # past window_size days features
            y.append(target.iloc[i])                         # target is the value at day i
        
        x = np.array(x)
        y = np.array(y)
        
        print(f"Created {len(x)} windows with shape {x.shape}")
        print(f"Target range before scaling: [{y.min():.6f}, {y.max():.6f}]")
        
        print("Scaling features and targets...")
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        getFeatureScaler = repo_root / self.cfg.data_module.feature_scaler_path.lstrip('../')
        getTargetScaler = repo_root / self.cfg.data_module.target_scaler_path.lstrip('../')
        feature_scaler = joblib.load(getFeatureScaler)
        target_scaler = joblib.load(getTargetScaler)
        
        scaledX = feature_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        scaledY = target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        print(f"Feature range after scaling: [{scaledX.min():.6f}, {scaledX.max():.6f}]")
        print(f"Target range after scaling: [{scaledY.min():.6f}, {scaledY.max():.6f}]")
        
        scaledX = torch.tensor(scaledX, dtype=torch.float32)
        scaledY = torch.tensor(scaledY, dtype=torch.float32)
        
        torch.save(scaledX, repo_root / self.cfg.data_module.x_scaled_save_path.lstrip('../'))
        torch.save(scaledY, repo_root / self.cfg.data_module.y_scaled_save_path.lstrip('../'))
        return scaledX, scaledY



@hydra.main(version_base=None, config_path="../configs", config_name="backtest")
def main(cfg: DictConfig):
    try:
        data_processor = DataProcessor(cfg)
        data_processor.load_data()
        data_processor.engineer_features()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
