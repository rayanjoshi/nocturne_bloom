import numpy as np
import pandas as pd
import torch
import hydra
from torch.utils.data import Dataset
from pathlib import Path
from omegaconf import DictConfig


class StockDataset(Dataset):
    def __init__(self, cfg: DictConfig):
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor

        x_path = repo_root / cfg.data_processor.x_load_path.lstrip('../')
        y_path = repo_root / cfg.data_processor.y_load_path.lstrip('../')

        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def data_module(self, dataFrame, cfg: DictConfig):
        window_size = cfg.data_module.window_size
        target_col = cfg.data_module.target_col
        
        features = dataFrame.drop(columns=[target_col])
        target = dataFrame[target_col]

        x, y = [], []
        for i in range(window_size, len(dataFrame)):
            x.append(features.iloc[i-window_size:i].values)  # past window_size days features
            y.append(target.iloc[i])                         # target is the value at day i

        x = np.array(x)
        y = np.array(y)
        
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor

        x_save_path = repo_root / cfg.data_module.x_save_path.lstrip('../')
        y_save_path = repo_root / cfg.data_module.y_save_path.lstrip('../')

        np.save(x_save_path, x)
        np.save(y_save_path, y)
        return x, y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
    
    def save_tensors(self, cfg: DictConfig):
        """Save all tensors to files - call this once after creating the dataset"""
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor

        x_save_path = repo_root / cfg.data_processor.x_save_path.lstrip('../')
        y_save_path = repo_root / cfg.data_processor.y_save_path.lstrip('../')

        # Create directories if they don't exist
        x_save_path.parent.mkdir(parents=True, exist_ok=True)
        y_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to tensors and save
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        
        torch.save(x_tensor, x_save_path)
        torch.save(y_tensor, y_save_path)
        
        print(f"Tensors saved to:")
        print(f"  X: {x_save_path.absolute()}")
        print(f"  Y: {y_save_path.absolute()}")
        print(f"X tensor shape: {x_tensor.shape}")
        print(f"Y tensor shape: {y_tensor.shape}")


@hydra.main(version_base=None, config_path="../configs", config_name="data_processor")
def main(cfg: DictConfig):
    try:
        dataset = StockDataset(cfg)
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        preprocessing_data_path = repo_root / cfg.data_processor.preprocessing_data_path.lstrip('../')

        print(f"Reading processed data from: {preprocessing_data_path.absolute()}")
        dataFrame = pd.read_csv(preprocessing_data_path, header=0, index_col=0, parse_dates=True)

        x,y = dataset.data_module(dataFrame, cfg)
        print("---------  Data Processing Statistics ---------")
        print(f"Lookback window shape: {x.shape}")
        print(f"Target vector shape: {y.shape}")
        dataset.save_tensors(cfg)
        print("--------- Data Processing Completed ---------")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()