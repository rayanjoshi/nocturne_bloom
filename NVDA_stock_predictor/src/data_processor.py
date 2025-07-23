import numpy as np
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
        print(f"  X shape: {x_tensor.shape}")
        print(f"  Y shape: {y_tensor.shape}")
        

@hydra.main(version_base=None, config_path="../configs", config_name="data_processor")
def main(cfg: DictConfig):
    try:
        print("---------  Data Processing Statistics ---------")
        dataset = StockDataset(cfg)
        dataset.save_tensors(cfg)
        print("--------- Data Processing Completed ---------")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()