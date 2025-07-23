import hydra
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from omegaconf import DictConfig

class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class StockDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        # Load the tensor files you just created
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        
        x_path = repo_root / self.cfg.data_module.x_save_path.lstrip('../')
        y_path = repo_root / self.cfg.data_module.y_save_path.lstrip('../')

        print(f"Loading tensors from:")
        print(f"  X: {x_path.absolute()}")
        print(f"  Y: {y_path.absolute()}")

        self.x = torch.load(x_path)
        self.y = torch.load(y_path)

        # Split into train/validation sets
        split_idx = int(len(self.x) * 0.8)
        self.train_x, self.val_x = self.x[:split_idx], self.x[split_idx:]
        self.train_y, self.val_y = self.y[:split_idx], self.y[split_idx:]

    def train_dataloader(self):
        return DataLoader(
            StockDataset(self.train_x, self.train_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=True,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            StockDataset(self.val_x, self.val_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )


@hydra.main(version_base=None, config_path="../configs", config_name="data_module")
def main(cfg: DictConfig):
    data_module = StockDataModule(cfg)
    data_module.setup()
    
    print("--------- Data Module Statistics ---------")
    print(f"Train set size: {len(data_module.train_x)}")
    print(f"Validation set size: {len(data_module.val_x)}")
    print("--------- Data Module Completed ---------")

    return data_module
if __name__ == "__main__":
    main()