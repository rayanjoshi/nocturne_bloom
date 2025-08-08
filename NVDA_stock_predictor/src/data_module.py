import hydra
import torch
import joblib
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from omegaconf import DictConfig
from sklearn.preprocessing import RobustScaler

class SimpleTensorDataset(Dataset):
    """Simple dataset for tensor data used in Lightning dataloaders"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class StockDataset(Dataset):
    def __init__(self, cfg: DictConfig = None):
        # Initialize with empty arrays - will be populated by data_module method
        self.x = None
        self.y = None

    def generate_windows(self, dataFrame, cfg: DictConfig):
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
        
        print(f"Created {len(x)} windows with shape {x.shape}")
        print(f"Target range before scaling: [{y.min():.6f}, {y.max():.6f}]")
        
        # CRITICAL FIX: Split BEFORE scaling to avoid data leakage
        split_ratio = cfg.data_module.train_val_split
        split_idx = int(len(y) * split_ratio)
        
        x_train, x_val = x[:split_idx], x[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Train/Val split: {len(x_train)}/{len(x_val)} samples")
        print(f"Train targets range: [{y_train.min():.6f}, {y_train.max():.6f}]")
        print(f"Val targets range: [{y_val.min():.6f}, {y_val.max():.6f}]")
        
        # Scale features (fit on train only)
        print("Scaling features...")
        feature_scaler = RobustScaler(quantile_range=(10.0, 95.0))
        
        # Reshape for scaling: (samples * timesteps, features)
        x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
        x_train_scaled = feature_scaler.fit_transform(x_train_reshaped)
        x_train_scaled = x_train_scaled.reshape(x_train.shape)
        
        x_val_reshaped = x_val.reshape(-1, x_val.shape[-1])
        x_val_scaled = feature_scaler.transform(x_val_reshaped)
        x_val_scaled = x_val_scaled.reshape(x_val.shape)
        
        # Scale targets (fit on train only)
        print("Scaling targets...")
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        print(f"Train targets after scaling: [{y_train_scaled.min():.6f}, {y_train_scaled.max():.6f}]")
        print(f"Val targets after scaling: [{y_val_scaled.min():.6f}, {y_val_scaled.max():.6f}]")
        
        # Combine back maintaining temporal order
        x_scaled = np.concatenate([x_train_scaled, x_val_scaled])
        y_scaled = np.concatenate([y_train_scaled, y_val_scaled])
        
        # Save scalers
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        
        feature_scaler_path = repo_root / "models/feature_scaler.pkl"
        target_scaler_path = repo_root / cfg.data_module.y_scaled_save_path.lstrip('../')
        
        feature_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        target_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        
        print(f"Feature scaler saved to: {feature_scaler_path.absolute()}")
        print(f"Target scaler saved to: {target_scaler_path.absolute()}")
        print(f"Scalers fitted on training data only: {len(y_train)} samples")
        
        # Save numpy arrays to sequence_processing directory 
        x_save_path = repo_root / cfg.data_processor.x_load_path.lstrip('../')
        y_save_path = repo_root / cfg.data_processor.y_load_path.lstrip('../')
        
        # Create directories if they don't exist
        x_save_path.parent.mkdir(parents=True, exist_ok=True)
        y_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove .npy extension since np.save adds it automatically
        x_save_path = x_save_path.with_suffix('')
        y_save_path = y_save_path.with_suffix('')

        np.save(x_save_path, x_scaled)
        np.save(y_save_path, y_scaled)
        print(f"Scaled data saved to: {x_save_path}.npy and {y_save_path}.npy")
        
        return x_scaled, y_scaled
    
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
        split_idx = int(len(self.x) * self.cfg.data_module.train_val_split)
        self.train_x, self.val_x = self.x[:split_idx], self.x[split_idx:]
        self.train_y, self.val_y = self.y[:split_idx], self.y[split_idx:]

    def train_dataloader(self):
        return DataLoader(
            SimpleTensorDataset(self.train_x, self.train_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            SimpleTensorDataset(self.val_x, self.val_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        # Use validation set for testing
        return DataLoader(
            SimpleTensorDataset(self.val_x, self.val_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )


@hydra.main(version_base=None, config_path="../configs", config_name="data_module")
def main(cfg: DictConfig):
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Create the dataset (no files needed yet)
    dataset = StockDataset()
    
    # Read CSV and do feature engineering
    preprocessing_data_path = repo_root / cfg.data_processor.preprocessing_data_path
    print(f"Reading processed data from: {preprocessing_data_path.absolute()}")
    dataFrame = pd.read_csv(preprocessing_data_path, header=0, index_col=0, parse_dates=True)
    
    # Feature engineering and save as numpy
    x, y = dataset.generate_windows(dataFrame, cfg)
    print("---------  Data Processing Statistics ---------")
    print(f"Lookback window shape: {x.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Update dataset with the data and convert to tensors
    dataset.x = x
    dataset.y = y
    dataset.save_tensors(cfg)
    print("--------- Data Processing Completed ---------")
    
    # Now set up the Lightning DataModule
    data_module = StockDataModule(cfg)
    data_module.setup()
    
    print("--------- Data Module Statistics ---------")
    print(f"Train set size: {len(data_module.train_x)}")
    print(f"Validation set size: {len(data_module.val_x)}")
    print("--------- Data Module Completed ---------")

    return data_module
if __name__ == "__main__":
    main()