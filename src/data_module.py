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
from numpy.lib.stride_tricks import sliding_window_view
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

class SimpleTensorDataset(Dataset):
    """Simple dataset for tensor data used in Lightning dataloaders"""
    def __init__(self, x, y, direction_y = None):
        self.x = x
        self.y = y
        self.direction_y = direction_y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.direction_y is not None:
            return self.x[idx], self.y[idx], self.direction_y[idx]
        else:
            return self.x[idx], self.y[idx]

class StockDataset(Dataset):
    def __init__(self, cfg: DictConfig = None):
        # Initialize with empty arrays - will be populated by data_module method
        self.x = None
        self.y = None
        self.direction_y = None
        
    def generate_windows(self, dataFrame, cfg: DictConfig):
        logger = log_function_start("StockDataset.generate_windows")
        window_size = cfg.data_module.window_size
        price_target_col = cfg.data_module.price_target_col
        direction_target_col = cfg.data_module.direction_target_col
        
        target_cols = [price_target_col, direction_target_col]
        features = dataFrame.drop(columns=[col for col in dataFrame.columns if col in target_cols])
        price_target = dataFrame[price_target_col]
        direction_target = dataFrame[direction_target_col]

        features_array = features.values
        price_target_array = price_target.values
        direction_target_array = direction_target.values

        # Create sliding window for features using list comprehension
        x = sliding_window_view(
            features_array, 
            window_shape=(window_size, features_array.shape[1])
        )[:-1].reshape(-1, window_size, features_array.shape[1])
        price_y = price_target_array[window_size:]
        direction_y = direction_target_array[window_size:]
        
        x = np.array(x)
        price_y = np.array(price_y)
        direction_y = np.array(direction_y)

        logger.info(f"Created {len(x)} windows with shape {x.shape}")
        logger.info(f"Price target range before scaling: [{price_y.min():.6f}, {price_y.max():.6f}]")
        logger.info(f"Direction target range before scaling: [{direction_y.min():.6f}, {direction_y.max():.6f}]")

        logger.info("Splitting into train/val sets... before scaling")
        split_ratio = cfg.data_module.train_val_split
        split_idx = int(len(price_y) * split_ratio)
        
        x_train, x_val = x[:split_idx], x[split_idx:]
        price_y_train, price_y_val = price_y[:split_idx], price_y[split_idx:]
        direction_y_train, direction_y_val = direction_y[:split_idx], direction_y[split_idx:]

        logger.info(f"Train/Val split: {len(x_train)}/{len(x_val)} samples")
        logger.info(f"Train price targets range: [{price_y_train.min():.6f}, {price_y_train.max():.6f}]")
        logger.info(f"Val price targets range: [{price_y_val.min():.6f}, {price_y_val.max():.6f}]")
        logger.info(f"Train direction distribution: {np.bincount(direction_y_train)}")
        logger.info(f"Val direction distribution: {np.bincount(direction_y_val)}")

        # Scale features (fit on train only)
        logger.info("Scaling features...")
        feature_scaler = RobustScaler(quantile_range=(10.0, 95.0))
        
        # Reshape for scaling: (samples * timesteps, features)
        x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
        x_train_scaled = feature_scaler.fit_transform(x_train_reshaped)
        x_train_scaled = x_train_scaled.reshape(x_train.shape)
        
        x_val_reshaped = x_val.reshape(-1, x_val.shape[-1])
        x_val_scaled = feature_scaler.transform(x_val_reshaped)
        x_val_scaled = x_val_scaled.reshape(x_val.shape)
        
        # Scale targets (fit on train only)
        logger.info("Scaling price targets...")
        price_target_scaler = RobustScaler()
        y_train_scaled = price_target_scaler.fit_transform(price_y_train.reshape(-1, 1)).flatten()
        y_val_scaled = price_target_scaler.transform(price_y_val.reshape(-1, 1)).flatten()

        logger.info(f"Train targets after scaling: [{y_train_scaled.min():.6f}, {y_train_scaled.max():.6f}]")
        logger.info(f"Val targets after scaling: [{y_val_scaled.min():.6f}, {y_val_scaled.max():.6f}]")
        logger.info("Direction targets not scaled.")
        
        # Combine back maintaining temporal order
        x_scaled = np.concatenate([x_train_scaled, x_val_scaled])
        price_y_scaled = np.concatenate([y_train_scaled, y_val_scaled])
        direction_y_final = np.concatenate([direction_y_train, direction_y_val])

        # Save scalers
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/

        feature_scaler_path = Path(repo_root / cfg.data_module.x_scaled_save_path).resolve()
        price_target_scaler_path = Path(repo_root / cfg.data_module.price_y_scaled_save_path).resolve()

        feature_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        price_target_scaler_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(price_target_scaler, price_target_scaler_path)

        logger.info(f"Feature scaler saved to: {feature_scaler_path.absolute()}")
        logger.info(f"Price target scaler saved to: {price_target_scaler_path.absolute()}")
        logger.info(f"Scalers fitted on training data only: {len(price_y_train)} samples")
        
        # Save numpy arrays to sequence_processing directory 
        x_save_path = repo_root / Path(cfg.data_processor.x_load_path).resolve()
        price_y_save_path = repo_root / Path(cfg.data_processor.price_y_load_path).resolve()
        direction_y_save_path = repo_root / Path(cfg.data_processor.direction_y_load_path).resolve()

        # Create directories if they don't exist
        x_save_path.parent.mkdir(parents=True, exist_ok=True)
        price_y_save_path.parent.mkdir(parents=True, exist_ok=True)
        direction_y_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove .npy extension since np.save adds it automatically
        x_save_path = x_save_path.with_suffix('')
        price_y_save_path = price_y_save_path.with_suffix('')
        direction_y_save_path = direction_y_save_path.with_suffix('')

        np.save(x_save_path, x_scaled)
        np.save(price_y_save_path, price_y_scaled)
        np.save(direction_y_save_path, direction_y_final)
        logger.info(f"Scaled data saved to: {x_save_path}.npy and {price_y_save_path}.npy and {direction_y_save_path}.npy")
        log_function_end("StockDataset.generate_windows", success=True)
        return x_scaled, price_y_scaled, direction_y_final

    def __len__(self):
        return len(self.price_y)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        price_y = torch.tensor(self.price_y[idx], dtype=torch.float32)
        direction_y = torch.tensor(self.direction_y[idx], dtype=torch.long)
        return x, price_y, direction_y

    def save_tensors(self, cfg: DictConfig):
        """Save all tensors to files - call this once after creating the dataset"""
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        
        logger = get_logger("StockDataset.save_tensors")

        x_save_path = Path(repo_root / cfg.data_processor.x_save_path).resolve()
        price_y_save_path = Path(repo_root / cfg.data_processor.price_y_save_path).resolve()
        direction_y_save_path = Path(repo_root / cfg.data_processor.direction_y_save_path).resolve()

        # Create directories if they don't exist
        x_save_path.parent.mkdir(parents=True, exist_ok=True)
        price_y_save_path.parent.mkdir(parents=True, exist_ok=True)
        direction_y_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to tensors and save
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        price_y_tensor = torch.tensor(self.price_y, dtype=torch.float32)
        direction_y_tensor = torch.tensor(self.direction_y, dtype=torch.long)

        torch.save(x_tensor, x_save_path)
        torch.save(price_y_tensor, price_y_save_path)
        torch.save(direction_y_tensor, direction_y_save_path)

        logger.info(f"Tensors saved to:")
        logger.info(f"  X: {x_save_path.absolute()}")
        logger.info(f"  Price Y: {price_y_save_path.absolute()}")
        logger.info(f"  Direction Y: {direction_y_save_path.absolute()}")
        logger.info(f"X tensor shape: {x_tensor.shape}")
        logger.info(f"Price Y tensor shape: {price_y_tensor.shape}")
        logger.info(f"Direction Y tensor shape: {direction_y_tensor.shape}")
        log_function_end("StockDataset.save_tensors", success=True)

class StockDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        logger = get_logger("StockDataModule.setup")
        # Load the tensor files you just created
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        
        x_path = Path(repo_root / self.cfg.data_module.x_save_path).resolve()
        price_y_path = Path(repo_root / self.cfg.data_module.price_y_save_path).resolve()
        direction_y_path = Path(repo_root / self.cfg.data_module.direction_y_save_path).resolve()

        logger.info(f"Loading tensors from:")
        logger.info(f"  X: {x_path.absolute()}")
        logger.info(f"  Price Y: {price_y_path.absolute()}")
        logger.info(f"  Direction Y: {direction_y_path.absolute()}")

        self.x = torch.load(x_path)
        self.price_y = torch.load(price_y_path)
        self.direction_y = torch.load(direction_y_path)

        # Split into train/validation sets
        split_idx = int(len(self.x) * self.cfg.data_module.train_val_split)
        self.train_x, self.val_x = self.x[:split_idx], self.x[split_idx:]
        self.train_price_y, self.val_price_y = self.price_y[:split_idx], self.price_y[split_idx:]
        self.train_direction_y, self.val_direction_y = self.direction_y[:split_idx], self.direction_y[split_idx:]
        log_function_end("StockDataModule.setup", success=True)
        
    def train_dataloader(self):
        return DataLoader(
            SimpleTensorDataset(self.train_x, self.train_price_y, self.train_direction_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            SimpleTensorDataset(self.val_x, self.val_price_y, self.val_direction_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )
        
    def test_dataloader(self):
        # Use validation set for testing
        return DataLoader(
            SimpleTensorDataset(self.val_x, self.val_price_y, self.val_direction_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )
        


@hydra.main(version_base=None, config_path="../configs", config_name="data_module")
def main(cfg: DictConfig):
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Create the dataset (no files needed yet)
    dataset = StockDataset()
    
    # Read CSV and do feature engineering
    preprocessing_data_path = Path(repo_root / cfg.data_processor.preprocessing_data_path).resolve()
    logger.info(f"Reading processed data from: {preprocessing_data_path.absolute()}")
    dataFrame = pd.read_csv(preprocessing_data_path, header=0, index_col=0, parse_dates=True)
    
    # Feature engineering and save as numpy
    x, price_y, direction_y = dataset.generate_windows(dataFrame, cfg)
    logger.info("---------  Data Processing Statistics ---------")
    logger.info(f"Lookback window shape: {x.shape}")
    logger.info(f"Target vector shape: {price_y.shape}")
    logger.info(f"Direction vector shape: {direction_y.shape}")

    # Update dataset with the data and convert to tensors
    dataset.x = x
    dataset.price_y = price_y
    dataset.direction_y = direction_y
    dataset.save_tensors(cfg)
    logger.info("--------- Data Processing Completed ---------")
    
    # Now set up the Lightning DataModule
    data_module = StockDataModule(cfg)
    data_module.setup()
    
    logger.info("--------- Data Module Statistics ---------")
    logger.info(f"Train set size: {len(data_module.train_x)}")
    logger.info(f"Validation set size: {len(data_module.val_x)}")
    logger.info("--------- Data Module Completed ---------")

    return data_module
if __name__ == "__main__":
    main()