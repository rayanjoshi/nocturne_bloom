"""
Data preparation pipeline for stock price forecasting using PyTorch Lightning.

This module includes dataset and datamodule classes for loading, preprocessing, 
and managing stock market time series data. It supports:
    - Sliding window creation for temporal features.
    - Feature and target scaling using `RobustScaler`.
    - Dataset serialization into both NumPy and PyTorch tensor formats.
    - Modular integration with Hydra for configuration management.
    - Compatibility with PyTorch Lightning's `LightningDataModule`.

Main classes:
    - SimpleTensorDataset: Wraps tensors for basic (x, y, direction) modeling.
    - StockDataset: Handles preprocessing, scaling, and saving of time series data.
    - StockDataModule: Loads tensors, splits data, and returns PyTorch DataLoaders.
"""
from pathlib import Path
from typing import Optional
import hydra
import torch
import joblib
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from sklearn.preprocessing import RobustScaler
from numpy.lib.stride_tricks import sliding_window_view
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

class SimpleTensorDataset(Dataset):
    """
    A simple dataset for PyTorch Lightning that supports optional directional labels.
    
    This dataset is designed to wrap input (`x`), target (`y`), and optionally 
    direction labels (`direction_y`) as PyTorch tensors for use with DataLoaders.
    
    Attributes:
        x (Tensor): Input features.
        y (Tensor): Regression or classification targets.
        direction_y (Tensor, optional): Optional directional classification labels.
    """

    def __init__(self, x, y, direction_y=None):
        """
        Initialize the dataset with features, targets, and optional direction labels.
        
        Args:
            x (Tensor): Input features.
            y (Tensor): Regression or classification targets.
            direction_y (Tensor, optional): Directional classification targets. Defaults to None.
        """
        self.x = x
        self.y = y
        self.direction_y = direction_y

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (x, y) if direction_y is None, otherwise (x, y, direction_y).
        """
        if self.direction_y is not None:
            return self.x[idx], self.y[idx], self.direction_y[idx]
        return self.x[idx], self.y[idx]

class StockDataset(Dataset):
    """
    PyTorch Dataset for stock time series data with sliding window preprocessing.
    
    This dataset supports:
        - Windowed feature extraction.
        - Scaling of features and price targets.
        - Saving of processed numpy arrays and PyTorch tensors.
    
    Attributes:
        x (np.ndarray): Feature sequences.
        y (np.ndarray): Scaled price targets.
        direction_y (np.ndarray): Directional targets (classification).
    """
    def __init__(self, cfg: DictConfig = None):
        """
        Initialize an empty dataset. Must call generate_windows before usage.
        
        Args:
            cfg (DictConfig, optional): Configuration object. Defaults to None.
        """
        self.cfg = cfg
        # Initialize with empty arrays - will be populated by data_module method
        self.x = None
        self.y = None
        self.price_y = None
        self.direction_y = None

    def generate_windows(self, df, cfg: DictConfig):
        """
        Create windowed sequences from time series data and scale features/targets.
        
        This method:
            - Splits data into train/validation sets.
            - Applies sliding window to features.
            - Scales features and price targets.
            - Saves scalers and processed arrays to disk.
        
        Args:
            df (pd.df): Input data with feature and target columns.
            cfg (DictConfig): Configuration object with paths and parameters.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                Scaled features (x), price targets (y), and direction targets.
        """
        logger = log_function_start("StockDataset.generate_windows")
        window_size = cfg.data_module.window_size
        price_target_col = cfg.data_module.price_target_col
        direction_target_col = cfg.data_module.direction_target_col

        target_cols = [price_target_col, direction_target_col]
        features = df.drop(columns=[col for col in df.columns if col in target_cols])
        price_target = df[price_target_col]
        direction_target = df[direction_target_col]

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
        price_min = price_y.min()
        price_max = price_y.max()
        logger.info(
            f"Price target range before scaling: [{price_min:.6f}, {price_max:.6f}]"
        )
        direction_min = direction_y.min()
        direction_max = direction_y.max()
        logger.info(
            "Direction target range before scaling: "
            f"[{direction_min:.6f}, {direction_max:.6f}]"
        )

        logger.info("Splitting into train/val sets... before scaling")
        split_ratio = cfg.data_module.train_val_split
        split_idx = int(len(price_y) * split_ratio)

        x_train, x_val = x[:split_idx], x[split_idx:]
        price_y_train, price_y_val = price_y[:split_idx], price_y[split_idx:]
        direction_y_train, direction_y_val = direction_y[:split_idx], direction_y[split_idx:]

        logger.info(f"Train/Val split: {len(x_train)}/{len(x_val)} samples")
        train_min = price_y_train.min()
        train_max = price_y_train.max()
        logger.info(f"Train price targets range: [{train_min:.6f}, {train_max:.6f}]")
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

        min_train = y_train_scaled.min()
        max_train = y_train_scaled.max()
        logger.info(
            "Train targets after scaling: "
            f"[{min_train:.6f}, {max_train:.6f}]"
        )
        min_val = y_val_scaled.min()
        max_val = y_val_scaled.max()
        logger.info(
            "Val targets after scaling: "
            f"[{min_val:.6f}, {max_val:.6f}]"
        )
        logger.info("Direction targets not scaled.")

        # Combine back maintaining temporal order
        x_scaled = np.concatenate([x_train_scaled, x_val_scaled])
        price_y_scaled = np.concatenate([y_train_scaled, y_val_scaled])
        direction_y_final = np.concatenate([direction_y_train, direction_y_val])

        # Save scalers
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/

        feature_scaler_path = Path(repo_root / cfg.data_module.x_scaled_save_path).resolve()
        price_target_scaler_path = Path(
            repo_root / cfg.data_module.price_y_scaled_save_path
        ).resolve()

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
        logger.info(f"Scaled X saved to: {x_save_path}.npy")
        logger.info(f"Scaled price_y saved to: {price_y_save_path}.npy")
        logger.info(f"Scaled direction_y saved to: {direction_y_save_path}.npy")
        log_function_end("StockDataset.generate_windows", success=True)
        return x_scaled, price_y_scaled, direction_y_final

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.price_y)

    def __getitem__(self, idx):
        """
        Retrieve one sample by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: x, price_y, and direction_y as tensors.
        """
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        price_y = torch.tensor(self.price_y[idx], dtype=torch.float32)
        direction_y = torch.tensor(self.direction_y[idx], dtype=torch.long)
        return x, price_y, direction_y

    def save_tensors(self, cfg: DictConfig):
        """
        Save the dataset tensors (x, price_y, direction_y) to disk.
        
        Args:
            cfg (DictConfig): Configuration object with save paths.
        """
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

        logger.info("Tensors saved to:")
        logger.info(f"  X: {x_save_path.absolute()}")
        logger.info(f"  Price Y: {price_y_save_path.absolute()}")
        logger.info(f"  Direction Y: {direction_y_save_path.absolute()}")
        logger.info(f"X tensor shape: {x_tensor.shape}")
        logger.info(f"Price Y tensor shape: {price_y_tensor.shape}")
        logger.info(f"Direction Y tensor shape: {direction_y_tensor.shape}")
        log_function_end("StockDataset.save_tensors", success=True)

class StockDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preparing stock time series data.
    
    Loads preprocessed tensors (features, price targets, direction targets),
    splits them into training and validation sets, and provides dataloaders.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize the data module with configuration.
        
        Args:
            cfg (DictConfig): Configuration object with paths and data settings.
        """
        super().__init__()
        self.cfg = cfg
        self.x = None
        self.price_y = None
        self.direction_y = None
        self.train_x = None
        self.val_x = None
        self.train_price_y = None
        self.val_price_y = None
        self.train_direction_y = None
        self.val_direction_y = None

    def setup(self, stage=None):
        """
        Setup the data module. Loads tensors and splits into train/val sets.
        
        Args:
            stage (str, optional): Stage to set up (fit/test/predict). Defaults to None.
        """
        logger = get_logger("StockDataModule.setup")
        # Load the tensor files you just created
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/

        x_path = Path(repo_root / self.cfg.data_module.x_save_path).resolve()
        price_y_path = Path(repo_root / self.cfg.data_module.price_y_save_path).resolve()
        direction_y_path = Path(repo_root / self.cfg.data_module.direction_y_save_path).resolve()

        logger.info("Loading tensors from:")
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
        self.train_direction_y = self.direction_y[:split_idx]
        self.val_direction_y = self.direction_y[split_idx:]
        log_function_end("StockDataModule.setup", success=True)

    def train_dataloader(self):
        """
        Returns the training DataLoader.
        
        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            SimpleTensorDataset(self.train_x, self.train_price_y, self.train_direction_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader.
        
        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(
            SimpleTensorDataset(self.val_x, self.val_price_y, self.val_direction_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader (uses validation split).
        
        Returns:
            DataLoader: Test dataloader.
        """
        # Use validation set for testing
        return DataLoader(
            SimpleTensorDataset(self.val_x, self.val_price_y, self.val_direction_y),
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            persistent_workers=True
        )



@hydra.main(version_base=None, config_path="../configs", config_name="data_module")
def main(cfg: Optional[DictConfig] = None):
    """
    Main entry point for preprocessing stock data and preparing the data module.
    
    This function performs the following steps:
    1. Sets up logging.
    2. Creates an empty StockDataset instance.
    3. Reads preprocessed CSV data from disk.
    4. Applies feature engineering to create input and target tensors.
    5. Saves the processed data as tensors.
    6. Initializes and sets up a PyTorch Lightning DataModule for training.
    
    Args:
        cfg (DictConfig): Configuration object loaded via OmegaConf containing 
            paths and parameters for data processing and module setup.
    
    Returns:
        StockDataModule: An initialized and setup Lightning DataModule ready for training.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Create the dataset (no files needed yet)
    dataset = StockDataset()

    # Read CSV and do feature engineering
    preprocessing_data_path = Path(repo_root / cfg.data_processor.preprocessing_data_path).resolve()
    logger.info(f"Reading processed data from: {preprocessing_data_path.absolute()}")
    df = pd.read_csv(preprocessing_data_path, header=0, index_col=0, parse_dates=True)

    # Feature engineering and save as numpy
    x, price_y, direction_y = dataset.generate_windows(df, cfg)
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
