"""
Training script for stock price prediction using PyTorch Lightning.

This module orchestrates the training of an ensemble model for stock price
prediction. It uses Hydra for configuration management, PyTorch Lightning for
training, and Weights & Biases (Wandb) for logging. The script sets up a trainer
with model checkpointing and early stopping, loads data and model modules, and
executes the training process.

Dependencies:
    - os: For environment variable handling.
    - subprocess: For running shell commands.
    - dotenv: For loading environment variables from a .env file.
    - pathlib: For file path handling.
    - typing: For type hints in function signatures.
    - hydra: For configuration management.
    - torch: For PyTorch operations and random seed setting.
    - lightning.pytorch: For training, checkpointing, and early stopping.
    - omegaconf: For handling configuration as DictConfig.
    - src.model_ensemble: Custom module containing the EnsembleModule class.
    - src.data_module: Custom module containing the StockDataModule class.
    - scripts.logging_config: Custom module for logging utilities.
"""
from pathlib import Path
from typing import Optional
import os
import subprocess
from dotenv import load_dotenv
import hydra
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.model_ensemble import EnsembleModule
from src.data_module import StockDataModule
from scripts.logging_config import get_logger, setup_logging

setup_logging(log_level="INFO", console_output=True, file_output=True)
logger = get_logger("trainer")

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
COMMAND = f"wandb login {wandb_api_key}"

subprocess.run(COMMAND, shell=True, check=True)

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: Optional[DictConfig] = None):
    """
    Main training function for the stock price prediction model.

    Configures and runs the training process using PyTorch Lightning. Sets up
    logging with Weights & Biases, model checkpointing, and early stopping based
    on validation metrics. Initializes the ensemble model and data module, sets
    a random seed for reproducibility, and executes training.

    Args:
        cfg (Optional[DictConfig]): Hydra configuration object containing trainer,
            model, and data parameters. Defaults to None, loaded by Hydra.

    Returns:
        None
    """
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    logger.info(f"Starting training script with config: {cfg.trainer}")

    logger_save_path = Path(repo_root / "logs").resolve()
    wandb_logger = WandbLogger(
        project=cfg.trainer.project_name,
        name=cfg.trainer.run_name,
        save_dir=logger_save_path,
    )
    checkpoint_save_path = Path(repo_root / "models").resolve()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_price_mae",
        dirpath=checkpoint_save_path,
        filename="model-{epoch:02d}-{val_price_mae:.2f}",
        save_top_k=1,
        mode="min",
    )
    # Configure early stopping from config
    early_stopping_patience = getattr(cfg.trainer, 'early_stopping_patience', 10)
    early_stopping_delta = getattr(cfg.trainer, 'early_stopping_delta', 0.005)
    logger.info(
        f"Early stopping: patience={early_stopping_patience}, "
        f"min_delta={early_stopping_delta:.4f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        min_delta=early_stopping_delta,
    )

    # Create the Trainer instance
    logger.info("Creating PyTorch Lightning Trainer instance.")
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=1,  # Run validation every epoch
        val_check_interval=1.0,     # Run validation at the end of each epoch
        deterministic = True,
        gradient_clip_val = cfg.trainer.gradient_clip_val,
    )

    logger.info("Instantiating model and data module.")
    model = EnsembleModule(cfg)
    data_module = StockDataModule(cfg)

    logger.info("Setting random seed: %d", cfg.trainer.seed)
    torch.manual_seed(cfg.trainer.seed)
    logger.info("Starting model training...")
    trainer.fit(model, datamodule=data_module)
    logger.info("Training complete. Saving model components.")
    model.save_components()  # Save model components after training

if __name__ == "__main__":
    main()
