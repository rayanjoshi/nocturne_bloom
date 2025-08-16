import hydra
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path

from src.model_Ensemble import EnsembleModule
from src.data_module import StockDataModule
from scripts.logging_config import get_logger, setup_logging

setup_logging(log_level="INFO", console_output=True, file_output=True)
logger = get_logger("trainer")

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    logger.info("Starting training script with config: %s", cfg.trainer)

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
    logger.info("Early stopping: patience=%d, min_delta=%.4f", early_stopping_patience, early_stopping_delta)
    
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