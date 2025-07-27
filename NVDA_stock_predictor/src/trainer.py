from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import hydra
import torch
import sys
import os
from omegaconf import DictConfig, OmegaConf


checkpoint_callback = ModelCheckpoint(
    monitor="val_r2",
    dirpath="../models",
    filename="model-{epoch:02d}-{val_r2:.2f}",
    save_top_k=3,
    mode="max",
)

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(
        project=cfg.trainer.project_name,
        name=cfg.trainer.run_name,
        save_dir="../logs"
    )
    
    # Configure early stopping from config
    early_stopping_patience = getattr(cfg.trainer, 'early_stopping_patience', 10)
    early_stopping_delta = getattr(cfg.trainer, 'early_stopping_delta', 0.005)
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        min_delta=early_stopping_delta,
    )
    
    # Create the Trainer instance
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
    
    
    from model_Ensemble import EnsembleModule
    from data_module import StockDataModule

    model = EnsembleModule(cfg)
    data_module = StockDataModule(cfg)

    torch.manual_seed(cfg.trainer.seed)
    trainer.fit(model, datamodule=data_module)
    
if __name__ == "__main__":
    main()