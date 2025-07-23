from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="../logs",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(
        project=cfg.trainer.project_name,
        name=cfg.trainer.run_name,
        save_dir="../logs"
    )
    
    # Create the Trainer instance
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    
    
    from model_CNNLSTM import CNNLSTMModule
    from data_module import StockDataModule
    
    model = CNNLSTMModule(cfg)
    data_module = StockDataModule(cfg)

    trainer.fit(model, datamodule=data_module)
    
if __name__ == "__main__":
    main()