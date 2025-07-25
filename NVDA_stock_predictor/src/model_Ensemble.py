import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.regression import R2Score
from torchvision import transforms
import lightning as L
from omegaconf import DictConfig
import numpy as np
class CNNFeatureExtractor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        input_channels = cfg.CNNEncoder.input_channels
        CNN_channels = [cfg.CNNEncoder.CNN_channels[0], cfg.CNNEncoder.CNN_channels[1], cfg.CNNEncoder.CNN_channels[2]]
        self.CNN = nn.Sequential(
            nn.Conv1d(input_channels, CNN_channels[0], kernel_size=cfg.CNNEncoder.kernel_size[0], padding=cfg.CNNEncoder.padding[0]),
            nn.BatchNorm1d(CNN_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.CNNEncoder.pool_size[0], stride=cfg.CNNEncoder.stride, padding=cfg.CNNEncoder.pool_padding[0]),
            nn.Dropout(cfg.CNNEncoder.dropout),
            
            nn.Conv1d(CNN_channels[0], CNN_channels[1], kernel_size=cfg.CNNEncoder.kernel_size[1], padding=cfg.CNNEncoder.padding[1]),
            nn.BatchNorm1d(CNN_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.CNNEncoder.pool_size[1], stride=cfg.CNNEncoder.stride, padding=cfg.CNNEncoder.pool_padding[1]),
            nn.Dropout(cfg.CNNEncoder.dropout),

            nn.Conv1d(CNN_channels[1], CNN_channels[2], kernel_size=cfg.CNNEncoder.kernel_size[2], padding=cfg.CNNEncoder.padding[2]),
            nn.BatchNorm1d(CNN_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.CNNEncoder.pool_size[2], stride=cfg.CNNEncoder.stride, padding=cfg.CNNEncoder.pool_padding[2]),
            nn.Dropout(cfg.CNNEncoder.dropout)
        )
    def forward(self, x):
        x = self.CNN(x)
        return x

""""
class ensembleModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Pass the entire config since CNNLSTMModel expects CNNEncoder and LSTMPredictor sections
        self.model = CNNLSTMModel(cfg)
        self.criterion = nn.HuberLoss(delta=cfg.model.huber_delta)  # Use Huber loss for regression tasks
        
        self.mae_train = MeanAbsoluteError()
        self.rmse_train = MeanSquaredError()
        self.r2_train = R2Score()
        self.mae_val = MeanAbsoluteError()
        self.rmse_val = MeanSquaredError()
        self.r2_val = R2Score()

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)
        loss = self.criterion(y_hat, y)

        self.mae_val.update(y_hat, y)
        self.rmse_val.update(y_hat, y)
        self.r2_val.update(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        avg_mae = self.mae_val.compute()
        avg_rmse = torch.sqrt(self.rmse_val.compute())
        avg_r2 = self.r2_val.compute()

        self.log('val_mae', avg_mae, prog_bar=True)
        self.log('val_rmse', avg_rmse, prog_bar=True)
        self.log('val_r2', avg_r2, prog_bar=True)
        print(f"\n \n")
        print(f"\n --------- Validation Results ---------")
        print(f"Loss: {avg_loss:.4f}")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"R2: {avg_r2:.4f}")
        print(f"--------- Validation Complete ---------\n")

        # Reset metrics for next epoch
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)  # Remove the last dimension if needed
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)  # Ensure y_hat has the same shape as y 
        
        self.mae_train.update(y_hat, y)
        self.rmse_train.update(y_hat, y)
        self.r2_train.update(y_hat, y)
        
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('train_loss', 0.0)
        avg_mae = self.mae_train.compute()
        avg_rmse = torch.sqrt(self.rmse_train.compute())
        avg_r2 = self.r2_train.compute()

        self.log('train_mae', avg_mae, prog_bar=True)
        self.log('train_rmse', avg_rmse, prog_bar=True)
        self.log('train_r2', avg_r2, prog_bar=True)
        
        print(f"\n \n")
        print(f"\n --------- Training Results ---------")
        print(f"Loss: {avg_loss:.4f}")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"R2: {avg_r2:.4f}")
        print(f"--------- Training Complete ---------\n")

        # Reset metrics for next epoch
        self.mae_train.reset()
        self.rmse_train.reset()
        self.r2_train.reset()

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }
            
"""