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
        CNN_channels = [cfg.CNNEncoder.CNN_channels[0], cfg.CNNEncoder.CNN_channels[1]]
        self.CNN = nn.Sequential(
            nn.Conv1d(input_channels, CNN_channels[0], kernel_size=cfg.CNNEncoder.kernel_size[0], padding=cfg.CNNEncoder.padding[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.CNNEncoder.kernel_size[1], stride=cfg.CNNEncoder.stride),
            
            nn.Conv1d(CNN_channels[0], CNN_channels[1], kernel_size=cfg.CNNEncoder.kernel_size[0], padding=cfg.CNNEncoder.padding[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.CNNEncoder.kernel_size[1], stride=cfg.CNNEncoder.stride)
        )
    def forward(self, x):
        x = self.CNN(x)
        return x

class LSTMPredictor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        input_size = cfg.LSTMPredictor.input_size
        hidden_size = cfg.LSTMPredictor.hidden_size
        num_layers = cfg.LSTMPredictor.num_layers
        bidirectional = cfg.LSTMPredictor.bidirectional
        dropout = cfg.LSTMPredictor.dropout
        output_size = cfg.LSTMPredictor.output_size
        batch_first = cfg.LSTMPredictor.batch_first
        
        self.lstm = nn.LSTM (
            input_size, 
            hidden_size, 
            num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout if num_layers > 1 else 0.0, 
            batch_first=batch_first)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x

class CNNLSTMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cnn = CNNFeatureExtractor(cfg)
        self.lstm = LSTMPredictor(cfg)

    def forward(self, x):
        # Input shape: [batch_size, seq_len, features] = [batch_size, 10, 40]
        # For CNN: transpose to [batch_size, features, seq_len] = [batch_size, 40, 10]
        
        x = x.permute(0, 2, 1)  # [batch_size, 40, 10] - treat features as input channels
        
        cnn_out = self.cnn(x)  # [batch_size, 64, reduced_seq_len]
        
        # Prepare for LSTM: [batch_size, seq_len, features]
        lstm_in = cnn_out.permute(0, 2, 1)  # [batch_size, reduced_seq_len, 64]
        
        output = self.lstm(lstm_in)         
        return output

class CNNLSTMModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Pass the entire config since CNNLSTMModel expects CNNEncoder and LSTMPredictor sections
        self.model = CNNLSTMModel(cfg)
        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError()
        self.r2 = R2Score()
    
    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y)
        
        self.mae.update(y_hat, y)
        self.rmse.update(y_hat, y)
        self.r2.update(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        avg_mae = self.mae.compute()
        avg_rmse = torch.sqrt(self.rmse.compute())
        avg_r2 = self.r2.compute()
        
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
        self.mae.reset()
        self.rmse.reset()
        self.r2.reset()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)  # Remove the last dimension if needed
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer