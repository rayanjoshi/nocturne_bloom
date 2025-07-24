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
        
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * (2 if bidirectional else 1)
            
        self.lstm = nn.LSTM (
            layer_input_size, 
            hidden_size, 
            num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout if num_layers > 1 else 0.0, 
            batch_first=batch_first)
        self.lstm_layers.append(self.lstm)
        
        self.layer_norms.append(nn.LayerNorm(hidden_size * (2 if bidirectional else 1)))
        self.dropouts.append(nn.Dropout(dropout if num_layers > 1 else 0.0))
        
        final_hidden_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(final_hidden_size, final_hidden_size // 2)
        self.fc2 = nn.Linear(final_hidden_size // 2, output_size)
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, x):
        for i, (lstm_layer, layer_norm, dropout) in enumerate(zip(self.lstm_layers, self.layer_norms, self.dropouts)):
            lstm_out, _ = lstm_layer(x)
            
            # Apply layer normalization and dropout (except for last layer)
            if i < len(self.lstm_layers) - 1:
                lstm_out = layer_norm(lstm_out)
                lstm_out = dropout(lstm_out)
                x = lstm_out
            else:
                # For the last layer, take only the final time step
                x = lstm_out[:, -1, :]
        
        # Final prediction layers
        x = torch.relu(self.fc1(x))
        x = self.dropout_final(x)
        x = self.fc2(x)
        
        return x

class CNNLSTMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cnn = CNNFeatureExtractor(cfg)
        self.lstm = LSTMPredictor(cfg)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(cfg.model.target_sequence_length)

    def forward(self, x):
        # Input shape: [batch_size, seq_len, features] = [batch_size, 10, 40]
        # For CNN: transpose to [batch_size, features, seq_len] = [batch_size, 40, 10]
        
        x = x.permute(0, 2, 1)  # [batch_size, 40, 10] - treat features as input channels
        
        cnn_out = self.cnn(x)  # [batch_size, 64, reduced_seq_len]
        cnn_out = self.adaptive_pool(cnn_out)  # [batch_size, 64, target_sequence_length]
        
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