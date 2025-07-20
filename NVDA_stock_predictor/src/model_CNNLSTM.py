import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

cfg = OmegaConf.load("configs/model_CNNLSTM.yaml")

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
        input_size = cfg.LSTM.input_size
        hidden_size = cfg.LSTM.hidden_size
        num_layers = cfg.LSTM.num_layers
        bidirectional = cfg.LSTM.bidirectional
        dropout = cfg.LSTM.dropout
        output_size = cfg.LSTM.output_size
        batch_first = cfg.LSTM.batch_first

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
        cnn_out = self.cnn(x)
        lstm_in = cnn_out.permute(0, 2, 1) 
        output = self.lstm(lstm_in)         
        return output
