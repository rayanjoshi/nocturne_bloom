import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.regression import R2Score
from torchvision import transforms
from sklearn.linear_model import Ridge, LogisticRegression, SVC, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightning as L
from omegaconf import DictConfig
import numpy as np
class CNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        inputChannels = cfg.cnn.inputChannels
        cnnChannels = [cfg.cnn.cnnChannels[0], cfg.cnn.cnnChannels[1], cfg.cnn.cnnChannels[2]]
        self.cnn = nn.Sequential(
            nn.Conv1d(inputChannels, cnnChannels[0], kernel_size=cfg.cnn.kernelSize[0], padding=cfg.cnn.padding[0]),
            nn.BatchNorm1d(cnnChannels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.cnn.poolSize[0], stride=cfg.cnn.stride, padding=cfg.cnn.poolPadding[0]),
            nn.Dropout(cfg.cnn.dropout[0]),
            
            nn.Conv1d(cnnChannels[0], cnnChannels[1], kernel_size=cfg.cnn.kernelSize[1], padding=cfg.cnn.padding[1]),
            nn.BatchNorm1d(cnnChannels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.cnn.poolSize[1], stride=cfg.cnn.stride, padding=cfg.cnn.poolPadding[1]),
            nn.Dropout(cfg.cnn.dropout[0]),
            
            nn.Conv1d(cnnChannels[1], cnnChannels[2], kernel_size=cfg.cnn.kernelSize[2], padding=cfg.cnn.padding[2]),
            nn.BatchNorm1d(cnnChannels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(kernel_size=1),
            nn.Dropout(cfg.cnn.dropout[0])
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(cnnChannels[0], cnnChannels[1]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[1]),
            nn.Linear(cnnChannels[1], cnnChannels[2]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[0]),
            nn.Linear(cnnChannels[2], cfg.cnn.outputSize)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.squeeze(-1)
        x = self.fc_layer(x)
        x = x.squeeze(-1)
        return x

class magnitudeEnsemble(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cnn = CNN(cfg)
        self.ridge = Ridge(cfg.ridge.alpha)
        self.ridgeFitted = False
        
    def forward(self, cfg, x):
        cnnPredictions = self.cnn(x)
        if self.ridgeFitted:
            x_flat = x.view(x.size(0), -1).detach().numpy()
            ridgePredictions = torch.tensor(self.ridge.predict(x_flat), dtype=torch.float32)
            
            finalPredictions = cfg.model.cnnWeight * cnnPredictions + cfg.model.ridgeWeight * ridgePredictions
            return finalPredictions
        else:
            return cnnPredictions
    def fitRidge(self, x, y):
        xFlat = x.view(x.size(0), -1).detach().numpy()
        self.ridge.fit(xFlat, y.detach().numpy())
        self.ridgeFitted = True

def directionalFeatures(self, cfg: DictConfig, xTensor):
    batchSize, numFeatures, sequenceLength = xTensor.shape
    features = []
    currentFeatures = xTensor[:, -1,:].numpy()
    features.append(currentFeatures)
    
    if sequenceLength >= 2:
        shortMomentum = (xTensor[:, -1, :] - xTensor[:, -2, :]).numpy()
        features.append(shortMomentum)
    
    if sequenceLength >= 3:
        medMomentum = (xTensor[:, -1, :] - xTensor[:, -3, :]).numpy()
        features.append(medMomentum)
    
    if sequenceLength >= 5:
        longMomentum = (xTensor[:, -1, :] - xTensor[:, -5, :]).numpy()
        features.append(longMomentum)
    
    if sequenceLength >= 3:
        recentVolatility = torch.std(xTensor[:, -3:, :], dim=1).numpy()
        features.append(recentVolatility)
    
    if sequenceLength >= 5:
        mediumVolatility = torch.std(xTensor[:, -5:, :], dim=1).numpy()
        features.append(mediumVolatility)
    
    fullVolatility = torch.std(xTensor, dim=1).numpy()
    features.append(fullVolatility)
    
    skewnessFeature = []
    for i in range(batchSize):
        sampleData = xTensor[i, :, :].numpy()
        meanValue = np.mean(sampleData, axis=1)
        stdDev = np.std(sampleData, axis=1)
        skewnessValue = torch.mean(((sampleData - meanValue) / stdDev) ** 3, dim=0)
    skewnessFeature.append(skewnessValue.numpy())
    features.append(np.array(skewnessFeature))
    
    trendFeature =[]
    for i in range(numFeatures):
        for j in range(batchSize):
            xValues = np.arange(sequenceLength)
            yValues = xTensor[j, :, i].numpy()
            if len(np.unique(yValues)) > 1:
                trend = np.corrcoef(xValues, yValues)[0, 1]
            else:
                trend = 0.0
                trendFeature.append(trend)
        trends.append(trendFeature)
    trends = np.array(trends).T
    features.append(trends)
    
    topFeatures_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for i in range(len(topFeatures_idx)):
        for j in range(i+1, min(i+4, len(topFeatures_idx))):
            idx1, idx2 = topFeatures_idx[i], topFeatures_idx[j]
            interactionFeature = (features[idx1] * features[idx2])
            features.append(interactionFeature.reshape(-1, 1))
    
    if numFeatures > 4:
        ranges = currentFeatures[:, 1] - currentFeatures[:, 2]
        features.append(ranges.reshape(-1, 1))
        
        bodySize = np.abs(currentFeatures[:, 3] - currentFeatures[:, 0])
        features.append(bodySize.reshape(-1, 1))
    
    finalFeatures = np.concatenate(features, axis=1)
    return finalFeatures

def directionalClassifiers(self, cfg: DictConfig):
    self.directinalClassifiers = {
        'gradientBoosting': GradientBoostingClassifier(
            numEstimators = cfg.classifiers.numEstimators[0],
            learningRate = cfg.classifiers.learningRate,
            maxDepth = cfg.classifiers.maxDepth[0],
            subsample = cfg.classifiers.subSample,
            minSamplesSplit = cfg.classifiers.minSamplesSplit[0],
            minSamplesLeaf = cfg.classifiers.minSamplesLeaf[0],
            randomState = cfg.classifiers.randomState
            ),
        'randomForest': RandomForestClassifier(
            numEstimators = cfg.classifiers.numEstimators[1],
            maxDepth = cfg.classifiers.maxDepth[1],
            minSamplesSplit = cfg.classifiers.minSamplesSplit[1],
            minSamplesLeaf = cfg.classifiers.minSamplesLeaf[1],
            maxFeatures = cfg.classifiers.maxFeatures,
            classWeight = cfg.classifiers.classWeight,
            randomState = cfg.classifiers.randomState
            ),
        'logisticRegression': LogisticRegression(
            C = cfg.classifiers.C[0],
            solver = cfg.classifiers.solver,
            classWeight = cfg.classifiers.classWeight,
            maxIter = cfg.classifiers.maxIterations,
            randomState = cfg.classifiers.randomState
        ),
        'svm': SVC(
            C = cfg.classifiers.C[1],
            kernel = cfg.classifiers.kernel,
            probability = cfg.classifiers.probability,
            classWeight = cfg.classifiers.classWeight,
            randomState = cfg.classifiers.randomState
        )
    }
    self.voting_classifier = VotingClassifier(
    estimators=list(self.directionalClassifiers.items()),
    voting='soft',
    weights=[0.3, 0.3, 0.2, 0.2]  # Optimized weights
    )
    
    self.featureScaler = StandardScaler()
    self.directionalThreshold = 0.5
    
"""
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