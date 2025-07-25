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
            nn.Linear(cnnChannels[2], cnnChannels[1]),
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
        if x.dim() > 1 and x.size(-1) == 1:
            x = x.squeeze(-1)
        return x

class magnitudeEnsemble(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = CNN(cfg)
        self.ridge = Ridge(cfg.ridge.alpha)
        self.ridgeFitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        cnnPredictions = self.cnn(x)
        if self.ridgeFitted:
            xFlat = x.view(x.size(0), -1)
            if xFlat.is_cuda:
                xFlat_CPU = xFlat.detach().cpu().numpy()
            else:
                xFlat_CPU = xFlat.detach().numpy()
            ridgePredictions = torch.tensor(self.ridge.predict(xFlat_CPU), dtype=torch.float32, device=self.device)
            finalPredictions = self.cfg.model.cnnWeight * cnnPredictions + self.cfg.model.ridgeWeight * ridgePredictions
            return finalPredictions
        else:
            return cnnPredictions
        
    def fitRidge(self, x, y):
        xFlat = x.view(x.size(0), -1)
        if xFlat.is_cuda:
            xFlat_CPU = xFlat.detach().cpu().numpy()
        else:
            xFlat_CPU = xFlat.detach().numpy()
        if y.is_cuda:
            yCPU = y.detach().cpu().numpy()
        else:
            yCPU = y.detach().numpy()
        self.ridge.fit(xFlat_CPU, yCPU)
        self.ridgeFitted = True

def directionalFeatures(xTensor):
    batchSize, numFeatures, sequenceLength = xTensor.shape
    features = []
    if xTensor.is_cuda:
        xTensor_np = xTensor.detach().cpu().numpy()
    else:
        xTensor_np = xTensor.detach().numpy()
    currentFeatures = xTensor_np[:, -1,:]
    features.append(currentFeatures)
    
    if sequenceLength >= 2:
        shortMomentum = (xTensor_np[:, -1, :] - xTensor_np[:, -2, :])
        features.append(shortMomentum)
    
    if sequenceLength >= 3:
        medMomentum = (xTensor_np[:, -1, :] - xTensor_np[:, -3, :])
        features.append(medMomentum)
    
    if sequenceLength >= 5:
        longMomentum = (xTensor_np[:, -1, :] - xTensor_np[:, -5, :])
        features.append(longMomentum)
    
    if sequenceLength >= 3:
        recentVolatility =  np.std(xTensor_np[:, -3:, :], axis=1)
        features.append(recentVolatility)
    
    if sequenceLength >= 5:
        mediumVolatility = np.std(xTensor_np[:, -5:, :], axis=1)
        features.append(mediumVolatility)

    fullVolatility = np.std(xTensor_np, axis=1)
    features.append(fullVolatility)
    
    skewnessFeature = []
    for i in range(batchSize):
        sampleData = xTensor_np[i, :, :]
        meanValue = np.mean(sampleData, axis=1, keepdims=True)
        stdDev = np.std(sampleData, axis=1, keepdims=True)
        stdDev = np.where(stdDev == 0, 1e-8, stdDev) # Avoid division by zero
        skewnessValue = np.mean(((sampleData - meanValue) / stdDev) ** 3, axis= 1)
    skewnessFeature.append(skewnessValue)
    features.append(np.array(skewnessFeature))
    
    trendFeatures =[]
    for i in range(batchSize):
        batchTrends = []
        for j in range(numFeatures):
            xValues = np.arange(sequenceLength)
            yValues = xTensor_np[i, j, :]
            if len(np.unique(yValues)) > 1:
                correlationMatrix = np.corrcoef(xValues, yValues)[0, 1]
                trend = correlationMatrix[0, 1] if not np.isnan(correlationMatrix[0, 1]) else 0.0
            else:
                trend = 0.0
                batchTrends.append(trend)
        trendFeatures.append(batchTrends)
    features.append(trendFeatures)

    topFeatures_idx = list(range(min(10, len(features))))
    for i in range(len(topFeatures_idx)):
        for j in range(i+1, min(i+4, len(topFeatures_idx))):
            idx1, idx2 = topFeatures_idx[i], topFeatures_idx[j]
            if features[idx1].shape == features[idx2].shape:
                interactionFeature = (features[idx1] * features[idx2])
                features.append(interactionFeature)

    if numFeatures > 4:
        ranges = currentFeatures[:, 1] - currentFeatures[:, 2]
        features.append(ranges.reshape(-1, 1))
        
        bodySize = np.abs(currentFeatures[:, 3] - currentFeatures[:, 0])
        features.append(bodySize.reshape(-1, 1))
    
    finalFeatures = np.concatenate(features, axis=1)
    return finalFeatures

def directionalClassifiers(self, cfg: DictConfig):
    self.directionalClassifiers = {
        'gradientBoosting': GradientBoostingClassifier(
            n_estimators = cfg.classifiers.numEstimators[0],
            learning_rate = cfg.classifiers.learningRate,
            max_depth = cfg.classifiers.maxDepth[0],
            subsample = cfg.classifiers.subSample,
            min_samples_split = cfg.classifiers.minSamplesSplit[0],
            min_samples_leaf = cfg.classifiers.minSamplesLeaf[0],
            random_state = cfg.classifiers.randomState
            ),
        'randomForest': RandomForestClassifier(
            n_estimators = cfg.classifiers.numEstimators[1],
            max_depth = cfg.classifiers.maxDepth[1],
            min_samples_split = cfg.classifiers.minSamplesSplit[1],
            min_samples_leaf = cfg.classifiers.minSamplesLeaf[1],
            max_features = cfg.classifiers.maxFeatures,
            class_weight = cfg.classifiers.classWeight,
            random_state = cfg.classifiers.randomState
            ),
        'logisticRegression': LogisticRegression(
            C = cfg.classifiers.C[0],
            solver = cfg.classifiers.solver,
            class_weight = cfg.classifiers.classWeight,
            max_iter = cfg.classifiers.maxIterations,
            random_state = cfg.classifiers.randomState
        ),
        'svm': SVC(
            C = cfg.classifiers.C[1],
            kernel = cfg.classifiers.kernel,
            probability = cfg.classifiers.probability,
            class_weight = cfg.classifiers.classWeight,
            random_state = cfg.classifiers.randomState
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