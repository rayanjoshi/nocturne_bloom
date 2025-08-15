import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.regression import R2Score
import lightning as L
from omegaconf import DictConfig
from pathlib import Path
from scripts.logging_config import get_logger, setup_logging

setup_logging(log_level="INFO", console_output=True, file_output=True)
logger = get_logger("model_Ensemble")
class CNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing CNN model with config: %s", cfg.cnn)
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
            nn.AdaptiveAvgPool1d(1),
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
        logger.debug("CNN forward pass with input shape: %s", x.shape)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.squeeze(-1)
        x = self.fc_layer(x)
        if x.dim() > 1 and x.size(-1) == 1:
            x = x.squeeze(-1)
        return x

class RidgeRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing RidgeRegressor with alpha=%.4f, fit_intercept=%s", cfg.Ridge.alpha, cfg.Ridge.get('fit_intercept', True))
        super().__init__()
        self.alpha = cfg.Ridge.alpha
        self.fit_intercept = cfg.Ridge.get('fit_intercept', True)

        # Ridge parameters (will be set during fit)
        self.weight = nn.Parameter(torch.tensor([0,0]), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.is_fitted = False
        
    def fit(self, X, y):
        logger.info("Fitting RidgeRegressor on data: X shape %s, y shape %s", X.shape, y.shape)
        """
        Fit Ridge regression using pure PyTorch tensors
        
        Args:
            X: Input features (torch.Tensor) shape (batch_size, seq_len, features) or (batch_size, features)
            y: Target values (torch.Tensor) shape (batch_size,)
        """
        # Ensure tensors are on same device
        device = X.device
        y = y.to(device)
        
        # Flatten X to 2D if it's 3D: (batch_size, seq_len, features) -> (batch_size, seq_len * features)
        if len(X.shape) == 3:
            X = X.view(X.shape[0], -1)
        
        # Add intercept column if needed
        if self.fit_intercept:
            ones = torch.ones(X.shape[0], 1, device=device)
            XWithIntercept = torch.cat([X, ones], dim=1)
        else:
            XWithIntercept = X

        # Ridge regression solution: (X^T X + α I)^(-1) X^T y
        numFeatures = XWithIntercept.shape[1]
        
        # Create identity matrix for regularization
        I = torch.eye(numFeatures, device=device)
        
        # Ridge regression closed form solution
        XtX = torch.mm(XWithIntercept.t(), XWithIntercept)
        XtX_reg = XtX + self.alpha * I
        Xty = torch.mv(XWithIntercept.t(), y)

        # Solve the system
        try:
            # Try Cholesky decomposition first (faster and more stable for positive definite matrices)
            L = torch.linalg.cholesky(XtX_reg)
            theta = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            try:
                # Fallback to standard solve
                theta = torch.linalg.solve(XtX_reg, Xty)
            except RuntimeError:
                # Final fallback to SVD-based pseudo-inverse (most robust)
                print("Warning: Using pseudo-inverse due to singular matrix")
                theta = torch.linalg.pinv(XtX_reg) @ Xty
        
        # Split weight and bias
        if self.fit_intercept:
            self.weight.data = theta[:-1].clone()
            self.bias.data = theta[-1].clone()
        else:
            self.weight.data = theta.clone()
            self.bias.data = torch.tensor(0.0, device=device)
        # Register as parameters (no gradient needed)
        self.is_fitted = True
        
    def forward(self, x):
        logger.debug("RidgeRegressor forward pass with input shape: %s", x.shape)
        """
        Forward pass for Ridge regression using pure tensors
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features) or (batch_size, features)
            
        Returns:
            Predictions as torch.Tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Ridge model must be fitted before forward pass. Call .fit() first.")
        
        # Flatten to 2D if needed
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        
        # Ensure tensors are on same device
        x = x.to(self.weight.device)
        
        # Linear prediction: y = X @ w + b
        predictions = torch.mv(x, self.weight)
        if self.fit_intercept:
            predictions = predictions + self.bias
            
        return predictions
    
class ElasticNetRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing ElasticNetRegressor with alpha=%.4f, l1_ratio=%.4f, fit_intercept=%s",
                    cfg.ElasticNet.alpha, cfg.ElasticNet.l1_ratio, cfg.ElasticNet.get('fit_intercept', True))
        super().__init__()
        self.alpha = cfg.ElasticNet.alpha
        self.l1_ratio = cfg.ElasticNet.l1_ratio
        self.tol = cfg.ElasticNet.get('tol', 1e-4)
        self.max_iter = cfg.ElasticNet.get('max_iter', 1000)
        self.fit_intercept = cfg.ElasticNet.get('fit_intercept', True)
        self.eps = cfg.ElasticNet.get('eps', 1e-8)

        # ElasticNet parameters (will be set during fit)
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.is_fitted = False
        
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        
    def soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)
        
    def fit(self, X, y):
        """
        Fit ElasticNet regression using pure PyTorch tensors

        Args:
            X: Input features (torch.Tensor) shape (batch_size, seq_len, features) or (batch_size, features)
            y: Target values (torch.Tensor) shape (batch_size,)
        """
        # Ensure tensors are on same device
        device = X.device
        y = y.to(device)
        
        # Flatten X to 2D if it's 3D: (batch_size, seq_len, features) -> (batch_size, seq_len * features)
        if len(X.shape) == 3:
            X = X.view(X.shape[0], -1)
        
        numSamples, numFeatures = X.shape
        self.weight = nn.Parameter(torch.tensor(numFeatures), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.is_fitted = False
        
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0) + self.eps # add eps to prevent division by zero
        X_standardised = (X - X_mean) / X_std
        
        if self.fit_intercept:
            y_mean = y.mean()
            y_centred = y - y_mean
            self.y_mean = y_mean.clone()
        else:
            y_mean = 0.0
            y_centred = y
            self.y_mean = torch.tensor(0.0, device=device, dtype=X.dtype)
        
        self.X_mean = X_mean.clone()
        self.X_std = X_std.clone()
        beta = torch.zeros(numFeatures, device=device, dtype=X.dtype)
        
        l1_regularisation = self.alpha * self.l1_ratio
        l2_regularisation = self.alpha * (1 - self.l1_ratio)
        
        zero_var_mask = X_std < self.eps
        if zero_var_mask.any():
            logger.warning(f"Found {zero_var_mask.sum().item()} features with zero variance. Setting std to 1.0.")
            X_std = torch.where(zero_var_mask, torch.tensor(1.0, device=device, dtype=X.dtype), X_std)
        
        
        for interation in range(self.max_iter):
            beta_old = beta.clone()
            
            for j in range(numFeatures):
                if zero_var_mask[j]:
                    beta[j] = 0.0
                    continue
                residual = y_centred - X_standardised @ beta + X_standardised[:, j] * beta[j]
                rho_j = torch.dot(X_standardised[:, j], residual) / numSamples
                
                denominator = 1.0 + (l2_regularisation / numSamples)
                beta[j] = self.soft_threshold(rho_j, l1_regularisation / numSamples) / denominator
            
            if torch.norm(beta - beta_old) < self.tol:
                logger.info("Converged after %d iterations", interation + 1)
                break
        else:
            logger.warning("Iteration %d: Coefficients not converged yet", interation + 1)
        
        self.weight.data = (beta / X_std).clone()  # Scale back to original feature space
        
        if self.fit_intercept:
            intercept = y_mean - torch.dot(X_mean / X_std, beta)
            self.bias.data = intercept.clone()
        else:
            self.bias.data = torch.tensor(0.0, device=device)
        
        self.is_fitted = True
        logger.info("ElasticNetRegressor fitted successfully with weight shape %s and bias %s", self.weight.shape, self.bias.shape)
        
    def forward(self, x):
        logger.debug("ElasticNetRegressor forward pass with input shape: %s", x.shape)
        """
        Forward pass for ElasticNet regression using pure tensors

        Args:
            x: Input tensor of shape (batch_size, seq_len, features) or (batch_size, features)
            
        Returns:
            Predictions as torch.Tensor
        """
        if not self.is_fitted:
            raise RuntimeError("ElasticNet model must be fitted before forward pass. Call .fit() first.")

        # Flatten to 2D if needed
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        
        # Ensure tensors are on same device
        x = x.to(self.weight.device)
        
        # Linear prediction: y = X @ w + b
        predictions = torch.mv(x, self.weight)
        if self.fit_intercept:
            predictions = predictions + self.bias
            
        return predictions

class EnsembleModule(L.LightningModule):
    """
    CNN-ElasticNet Ensemble Lightning Module

    This combines CNN + ElasticNet in a proper Lightning module
    """
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing EnsembleModule with config: %s", cfg.model)
        super().__init__()
        self.cfg = cfg
        
        # CNN component
        self.cnn = CNN(cfg)

        # ElasticNet component as nn.Module
        self.elasticnet = ElasticNetRegressor(cfg)

        # Ensemble weights (optimized from your champion model)
        self.cnnWeight = cfg.model.cnnWeight
        self.elasticNetWeight = cfg.model.elasticNetWeight

        # Loss function
        self.criterion = nn.HuberLoss(delta=cfg.model.get('huber_delta', 1.0))
        
        # Metrics
        self.mae_train = MeanAbsoluteError()
        self.rmse_train = MeanSquaredError()
        self.r2_train = R2Score()
        self.mae_val = MeanAbsoluteError()
        self.rmse_val = MeanSquaredError()
        self.r2_val = R2Score()
        

        
        # Flags to track if models have been fitted
        self.elasticnet_fitted = False

    def on_train_start(self):
        logger.info("on_train_start: Fitting ElasticNet regressor on all training data...")
        """Fit ElasticNet regressor on all training data at the start of training"""
        if not self.elasticnet_fitted:
            print("Fitting ElasticNet regressor on all training data...")

            # Collect all training data
            allXValues = []
            allYValues = []
            
            # Get the dataloader
            train_dataloader = self.trainer.datamodule.train_dataloader()
            
            # Move model to correct device first
            device = next(self.parameters()).device
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_dataloader):
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    allXValues.append(x)
                    allYValues.append(y)

                    # Optional: Limit to first N batches to save memory
                    # if batch_idx >= 100:  # First 100 batches
                    #     break
            
            # Concatenate all batches
            XTrain = torch.cat(allXValues, dim=0)
            yTrain = torch.cat(allYValues, dim=0)

            # Fit ElasticNet
            if not self.elasticnet_fitted:
                self.fit_elasticnet_on_batch(XTrain, yTrain)
                print(f"ElasticNet regressor fitted on {len(yTrain)} samples!")

            # Clear memory
            del allXValues, allYValues, XTrain, yTrain

    def fit_elasticnet_on_batch(self, X, y):
        logger.info("Fitting ElasticNet on batch: X shape %s, y shape %s", X.shape, y.shape)
        """
        Fit the ElasticNet component on a batch of training data
        This can be called during training setup
        
        Args:
            X: Training features (torch.Tensor)
            y: Training targets (torch.Tensor)
        """
        self.elasticnet.fit(X, y)
        self.elasticnet_fitted = True


    def forward(self, x):
        logger.debug("Ensemble forward pass with input shape: %s", x.shape)
        """
        Forward pass combining CNN and ElasticNet predictions

        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Combined predictions
        """
        # CNN prediction
        cnnPrediction = self.cnn(x)

        # ElasticNet prediction (only if fitted)
        if self.elasticnet_fitted and self.elasticnet.is_fitted:
            try:
                elasticnetPrediction = self.elasticnet(x)

                # Ensure both predictions are on the same device and have same shape
                if elasticnetPrediction.device != cnnPrediction.device:
                    elasticnetPrediction = elasticnetPrediction.to(cnnPrediction.device)

                if elasticnetPrediction.shape != cnnPrediction.shape:
                    elasticnetPrediction = elasticnetPrediction.view_as(cnnPrediction)

                # Validate predictions before combining
                if torch.isnan(cnnPrediction).any() or torch.isnan(elasticnetPrediction).any():
                    print("Warning: NaN detected in predictions, using CNN only")
                    return cnnPrediction

                # Combine predictions with optimized weights
                finalPrediction = self.cnnWeight * cnnPrediction + self.elasticNetWeight * elasticnetPrediction

                # Clamp final predictions to prevent extreme values
                finalPrediction = torch.clamp(finalPrediction, min=-1e6, max=1e6)

                return finalPrediction
            except Exception as e:
                print(f"Warning: ElasticNet prediction failed: {e}, using CNN only")
                return cnnPrediction
        else:
            # If ElasticNet not fitted yet, return CNN prediction only
            return cnnPrediction

    def training_step(self, batch, batch_idx):
        logger.debug("Training step: batch_idx=%d", batch_idx)
        x, y = batch
        y_hat = self(x).squeeze(-1)
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)
        
        loss = self.criterion(y_hat, y)
        
        # Update regression metrics
        self.mae_train.update(y_hat, y)
        self.rmse_train.update(y_hat, y)
        self.r2_train.update(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logger.debug("Validation step: batch_idx=%d", batch_idx)
        x, y = batch
        y_hat = self(x).squeeze(-1)
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)
        loss = self.criterion(y_hat, y)
        
        # Update regression metrics
        self.mae_val.update(y_hat, y)
        self.rmse_val.update(y_hat, y)
        self.r2_val.update(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        logger.info("Training epoch ended. Computing training metrics.")
        avg_mae = self.mae_train.compute()
        avg_rmse = torch.sqrt(self.rmse_train.compute())
        avg_r2 = self.r2_train.compute()
        
        self.log('train_mae', avg_mae, prog_bar=True)
        self.log('train_rmse', avg_rmse, prog_bar=True)
        self.log('train_r2', avg_r2, prog_bar=True)
        
        print(f"\n --------- Training Results ---------")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"R2: {avg_r2:.4f}")
        print(f"ElasticNet Fitted: {self.elasticnet_fitted}")
        print(f"--------- Training Complete ---------\n")
        
        # Reset metrics for next epoch
        self.mae_train.reset()
        self.rmse_train.reset()
        self.r2_train.reset()
        
    def on_validation_epoch_end(self):
        logger.info("Validation epoch ended. Computing validation metrics.")
        avg_mae = self.mae_val.compute()
        avg_rmse = torch.sqrt(self.rmse_val.compute())
        avg_r2 = self.r2_val.compute()
        
        self.log('val_mae', avg_mae, prog_bar=True)
        self.log('val_rmse', avg_rmse, prog_bar=True)
        self.log('val_r2', avg_r2, prog_bar=True)
        
        print(f"\n --------- Validation Results ---------")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"R2: {avg_r2:.4f}")
        print(f"--------- Validation Complete ---------\n")
        
        # Reset metrics for next epoch
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
    
    def test_step(self, batch, batch_idx):
        logger.debug("Test step: batch_idx=%d", batch_idx)
        """Test step for final evaluation"""
        x, y = batch
        y_hat = self(x).squeeze(-1)
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)
        loss = self.criterion(y_hat, y)
        
        # Update test metrics (using validation metrics for simplicity)
        self.mae_val.update(y_hat, y)
        self.rmse_val.update(y_hat, y)
        self.r2_val.update(y_hat, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        logger.info("Test epoch ended. Computing test metrics.")
        """Compute and log test metrics at the end of testing"""
        avg_mae = self.mae_val.compute()
        avg_rmse = torch.sqrt(self.rmse_val.compute())
        avg_r2 = self.r2_val.compute()
        
        self.log('test_mae', avg_mae)
        self.log('test_rmse', avg_rmse)
        self.log('test_r2', avg_r2)
        
        print(f"\n ============ TEST RESULTS ============")
        print(f"Test MAE: {avg_mae:.6f}")
        print(f"Test RMSE: {avg_rmse:.6f}")
        print(f"Test R²: {avg_r2:.6f}")
        print(f"=====================================\n")
        
        # Reset metrics
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
    
    def save_components(self):
        logger.info("Saving CNN and Elasticnet model components to disk.")
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        cnnPath = Path(repo_root / self.cfg.model.cnnPath).resolve()
        elasticNetPath = Path(repo_root / self.cfg.model.elasticNetPath).resolve()
        cnnPath.parent.mkdir(parents=True, exist_ok=True)
        elasticNetPath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cnn.state_dict(), cnnPath)
        torch.save(self.elasticnet.state_dict(), elasticNetPath)
        print(f"CNN model saved to {cnnPath} and ElasticNet model saved to {elasticNetPath}.")

    def configure_optimizers(self):
        logger.info("Configuring optimizers and learning rate scheduler.")
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg.optimiser.lr, 
            weight_decay=self.cfg.optimiser.weightDecay,
            eps=self.cfg.optimiser.eps  # Prevent division by zero
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=self.cfg.optimiser.schedulerMode, 
            factor=self.cfg.optimiser.schedulerFactor, 
            patience=self.cfg.optimiser.schedulerPatience,
            min_lr=self.cfg.optimiser.schedulerMinLR
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }