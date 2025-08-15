import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Accuracy, F1Score
from torchmetrics.regression import R2Score
import lightning as L
from omegaconf import DictConfig
from pathlib import Path
from scripts.logging_config import get_logger, setup_logging

setup_logging(log_level="INFO", console_output=True, file_output=True)
logger = get_logger("model_Ensemble")
class MultiHeadCNN(nn.Module):
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
        
        self.sharedfeatures = nn.Sequential(
            nn.Linear(cnnChannels[2], cnnChannels[1]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[1]),
            )
        
        self.pricehead = nn.Sequential(
            nn.Linear(cnnChannels[1], cnnChannels[2]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[0]),
            nn.Linear(cnnChannels[2], 1)  # Output for price prediction
        )
        
        num_classes = cfg.cnn.get('num_classes', 3)
        self.directionhead = nn.Sequential(
            nn.Linear(cnnChannels[1], cnnChannels[2]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[0]),
            nn.Linear(cnnChannels[2], num_classes)  # Output for direction prediction
        )

    def forward(self, x):
        logger.debug("MultiHeadCNN forward pass with input shape: %s", x.shape)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.squeeze(-1)
        x = self.sharedfeatures(x)
        price_pred = self.pricehead(x).squeeze(-1)
        direction_pred = self.directionhead(x)
        return price_pred, direction_pred

class RidgeRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing RidgeRegressor with alpha=%.4f, fit_intercept=%s", cfg.Ridge.alpha, cfg.Ridge.get('fit_intercept', True))
        super().__init__()
        self.alpha = cfg.Ridge.alpha
        self.fit_intercept = cfg.Ridge.get('fit_intercept', True)
        self.eps = cfg.Ridge.get('eps', 1e-8)  # Small value to prevent division by zero
        
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
        XtX_reg = XtX + self.alpha * I + (self.eps * torch.eye(numFeatures, device=device))
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
    
class ElasticNetClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing ElasticNetClassifier with alpha=%.4f, l1_ratio=%.4f, fit_intercept=%s",
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
        
    def fit_single_class(self, X, y_binary, device):
        numSamples, numFeatures = X.shape
        
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0) + self.eps
        X_standardised = (X - X_mean) / X_std
        
        if self.fit_intercept:
            y_mean = y_binary.float().mean()
            y_centred = y_binary.float() - y_mean
        else:
            self.y_mean = y_binary.float().mean()
        
        beta = torch.zeros(numFeatures, device=device, dtype=X.dtype)
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1 - self.l1_ratio)
        
        # Create zero variance mask
        zero_var_mask = X_std < self.eps
        
        # Replace zero variances with 1.0
        X_std = torch.where(zero_var_mask, torch.tensor(1.0, device=device, dtype=X.dtype), X_std)
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            beta_old = beta.clone()
            
            # Vectorized computation of residuals for all features
            residuals = y_centred - X_standardised @ beta
            
            # Vectorized computation of rho_j for all features
            rho = torch.matmul(X_standardised.t(), residuals) / numSamples
            
            # Apply zero variance mask
            beta = torch.where(zero_var_mask, torch.tensor(0.0, device=device, dtype=X.dtype), beta)
            
            # Vectorized update of beta
            denominator = 1.0 + (l2_reg / numSamples)
            beta = torch.where(~zero_var_mask, 
                            self.soft_threshold(rho, l1_reg / numSamples) / denominator,
                            beta)
            if torch.norm(beta - beta_old) < self.tol:
                logger.info("Converged after %d iterations", iteration + 1)
                break
        weight = beta / X_std
        if self.fit_intercept:
            bias = y_mean - torch.dot(X_mean / X_std, beta)
        else:
            bias = torch.tensor(0.0, device=device)
            
        return weight, bias
    
    def fit(self, X, y):
        """
        Fit ElasticNet using one-vs-rest for multi-class classification
        """
        logger.info("Fitting ElasticNetClassifier on data: X shape %s, y shape %s", X.shape, y.shape)
        # Ensure tensors are on same device
        device = X.device
        y = y.to(device)
        
        # Flatten X to 2D if it's 3D: (batch_size, seq_len, features) -> (batch_size, seq_len * features)
        if len(X.shape) == 3:
            X = X.view(X.shape[0], -1)
        
        numSamples, numFeatures = X.shape
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.is_fitted = False
        
        for class_idx in range(self.num_classes):
            y_binary = (y == class_idx).int()
            weight, bias = self.fit_single_class(X, y_binary, device)
            self.weights.append(nn.Parameter(weight.clone(), requires_grad=False))
            self.biases.append(nn.Parameter(bias.clone(), requires_grad=False))
        
        self.is_fitted = True
        logger.info("ElasticNetRegressor fitted successfully with weight shape %s and bias %s", self.weight.shape, self.bias.shape)
        
    def forward(self, x):
        logger.debug("ElasticNetClassifier forward pass with input shape: %s", x.shape)
        if not self.is_fitted:
            raise RuntimeError("ElasticNet classifier must be fitted before forward pass. Call .fit() first.")
        
        # Flatten to 2D if needed
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        
        # Ensure tensors are on same device
        device = self.weights[0].device
        x = x.to(device)
        
        # Compute scores for all classes
        scores = []
        for class_idx in range(self.num_classes):
            score = torch.mv(x, self.weights[class_idx])
            if self.fit_intercept:
                score = score + self.biases[class_idx]
            scores.append(score.unsqueeze(1))
        
        # Stack scores and return logits
        logits = torch.cat(scores, dim=1)
        return logits
    

class EnsembleModule(L.LightningModule):
    """
    Multi-headed CNN-Ridge-ElasticNet Ensemble Lightning Module
    - CNN provides both price and direction predictions
    - Ridge complements price predictions
    - ElasticNet complements direction predictions
    """
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing EnsembleModule with config: %s", cfg.model)
        super().__init__()
        self.cfg = cfg
        
        # CNN component
        self.cnn = MultiHeadCNN(cfg)
        
        # Ridge component for price prediction
        self.ridge = RidgeRegressor(cfg)
        
        # ElasticNet component as direction classification
        self.elasticnet = ElasticNetClassifier(cfg)
        
        # Ensemble weights
        self.price_cnn_weight = cfg.model.price_cnn_weight
        self.price_ridge_weight = cfg.model.ridge_weight
        self.direction_cnn_weight = cfg.model.direction_cnn_weight
        self.direction_elasticnet_weight = cfg.model.elasticnet_weight
        
        # Loss function
        self.price_criterion = nn.HuberLoss(delta=cfg.model.get('huber_delta', 1.0))
        self.direction_criterion = nn.CrossEntropyLoss()
        
        # Price Metrics
        self.mae_train = MeanAbsoluteError()
        self.rmse_train = MeanSquaredError()
        self.r2_train = R2Score()
        self.mae_val = MeanAbsoluteError()
        self.rmse_val = MeanSquaredError()
        self.r2_val = R2Score()
        
        # Direction Metrics
        self.direction_acc_train = Accuracy(task="multiclass", num_classes=3)
        self.direction_f1_train = F1Score(task="multiclass", num_classes=3, average='weighted')
        self.direction_acc_val = Accuracy(task="multiclass", num_classes=3)
        self.direction_f1_val = F1Score(task="multiclass", num_classes=3, average='weighted')
        
        
        # Flags to track if models have been fitted
        self.ridge_fitted = False
        self.elasticnet_fitted = False
        
    def on_train_start(self):
        logger.info("on_train_start: Fitting Ridge and ElasticNet on all training data...")
        """Fit Ridge and ElasticNet on all training data at the start of training"""
        if not (self.ridge_fitted and self.elasticnet_fitted):
            logger.info("Fitting Ridge and ElasticNet on all training data...")
            
            # Collect all training data
            all_X_Values = []
            all_Price_Values = []
            all_Direction_Values = []
            
            # Get the dataloader and move model to correct device
            train_dataloader = self.trainer.datamodule.train_dataloader()
            device = next(self.parameters()).device
            
            with torch.no_grad():
            # Concatenate all batches from dataloader
                x_all = []
                price_y_all = []
                for batch in train_dataloader:
                    x, price_y = batch
                    x_all.append(x)
                    price_y_all.append(price_y)
                
                # Move concatenated tensors to device
                x_all = torch.cat(x_all, dim=0).to(device)
                price_y_all = torch.cat(price_y_all, dim=0).to(device)
                
                # Initialize output lists
                all_X_Values = [x_all]
                all_Price_Values = [price_y_all]
                all_Direction_Values = []
                
                # Calculate price changes
                price_changes = torch.zeros_like(price_y_all)
                if len(all_Price_Values) > 1:
                    prev_prices = all_Price_Values[-2][-price_y_all.shape[0]:]
                    price_changes = ((price_y_all - prev_prices) / prev_prices) * 100
                # For the first batch, price_changes remains zeros
                
                # Convert to direction classes
                threshold = 0.5
                direction_y = torch.ones_like(price_y_all, dtype=torch.long)  # Default: sideways
                direction_y[price_changes > threshold] = 2  # Up
                direction_y[price_changes < -threshold] = 0  # Down
                
                all_X_Values.append(x_all)
                all_Price_Values.append(price_y_all)
                all_Direction_Values.append(direction_y)
            
            # Ridge and ElasticNet fitting
            if not self.ridge_fitted:
                self.ridge.fit(x_all, price_y_all)
                self.ridge_fitted = True
                print(f"Ridge fitted on {len(price_y_all)} samples!")
            
            if not self.elasticnet_fitted:
                self.elasticnet.fit(x_all, direction_y)
                self.elasticnet_fitted = True
                print(f"ElasticNet fitted on {len(direction_y)} samples!")
            # Clear memory
            del all_X_Values, all_Price_Values, all_Direction_Values
    
    def forward(self, x):
        """Forward pass combining all model predictions"""
        logger.debug("MultiHeadEnsemble forward pass with input shape: %s", x.shape)
        
        # CNN predictions (both heads)
        cnn_price, cnn_direction_logits = self.cnn(x)
        
        # Ridge price prediction
        ridge_price = None
        if self.ridge_fitted and self.ridge.is_fitted:
            try:
                ridge_price = self.ridge(x)
                if ridge_price.device != cnn_price.device:
                    ridge_price = ridge_price.to(cnn_price.device)
                if ridge_price.shape != cnn_price.shape:
                    ridge_price = ridge_price.view_as(cnn_price)
            except Exception as e:
                logger.warning(f"Ridge prediction failed: {e}, using CNN only")
                ridge_price = None
        
        # ElasticNet direction prediction
        elasticnet_direction = None
        if self.elasticnet_fitted and self.elasticnet.is_fitted:
            try:
                elasticnet_direction = self.elasticnet(x)
                if elasticnet_direction.device != cnn_direction_logits.device:
                    elasticnet_direction = elasticnet_direction.to(cnn_direction_logits.device)
                if elasticnet_direction.shape != cnn_direction_logits.shape:
                    elasticnet_direction = elasticnet_direction.view_as(cnn_direction_logits)
            except Exception as e:
                logger.warning(f"ElasticNet prediction failed: {e}, using CNN only")
                elasticnet_direction = None
        
        # Combine predictions
        if ridge_price is not None:
            final_price = (self.price_cnn_weight * cnn_price + 
                          self.price_ridge_weight * ridge_price)
        else:
            final_price = cnn_price
        
        if elasticnet_direction is not None:
            final_direction = (self.direction_cnn_weight * cnn_direction_logits + 
                              self.direction_elasticnet_weight * elasticnet_direction)
        else:
            final_direction = cnn_direction_logits
        
        # Clamp price predictions to prevent extreme values
        final_price = torch.clamp(final_price, min=-1e6, max=1e6)
        
        return final_price, final_direction
        
    def training_step(self, batch, batch_idx):
        logger.debug("Training step: batch_idx=%d", batch_idx)
        
        # Current batch structure: (x, price_target)
        x, price_y = batch
        
        # Derive direction targets from price changes
        # Since we don't have the actual previous prices, we'll use a proxy
        # You may need to modify your data module to include direction targets
        if batch_idx > 0 and hasattr(self, '_prev_price_batch'):
            price_changes = ((price_y - self._prev_price_batch) / self._prev_price_batch) * 100
        else:
            # For first batch or when previous batch not available, use zeros
            price_changes = torch.zeros_like(price_y)
        
        threshold = 0.5
        direction_y = torch.ones(price_y.shape[0], dtype=torch.long, device=self.device)
        direction_y[price_changes > threshold] = 2  # Up
        direction_y[price_changes < -threshold] = 0  # Down
        
        # Store current prices for next batch
        self._prev_price_batch = price_y.detach()
        
        price_pred, direction_pred = self(x)
        
        # Ensure predictions match target dimensions
        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)
        
        # Compute losses
        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)
        
        # Combined loss
        total_loss = (self.price_loss_weight * price_loss + 
                     self.direction_loss_weight * direction_loss)
        
        # Update metrics
        self.price_mae_train.update(price_pred, price_y)
        self.price_rmse_train.update(price_pred, price_y)
        self.price_r2_train.update(price_pred, price_y)
        
        direction_preds_class = torch.argmax(direction_pred, dim=1)
        self.direction_acc_train.update(direction_preds_class, direction_y)
        self.direction_f1_train.update(direction_preds_class, direction_y)
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_price_loss', price_loss, on_step=True, on_epoch=True)
        self.log('train_direction_loss', direction_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        logger.debug("Validation step: batch_idx=%d", batch_idx)
        
        # Current batch structure: (x, price_target)
        x, price_y = batch
        
        # For validation, we'll use a simpler direction derivation
        # You might want to include actual direction targets in your data
        if batch_idx > 0 and hasattr(self, '_prev_val_price_batch'):
            price_changes = ((price_y - self._prev_val_price_batch) / self._prev_val_price_batch) * 100
        else:
            price_changes = torch.zeros_like(price_y)
            
        threshold = 0.5
        direction_y = torch.ones(price_y.shape[0], dtype=torch.long, device=self.device)
        direction_y[price_changes > threshold] = 2  # Up
        direction_y[price_changes < -threshold] = 0  # Down
        
        self._prev_val_price_batch = price_y.detach()
        
        price_pred, direction_pred = self(x)
        
        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)
        
        # Compute losses
        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)
        total_loss = (self.price_loss_weight * price_loss + 
                     self.direction_loss_weight * direction_loss)
        
        # Update metrics
        self.price_mae_val.update(price_pred, price_y)
        self.price_rmse_val.update(price_pred, price_y)
        self.price_r2_val.update(price_pred, price_y)
        
        direction_preds_class = torch.argmax(direction_pred, dim=1)
        self.direction_acc_val.update(direction_preds_class, direction_y)
        self.direction_f1_val.update(direction_preds_class, direction_y)
        
        # Log losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_price_loss', price_loss, on_step=False, on_epoch=True)
        self.log('val_direction_loss', direction_loss, on_step=False, on_epoch=True)
        
        return total_loss
        
    def on_train_epoch_end(self):
        logger.info("Training epoch ended. Computing training metrics.")
        
        # Price metrics
        price_mae = self.price_mae_train.compute()
        price_rmse = torch.sqrt(self.price_rmse_train.compute())
        price_r2 = self.price_r2_train.compute()
        
        # Direction metrics
        direction_acc = self.direction_acc_train.compute()
        direction_f1 = self.direction_f1_train.compute()
        
        # Log metrics
        self.log('train_price_mae', price_mae, prog_bar=True)
        self.log('train_price_rmse', price_rmse, prog_bar=True)
        self.log('train_price_r2', price_r2, prog_bar=True)
        self.log('train_direction_acc', direction_acc, prog_bar=True)
        self.log('train_direction_f1', direction_f1, prog_bar=True)
        
        print(f"\n --------- Training Results ---------")
        print(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        print(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        print(f"Models Fitted - Ridge: {self.ridge_fitted}, ElasticNet: {self.elasticnet_fitted}")
        print(f"--------- Training Complete ---------\n")
        
        # Reset metrics
        self.price_mae_train.reset()
        self.price_rmse_train.reset()
        self.price_r2_train.reset()
        self.direction_acc_train.reset()
        self.direction_f1_train.reset()
    
    def on_validation_epoch_end(self):
        logger.info("Validation epoch ended. Computing validation metrics.")
        
        # Price metrics
        price_mae = self.price_mae_val.compute()
        price_rmse = torch.sqrt(self.price_rmse_val.compute())
        price_r2 = self.price_r2_val.compute()
        
        # Direction metrics
        direction_acc = self.direction_acc_val.compute()
        direction_f1 = self.direction_f1_val.compute()
        
        # Log metrics
        self.log('val_price_mae', price_mae, prog_bar=True)
        self.log('val_price_rmse', price_rmse, prog_bar=True)
        self.log('val_price_r2', price_r2, prog_bar=True)
        self.log('val_direction_acc', direction_acc, prog_bar=True)
        self.log('val_direction_f1', direction_f1, prog_bar=True)
        
        print(f"\n --------- Validation Results ---------")
        print(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        print(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        print(f"--------- Validation Complete ---------\n")
        
        # Reset metrics
        self.price_mae_val.reset()
        self.price_rmse_val.reset()
        self.price_r2_val.reset()
        self.direction_acc_val.reset()
        self.direction_f1_val.reset()
    
    def test_step(self, batch, batch_idx):
        logger.debug("Test step: batch_idx=%d", batch_idx)
        return self.validation_step(batch, batch_idx)  # Reuse validation logic
    
    def on_test_epoch_end(self):
        logger.info("Test epoch ended. Computing test metrics.")

        # Price metrics
        price_mae = self.price_mae_val.compute()
        price_rmse = torch.sqrt(self.price_rmse_val.compute())
        price_r2 = self.price_r2_val.compute()
        
        # Direction metrics
        direction_acc = self.direction_acc_val.compute()
        direction_f1 = self.direction_f1_val.compute()
        
        # Log metrics
        self.log('val_price_mae', price_mae, prog_bar=True)
        self.log('val_price_rmse', price_rmse, prog_bar=True)
        self.log('val_price_r2', price_r2, prog_bar=True)
        self.log('val_direction_acc', direction_acc, prog_bar=True)
        self.log('val_direction_f1', direction_f1, prog_bar=True)
        
        print(f"\n --------- Test Results ---------")
        print(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        print(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        print(f"--------- Test Complete ---------\n")

        # Reset metrics
        self.price_mae_val.reset()
        self.price_rmse_val.reset()
        self.price_r2_val.reset()
        self.direction_acc_val.reset()
        self.direction_f1_val.reset()
    
    def save_components(self):
        logger.info("Saving CNN and Elasticnet model components to disk.")
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        direction_cnnPath = Path(repo_root / self.cfg.model.direction_cnnPath).resolve()
        elasticNetPath = Path(repo_root / self.cfg.model.elasticNetPath).resolve()
        price_cnnPath = Path(repo_root / self.cfg.model.price_cnnPath).resolve()
        ridgePath = Path(repo_root / self.cfg.model.ridgePath).resolve()
        direction_cnnPath.parent.mkdir(parents=True, exist_ok=True)
        elasticNetPath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cnn.state_dict(), direction_cnnPath)
        torch.save(self.elasticnet.state_dict(), elasticNetPath)
        torch.save(self.price_cnn.state_dict(), price_cnnPath)
        torch.save(self.ridge.state_dict(), ridgePath)
        print(f"DirectionCNN model saved to {direction_cnnPath}, ElasticNet model saved to {elasticNetPath}")
        print(f"PriceCNN model saved to {price_cnnPath}, and Ridge model saved to {ridgePath}.")

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