import torch
from torch import nn
import torch.nn.functional as F
import skorch
from skorch import NeuralNetClassifier
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Accuracy
from torchmetrics.regression import R2Score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightning as L
from omegaconf import DictConfig
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
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.squeeze(-1)
        x = self.fc_layer(x)
        if x.dim() > 1 and x.size(-1) == 1:
            x = x.squeeze(-1)
        return x

class RidgeRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.alpha = cfg.Ridge.alpha
        self.fit_intercept = cfg.Ridge.get('fit_intercept', True)

        # Ridge parameters (will be set during fit)
        self.weight = None
        self.bias = None
        self.is_fitted = False
        
    def fit(self, X, y):
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
            X_with_intercept = torch.cat([X, ones], dim=1)
        else:
            X_with_intercept = X
        
        # Ridge regression solution: (X^T X + α I)^(-1) X^T y
        n_features = X_with_intercept.shape[1]
        
        # Create identity matrix for regularization
        I = torch.eye(n_features, device=device)
        
        # Ridge regression closed form solution
        XtX = torch.mm(X_with_intercept.t(), X_with_intercept)
        XtX_reg = XtX + self.alpha * I
        Xty = torch.mv(X_with_intercept.t(), y)
        
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
            self.weight = theta[:-1]
            self.bias = theta[-1]
        else:
            self.weight = theta
            self.bias = torch.tensor(0.0, device=device)
        
        self.is_fitted = True
        
    def forward(self, x):
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


class SklearnWrapper(nn.Module):
    """Simple wrapper to make sklearn estimators work with skorch"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        
    def forward(self, x):
        # This won't be used, but required for nn.Module
        return x


class DirectionalClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.device = None
        self.is_fitted = False
        
        gb_estimator = GradientBoostingClassifier(
            n_estimators=cfg.classifiers.numEstimators[0],
            learning_rate=cfg.classifiers.learningRate,
            max_depth=cfg.classifiers.maxDepth[0],
            subsample=cfg.classifiers.subSample,
            min_samples_split=cfg.classifiers.minSamplesSplit[0],
            min_samples_leaf=cfg.classifiers.minSamplesLeaf[0],
            random_state=cfg.classifiers.randomState
        )
        
        svm_estimator = SVC(
            C=cfg.classifiers.C[1],
            kernel=cfg.classifiers.kernel,
            probability=cfg.classifiers.probability,
            class_weight=cfg.classifiers.classWeight,
            random_state=cfg.classifiers.randomState
        )
        
        logistic_estimator = LogisticRegression(
            C=cfg.classifiers.C[0],
            solver=cfg.classifiers.solver,
            class_weight=cfg.classifiers.classWeight,
            max_iter=cfg.classifiers.maxIterations,
            random_state=cfg.classifiers.randomState
        )
        
        rf_estimator = RandomForestClassifier(
            n_estimators=cfg.classifiers.numEstimators[1],
            max_depth=cfg.classifiers.maxDepth[1],
            min_samples_split=cfg.classifiers.minSamplesSplit[1],
            min_samples_leaf=cfg.classifiers.minSamplesLeaf[1],
            max_features=cfg.classifiers.maxFeatures,
            class_weight=cfg.classifiers.classWeight,
            random_state=cfg.classifiers.randomState
        )
        
        # Wrap with skorch for tensor compatibility
        self.gb_classifier = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=gb_estimator,
            train_split=None,
            verbose=0,
            max_epochs=1,
        )
        
        self.svm_classifier = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=svm_estimator,
            train_split=None,
            verbose=0,
            max_epochs=1,
        )
        
        self.logistic_classifier = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=logistic_estimator,
            train_split=None,
            verbose=0,
            max_epochs=1,
        )
        
        self.random_forest = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=rf_estimator,
            train_split=None,
            verbose=0,
            max_epochs=1,
        )
        
        # Store estimators for direct access
        self.gb_est = gb_estimator
        self.svm_est = svm_estimator
        self.logistic_est = logistic_estimator
        self.rf_est = rf_estimator
        
        # Ensemble weights (4 models: GB, RF, Logistic, SVM)
        self.gb_weight = 0.3
        self.rf_weight = 0.3
        self.logistic_weight = 0.2
        self.svm_weight = 0.2
    
    def fit(self, X, y):
        """
        Fit all classifiers (staying in tensor space as much as possible)
        
        Args:
            X: Features (torch.Tensor) - already scaled
            y: Target labels (torch.Tensor) 
        """
        self.device = X.device
        
        # Flatten if 3D
        if len(X.shape) == 3:
            X_flat = X.view(X.shape[0], -1)
        else:
            X_flat = X
        
        # Only convert to numpy when needed for sklearn estimators
        X_numpy = X_flat.cpu().detach().numpy()
        y_numpy = y.cpu().detach().numpy()
        
        # Fit sklearn estimators directly
        print("Fitting Gradient Boosting classifier...")
        self.gb_est.fit(X_numpy, y_numpy)
        
        # Fit sklearn estimators directly (more reliable than skorch wrapper for this use case)
        print("Fitting Gradient Boosting classifier...")
        self.gb_est.fit(X_numpy, y_numpy)
        
        print("Fitting SVM classifier...")
        self.svm_est.fit(X_numpy, y_numpy)
        
        print("Fitting Logistic Regression classifier...")
        self.logistic_est.fit(X_numpy, y_numpy)
        
        print("Fitting Random Forest classifier...")
        self.rf_est.fit(X_numpy, y_numpy)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities using ensemble (return tensors)
        
        Args:
            X: Features (torch.Tensor) - already scaled
            
        Returns:
            Ensemble probabilities as torch.Tensor
        """
        if not self.is_fitted:
            raise RuntimeError("DirectionalClassifier must be fitted before prediction")
        
        # Flatten if 3D
        if len(X.shape) == 3:
            X_flat = X.view(X.shape[0], -1)
        else:
            X_flat = X
        
        # Convert to numpy for sklearn prediction
        X_numpy = X_flat.cpu().detach().numpy()
        
        # Get probabilities and convert to tensors
        gb_proba = torch.tensor(self.gb_est.predict_proba(X_numpy), device=X.device, dtype=torch.float32)
        svm_proba = torch.tensor(self.svm_est.predict_proba(X_numpy), device=X.device, dtype=torch.float32)
        logistic_proba = torch.tensor(self.logistic_est.predict_proba(X_numpy), device=X.device, dtype=torch.float32)
        rf_proba = torch.tensor(self.rf_est.predict_proba(X_numpy), device=X.device, dtype=torch.float32)
        
        # Ensemble probabilities (4-model ensemble: GB 30%, RF 30%, Logistic 20%, SVM 20%)
        ensemble_proba = (self.gb_weight * gb_proba + 
                         self.rf_weight * rf_proba +
                         self.logistic_weight * logistic_proba + 
                         self.svm_weight * svm_proba)
        
        return ensemble_proba
    
    def predict(self, X):
        """
        Predict classes using ensemble
        
        Args:
            X: Features (torch.Tensor)
            
        Returns:
            Predictions as torch.Tensor
        """
        proba = self.predict_proba(X)
        if isinstance(proba, torch.Tensor):
            return torch.argmax(proba, dim=1)
        else:
            return proba.argmax(axis=1)
    
    def forward(self, x):
        """
        Forward pass for PyTorch compatibility
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions as torch.Tensor
        """
        return self.predict_proba(x)



class EnsembleModule(L.LightningModule):
    """
    Champion CNN-Ridge Ensemble Lightning Module
    
    This combines CNN (10%) + Ridge (90%) in a proper Lightning module
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        # CNN component
        self.cnn = CNN(cfg)
        
        # Ridge component as nn.Module
        self.ridge = RidgeRegressor(cfg)
        
        # Directional classifier component
        self.direction_classifier = DirectionalClassifier(cfg)
        
        # Ensemble weights (optimized from your champion model)
        self.cnn_weight = getattr(cfg.model, 'cnn_weight', 0.1)  # 10% CNN
        self.ridge_weight = getattr(cfg.model, 'ridge_weight', 0.9)  # 90% Ridge
        
        # Loss function
        self.criterion = nn.HuberLoss(delta=cfg.model.get('huber_delta', 1.0))
        
        # Metrics
        self.mae_train = MeanAbsoluteError()
        self.rmse_train = MeanSquaredError()
        self.r2_train = R2Score()
        self.mae_val = MeanAbsoluteError()
        self.rmse_val = MeanSquaredError()
        self.r2_val = R2Score()
        
        # Directional accuracy metrics
        self.direction_accuracy_train = Accuracy(task='binary')
        self.direction_accuracy_val = Accuracy(task='binary')
        
        # Flags to track if models have been fitted
        self.ridge_fitted = False
        self.direction_fitted = False
    
    def on_train_start(self):
        """Fit Ridge regressor and DirectionalClassifier on all training data at the start of training"""
        if not self.ridge_fitted or not self.direction_fitted:
            print("Fitting Ridge regressor and DirectionalClassifier on all training data...")
            
            # Collect all training data
            all_x = []
            all_y = []
            
            # Get the dataloader
            train_dataloader = self.trainer.datamodule.train_dataloader()
            
            # Move model to correct device first
            device = next(self.parameters()).device
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_dataloader):
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    all_x.append(x)
                    all_y.append(y)
                    
                    # Optional: Limit to first N batches to save memory
                    # if batch_idx >= 100:  # First 100 batches
                    #     break
            
            # Concatenate all batches
            X_train = torch.cat(all_x, dim=0)
            y_train = torch.cat(all_y, dim=0)
            
            # Fit Ridge
            if not self.ridge_fitted:
                self.fit_ridge_on_batch(X_train, y_train)
                print(f"Ridge regressor fitted on {len(y_train)} samples!")
            
            # Fit DirectionalClassifier
            if not self.direction_fitted:
                self.fit_direction_on_batch(X_train, y_train)
                print(f"DirectionalClassifier fitted on {len(y_train)} samples!")
            
            # Clear memory
            del all_x, all_y, X_train, y_train
    
    def fit_ridge_on_batch(self, X, y):
        """
        Fit the Ridge component on a batch of training data
        This can be called during training setup
        
        Args:
            X: Training features (torch.Tensor)
            y: Training targets (torch.Tensor)
        """
        self.ridge.fit(X, y)
        self.ridge_fitted = True
    
    def fit_direction_on_batch(self, x, y):
        """Fit DirectionalClassifier on a batch of data with binary direction targets"""
        # Keep as tensors - create binary direction targets (1 for positive, 0 for negative/zero)
        y_direction = (y > 0).int()
        
        # Fit the DirectionalClassifier with tensors
        self.direction_classifier.fit(x, y_direction)
        self.direction_fitted = True
    
    def forward(self, x):
        """
        Forward pass combining CNN and Ridge predictions
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Combined predictions
        """
        # CNN prediction
        cnn_pred = self.cnn(x)
        
        # Ridge prediction (only if fitted)
        if self.ridge_fitted and self.ridge.is_fitted:
            try:
                ridge_pred = self.ridge(x)
                
                # Ensure both predictions are on the same device and have same shape
                if ridge_pred.device != cnn_pred.device:
                    ridge_pred = ridge_pred.to(cnn_pred.device)
                    
                if ridge_pred.shape != cnn_pred.shape:
                    ridge_pred = ridge_pred.view_as(cnn_pred)
                
                # Validate predictions before combining
                if torch.isnan(cnn_pred).any() or torch.isnan(ridge_pred).any():
                    print("Warning: NaN detected in predictions, using CNN only")
                    return cnn_pred
                
                # Combine predictions with optimized weights
                final_pred = self.cnn_weight * cnn_pred + self.ridge_weight * ridge_pred
                
                # Clamp final predictions to prevent extreme values
                final_pred = torch.clamp(final_pred, min=-1e6, max=1e6)
                
                return final_pred
            except Exception as e:
                print(f"Warning: Ridge prediction failed: {e}, using CNN only")
                return cnn_pred
        else:
            # If Ridge not fitted yet, return CNN prediction only
            return cnn_pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)
        
        loss = self.criterion(y_hat, y)
        
        # Update regression metrics
        self.mae_train.update(y_hat, y)
        self.rmse_train.update(y_hat, y)
        self.r2_train.update(y_hat, y)
        
        # Update direction accuracy if direction classifier is fitted
        if self.direction_fitted:
            # Get direction predictions (staying in tensor space)
            direction_pred = self.direction_classifier.predict(x)
            
            # Get true direction (1 for positive, 0 for negative/zero)
            direction_true = (y > 0).int()
            
            # Update direction accuracy
            self.direction_accuracy_train.update(direction_pred, direction_true)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        if y_hat.dim() != y.dim():
            y_hat = y_hat.view_as(y)
        loss = self.criterion(y_hat, y)
        
        # Update regression metrics
        self.mae_val.update(y_hat, y)
        self.rmse_val.update(y_hat, y)
        self.r2_val.update(y_hat, y)
        
        # Update direction accuracy if direction classifier is fitted
        if self.direction_fitted:
            # Get direction predictions (staying in tensor space)
            direction_pred = self.direction_classifier.predict(x)
            
            # Get true direction (1 for positive, 0 for negative/zero)
            direction_true = (y > 0).int()
            
            # Update direction accuracy
            self.direction_accuracy_val.update(direction_pred, direction_true)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        avg_mae = self.mae_train.compute()
        avg_rmse = torch.sqrt(self.rmse_train.compute())
        avg_r2 = self.r2_train.compute()
        avg_direction_acc = self.direction_accuracy_train.compute()
        
        self.log('train_mae', avg_mae, prog_bar=True)
        self.log('train_rmse', avg_rmse, prog_bar=True)
        self.log('train_r2', avg_r2, prog_bar=True)
        self.log('train_direction_acc', avg_direction_acc, prog_bar=True)
        
        print(f"\n --------- Training Results ---------")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"R2: {avg_r2:.4f}")
        print(f"Direction Accuracy: {avg_direction_acc:.4f}")
        print(f"Ridge Fitted: {self.ridge_fitted}")
        print(f"Direction Fitted: {self.direction_fitted}")
        print(f"--------- Training Complete ---------\n")
        
        # Reset metrics for next epoch
        self.mae_train.reset()
        self.rmse_train.reset()
        self.r2_train.reset()
        self.direction_accuracy_train.reset()
        
    def on_validation_epoch_end(self):
        avg_mae = self.mae_val.compute()
        avg_rmse = torch.sqrt(self.rmse_val.compute())
        avg_r2 = self.r2_val.compute()
        avg_direction_acc = self.direction_accuracy_val.compute()
        
        self.log('val_mae', avg_mae, prog_bar=True)
        self.log('val_rmse', avg_rmse, prog_bar=True)
        self.log('val_r2', avg_r2, prog_bar=True)
        self.log('val_direction_acc', avg_direction_acc, prog_bar=True)
        
        print(f"\n --------- Validation Results ---------")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"R2: {avg_r2:.4f}")
        print(f"Direction Accuracy: {avg_direction_acc:.4f}")
        print(f"--------- Validation Complete ---------\n")
        
        # Reset metrics for next epoch
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
        self.direction_accuracy_val.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg.optimiser.lr, 
            weight_decay=self.cfg.optimiser.weightDecay,
            eps=1e-8  # Prevent division by zero
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-7  # Prevent learning rate from becoming too small
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }