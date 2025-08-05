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
from pathlib import Path
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
            self.weight = theta[:-1]
            self.bias = theta[-1]
        else:
            self.weight = theta
            self.bias = torch.tensor(0.0, device=device)
        # Register as parameters (no gradient needed)
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.bias = nn.Parameter(self.bias, requires_grad=False)
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
        
        GBEstimator = GradientBoostingClassifier(
            n_estimators=cfg.classifiers.numEstimators,
            learning_rate=cfg.classifiers.learningRate,
            max_depth=cfg.classifiers.maxDepth,
            subsample=cfg.classifiers.subSample,
            min_samples_split=cfg.classifiers.minSamplesSplit,
            min_samples_leaf=cfg.classifiers.minSamplesLeaf,
            random_state=cfg.classifiers.randomState
        )
        
        SVMEstimator = SVC(
            C=cfg.classifiers.C,
            kernel=cfg.classifiers.kernel,
            probability=cfg.classifiers.probability,
            class_weight=cfg.classifiers.classWeight,
            random_state=cfg.classifiers.randomState
        )
        
        LogisticEstimator = LogisticRegression(
            C=cfg.classifiers.C,
            solver=cfg.classifiers.solver,
            class_weight=cfg.classifiers.classWeight,
            max_iter=cfg.classifiers.maxIterations,
            random_state=cfg.classifiers.randomState
        )
        
        RFEstimator = RandomForestClassifier(
            n_estimators=cfg.classifiers.numEstimators,
            max_depth=cfg.classifiers.maxDepth,
            min_samples_split=cfg.classifiers.minSamplesSplit,
            min_samples_leaf=cfg.classifiers.minSamplesLeaf,
            max_features=cfg.classifiers.maxFeatures,
            class_weight=cfg.classifiers.classWeight,
            random_state=cfg.classifiers.randomState
        )
        
        # Wrap with skorch for tensor compatibility
        self.gbClassifier = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=GBEstimator,
            train_split=None,
            verbose=cfg.classifiers.verbose,
            max_epochs=cfg.classifiers.MAXEPOCHS,
        )
        
        self.svmClassifier = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=SVMEstimator,
            train_split=None,
            verbose=cfg.classifiers.verbose,
            max_epochs=cfg.classifiers.MAXEPOCHS,
        )
        
        self.logisticClassifier = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=LogisticEstimator,
            train_split=None,
            verbose=cfg.classifiers.verbose,
            max_epochs=cfg.classifiers.MAXEPOCHS,
        )
        
        self.randomForest = NeuralNetClassifier(
            SklearnWrapper,
            module__estimator=RFEstimator,
            train_split=None,
            verbose=cfg.classifiers.verbose,
            max_epochs=cfg.classifiers.MAXEPOCHS,
        )
        
        # Store estimators for direct access
        self.gb_est = GBEstimator
        self.svm_est = SVMEstimator
        self.logistic_est = LogisticEstimator
        self.rf_est = RFEstimator
        
        # Ensemble weights (4 models: GB, RF, Logistic, SVM)
        self.gb_weight = cfg.classifiers.GBWEIGHT
        self.rf_weight = cfg.classifiers.RFWEIGHT
        self.logistic_weight = cfg.classifiers.LOGISTICWEIGHT
        self.svm_weight = cfg.classifiers.SVMWEIGHT
        
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
            XFlat = X.view(X.shape[0], -1)
        else:
            XFlat = X
        
        # Only convert to numpy when needed for sklearn estimators
        XNumpy = XFlat.cpu().detach().numpy()
        YNumpy = y.cpu().detach().numpy()
        
        # Fit sklearn estimators directly
        print("Fitting Gradient Boosting classifier...")
        self.gb_est.fit(XNumpy, YNumpy)
        
        print("Fitting SVM classifier...")
        self.svm_est.fit(XNumpy, YNumpy)
        
        print("Fitting Logistic Regression classifier...")
        self.logistic_est.fit(XNumpy, YNumpy)
        
        print("Fitting Random Forest classifier...")
        self.rf_est.fit(XNumpy, YNumpy)
        
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
            XFlat = X.view(X.shape[0], -1)
        else:
            XFlat = X
        
        # Convert to numpy for sklearn prediction
        XNumpy = XFlat.cpu().detach().numpy()
        
        # Get probabilities and convert to tensors
        gb_proba = torch.tensor(self.gb_est.predict_proba(XNumpy), device=X.device, dtype=torch.float32)
        svm_proba = torch.tensor(self.svm_est.predict_proba(XNumpy), device=X.device, dtype=torch.float32)
        logistic_proba = torch.tensor(self.logistic_est.predict_proba(XNumpy), device=X.device, dtype=torch.float32)
        rf_proba = torch.tensor(self.rf_est.predict_proba(XNumpy), device=X.device, dtype=torch.float32)
        
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
        self.cnnWeight = cfg.model.cnnWeight
        self.ridgeWeight = cfg.model.ridgeWeight

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

            # Fit Ridge
            if not self.ridge_fitted:
                self.fit_ridge_on_batch(XTrain, yTrain)
                print(f"Ridge regressor fitted on {len(yTrain)} samples!")
            
            # Fit DirectionalClassifier
            if not self.direction_fitted:
                self.fit_direction_on_batch(XTrain, yTrain)
                print(f"DirectionalClassifier fitted on {len(yTrain)} samples!")
            
            # Clear memory
            del allXValues, allYValues, XTrain, yTrain

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
        yPrevious = torch.cat([y[:1], y[:-1]])  # Shift by 1, duplicate first element
        deltaY = y - yPrevious  # Current price - previous price

        # Create binary direction targets based on price CHANGES (1 for up, 0 for down/flat)
        yDirection = (deltaY > 0).int()

        print(f"Direction targets - Positive changes: {(deltaY > 0).sum()}/{len(deltaY)} ({(deltaY > 0).float().mean()*100:.1f}%)")
        print(f"Direction targets - Negative changes: {(deltaY < 0).sum()}/{len(deltaY)} ({(deltaY < 0).float().mean()*100:.1f}%)")
        print(f"Direction targets - Zero changes: {(deltaY == 0).sum()}/{len(deltaY)} ({(deltaY == 0).float().mean()*100:.1f}%)")

        # Fit the DirectionalClassifier with tensors
        self.direction_classifier.fit(x, yDirection)
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
        cnnPrediction = self.cnn(x)
        
        # Ridge prediction (only if fitted)
        if self.ridge_fitted and self.ridge.is_fitted:
            try:
                ridgePrediction = self.ridge(x)

                # Ensure both predictions are on the same device and have same shape
                if ridgePrediction.device != cnnPrediction.device:
                    ridgePrediction = ridgePrediction.to(cnnPrediction.device)

                if ridgePrediction.shape != cnnPrediction.shape:
                    ridgePrediction = ridgePrediction.view_as(cnnPrediction)

                # Validate predictions before combining
                if torch.isnan(cnnPrediction).any() or torch.isnan(ridgePrediction).any():
                    print("Warning: NaN detected in predictions, using CNN only")
                    return cnnPrediction

                # Combine predictions with optimized weights
                finalPrediction = self.cnnWeight * cnnPrediction + self.ridgeWeight * ridgePrediction

                # Clamp final predictions to prevent extreme values
                finalPrediction = torch.clamp(finalPrediction, min=-1e6, max=1e6)

                return finalPrediction
            except Exception as e:
                print(f"Warning: Ridge prediction failed: {e}, using CNN only")
                return cnnPrediction
        else:
            # If Ridge not fitted yet, return CNN prediction only
            return cnnPrediction

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
            directionPrediction = self.direction_classifier.predict(x)

            yPrevious = torch.cat([y[:1], y[:-1]])  # Previous day's price
            deltaY = y - yPrevious  # Price change
            directionTrue = (deltaY > 0).int()  # 1 for up, 0 for down/flat

            # Update direction accuracy
            self.direction_accuracy_train.update(directionPrediction, directionTrue)

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
            directionPrediction = self.direction_classifier.predict(x)
            
            # CRITICAL FIX: Compute actual price changes for true direction
            yPrevious = torch.cat([y[:1], y[:-1]])  # Previous day's price
            deltaY = y - yPrevious  # Price change
            directionTrue = (deltaY > 0).int()  # 1 for up, 0 for down/flat

            # Update direction accuracy
            self.direction_accuracy_val.update(directionPrediction, directionTrue)

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
    
    def test_step(self, batch, batch_idx):
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
        
        # Update direction accuracy if direction classifier is fitted
        if self.direction_fitted:
            # Get direction predictions
            directionPrediction = self.direction_classifier.predict(x)
            
            # Compute actual price changes for true direction
            yPrevious = torch.cat([y[:1], y[:-1]])  # Previous day's price
            deltaY = y - yPrevious  # Price change
            directionTrue = (deltaY > 0).int()  # 1 for up, 0 for down/flat

            # Update direction accuracy
            self.direction_accuracy_val.update(directionPrediction, directionTrue)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        """Compute and log test metrics at the end of testing"""
        avg_mae = self.mae_val.compute()
        avg_rmse = torch.sqrt(self.rmse_val.compute())
        avg_r2 = self.r2_val.compute()
        avg_direction_acc = self.direction_accuracy_val.compute()
        
        self.log('test_mae', avg_mae)
        self.log('test_rmse', avg_rmse)
        self.log('test_r2', avg_r2)
        self.log('test_direction_acc', avg_direction_acc)
        
        print(f"\n ============ TEST RESULTS ============")
        print(f"Test MAE: {avg_mae:.6f}")
        print(f"Test RMSE: {avg_rmse:.6f}")
        print(f"Test R²: {avg_r2:.6f}")
        print(f"Test Direction Accuracy: {avg_direction_acc:.6f}")
        print(f"=====================================\n")
        
        # Reset metrics
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
        self.direction_accuracy_val.reset()
    
    def save_components(self):
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        cnnPath = repo_root / self.cfg.model.cnnPath.lstrip('../')
        ridgePath = repo_root / self.cfg.model.ridgePath.lstrip('../')
        cnnPath.parent.mkdir(parents=True, exist_ok=True)
        ridgePath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cnn.state_dict(), cnnPath)
        torch.save(self.ridge.state_dict(), ridgePath)
        print(f"CNN model saved to {cnnPath} and Ridge model saved to {ridgePath}.")

    def configure_optimizers(self):
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