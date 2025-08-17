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
        
        self.price_features = nn.Sequential(
            nn.Linear(cnnChannels[2], cnnChannels[1]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[1]),
            )
        
        self.direction_features = nn.Sequential(
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
        
        num_classes = cfg.cnn.num_classes
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
        
        price_features = self.price_features(x)
        direction_features = self.direction_features(x)
        
        price_pred = self.pricehead(price_features).squeeze(-1)
        direction_pred = self.directionhead(direction_features)
        
        return price_pred, direction_pred, price_features, direction_features

class RidgeRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing RidgeRegressor with alpha=%.4f, fit_intercept=%s", cfg.Ridge.alpha, cfg.Ridge.get('fit_intercept', True))
        super().__init__()
        self.alpha = cfg.Ridge.alpha
        self.fit_intercept = cfg.Ridge.get('fit_intercept', True)
        self.eps = cfg.Ridge.get('eps', 1e-8)  # Small value to prevent division by zero
        
        # Ridge parameters (will be set during fit)
        self.register_buffer('weight', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('bias', torch.tensor(0.0, requires_grad=False))
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
        XtX_reg = XtX + self.alpha * I + (self.eps * I)
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
            self.weight.data = theta[:-1].clone().detach()
            self.bias.data = theta[-1].clone().detach()
        else:
            self.weight.data = theta.clone().detach()
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
        self.num_classes = cfg.cnn.num_classes

        # ElasticNet parameters (will be set during fit)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.is_fitted = False
        
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        
    def soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)
        
    def fit_single_class(self, X, y_binary, device):
        numSamples, numFeatures = X.shape
        # Skip standardization since data is already scaled and clipped in the pipeline
        # Use the input data directly to avoid re-introducing extreme values
        X_standardised = X
        X_mean = torch.zeros(numFeatures, device=device, dtype=X.dtype)
        X_std = torch.ones(numFeatures, device=device, dtype=X.dtype)
        if self.fit_intercept:
            y_mean = y_binary.float().mean()
            y_centred = y_binary.float() - y_mean
        else:
            y_centred = y_binary.float()
            y_mean = torch.tensor(0.0, device=device)
        
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
            beta_new = self.soft_threshold(rho, l1_reg / numSamples) / denominator
            
            # Clip beta to prevent explosion
            beta = torch.clamp(beta_new, min=-1e6, max=1e6)
            
            # Check for convergence
            if torch.norm(beta - beta_old) < self.tol:
                logger.info("Converged after %d iterations", iteration + 1)
                break
                
            # Safety check: if beta becomes too large, stop and use zero weights
            if torch.abs(beta).max() > 1e5:
                logger.warning(f"Beta coefficients too large (max: {torch.abs(beta).max():.2e}), using zero weights")
                weight = torch.zeros_like(beta)
                bias = torch.tensor(0.0, device=device)
                return weight, bias
        weight = beta
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
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.is_fitted = False
        
        for class_idx in range(self.num_classes):
            y_binary = (y == class_idx).int()
            weight, bias = self.fit_single_class(X, y_binary, device)
            self.weights.append(nn.Parameter(weight.clone(), requires_grad=False))
            self.biases.append(nn.Parameter(bias.clone(), requires_grad=False))
        
        self.is_fitted = True
        logger.info("ElasticNetClassifier fitted successfully with %d classes", self.num_classes)
        
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
    
class MetaPriceRegressor(nn.Module):
    """
    Linear regressor over base model outputs (CNN price, Ridge price, direction probs, elastic probs).
    Acts like learned blending (ridge-like if you add weight_decay in optimizer).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    
    def forward(self, z):   # z: (B, in_dim)
        return self.linear(z).squeeze(-1)  # (B,)
    

class MetaDirectionClassifier(nn.Module):
    """
    Linear classifier over base outputs (CNN logits/probs, Elastic logits/probs, optional prices).
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    
    def forward(self, z):   # z: (B, in_dim)
        return self.linear(z)  # logits (B, C)
    

class EnsembleModule(L.LightningModule):
    """
    Multi-headed CNN-Ridge-ElasticNet Ensemble Lightning Module
    - CNN provides both price and direction predictions
    - Ridge complements price predictions
    - ElasticNet complements direction predictions
    - Meta-learners to blend all predictions
    """
    def __init__(self, cfg: DictConfig):
        logger.info("Initializing EnsembleModule with config: %s", cfg.model)
        super().__init__()
        self.cfg = cfg
        
        # NN components
        self.cnn = MultiHeadCNN(cfg)
        self.ridge = RidgeRegressor(cfg)
        self.elasticNet = ElasticNetClassifier(cfg)
        
        self.num_classes = cfg.cnn.num_classes
        meta_in_dim_dir = 1 + 1 + self.num_classes + self.num_classes
        meta_in_dim_price = 1 + 1 + self.num_classes + self.num_classes
        self.meta_price = MetaPriceRegressor(in_dim=meta_in_dim_price)
        self.meta_dir = MetaDirectionClassifier(in_dim=meta_in_dim_dir, num_classes=self.num_classes)
        
        # Meta-learning configurations
        self.use_meta_learning = cfg.model.use_meta_learning
        self.include_base_losses = self.cfg.model.include_base_losses
        self.meta_price_loss_weight = cfg.model.meta_price_loss_weight
        self.meta_direction_loss_weight = cfg.model.meta_direction_loss_weight
        
        # Ensemble weights
        self.price_cnn_weight = cfg.model.price_cnn_weight
        self.price_ridge_weight = cfg.model.ridge_weight
        self.direction_cnn_weight = cfg.model.direction_cnn_weight
        self.direction_elasticNet_weight = cfg.model.elasticNet_weight
        
        # Loss weights
        self.price_loss_weight = cfg.model.get('price_loss_weight', 0.5)
        self.direction_loss_weight = cfg.model.get('direction_loss_weight', 0.5)
        
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
        self.direction_acc_train = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.direction_f1_train = F1Score(task="multiclass", num_classes=self.num_classes, average='weighted')
        self.direction_acc_val = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.direction_f1_val = F1Score(task="multiclass", num_classes=self.num_classes, average='weighted')
        
        # Flags to track if models have been fitted
        self.ridge_fitted = False
        self.elasticNet_fitted = False
        
        self.direction_threshold = cfg.model.direction_threshold
    
    def orthogonal_loss(self, price_features, direction_features):
        """
        Penalizes cosine similarity between feature vectors to encourage orthogonality.
        Args:
            feat_price (torch.Tensor): Batch of feature vectors for price (shape: [batch_size, feature_dim]).
            feat_direction (torch.Tensor): Batch of feature vectors for direction (shape: [batch_size, feature_dim]).
        Returns:
            torch.Tensor: Mean squared cosine similarity loss (scalar).
        """
        price_features = F.normalize(price_features, dim=-1)
        direction_features = F.normalize(direction_features, dim=-1)
        cos = torch.sum(price_features * direction_features, dim=-1)
        return torch.mean(cos ** 2)
    
    def build_meta_features(self, x, cnn_price, cnn_dir_logits):
        """
        Returns:
            z_price: (B, in_dim_price)
            z_dir:   (B, in_dim_dir)
        """
        B = cnn_price.shape[0]
        device = cnn_price.device
            
        # CNN direction probs
        cnn_dir_probs = F.softmax(cnn_dir_logits, dim=1)
            
        # Ridge price
        if self.ridge_fitted and self.ridge.is_fitted:
            ridge_price = self.ridge(x)
            ridge_price = ridge_price.view(-1, 1)
        else:
            ridge_price = torch.zeros(B, 1, device=device)
            
        # ElasticNet logits -> probs
        if self.elasticNet_fitted and self.elasticNet.is_fitted:
            en_logits = self.elasticNet(x)
            en_probs = F.softmax(en_logits, dim=1)
        else:
            en_probs = torch.zeros(B, self.num_classes, device=device)
        
        # Assemble meta inputs
        z_price = torch.cat([
            cnn_price.view(-1, 1),     # (B,1)
            ridge_price,               # (B,1) or zeros
            cnn_dir_probs,             # (B,C)
            en_probs                   # (B,C) or zeros
        ], dim=1)
            
        z_dir = torch.cat([
            cnn_price.view(-1, 1),
            ridge_price,
            cnn_dir_probs,
            en_probs
        ], dim=1)
        
        return z_price, z_dir
    
    def on_train_start(self):
        logger.info("on_train_start: Fitting Ridge and ElasticNet on all training data...")
        """Fit Ridge and ElasticNet on all training data at the start of training"""
        
        if not (self.ridge_fitted and self.elasticNet_fitted):
            logger.info("Fitting Ridge and ElasticNet on all training data...")
            train_dataloader = self.trainer.datamodule.train_dataloader()
            
            device = next(self.parameters()).device
            x_all = []
            price_y_all = []
            direction_y_all = []
            
            with torch.no_grad():
                for batch in train_dataloader:
                    x, price_y, direction_y = batch
                    x_all.append(x)
                    price_y_all.append(price_y.to(device))
                    direction_y_all.append(direction_y.to(device))
                    
                x_all = torch.cat(x_all, dim=0).to(device)
                price_y_all = torch.cat(price_y_all, dim=0).to(device)
                direction_y_all = torch.cat(direction_y_all, dim=0).to(device)
                
            if not self.ridge_fitted:
                self.ridge.fit(x_all, price_y_all)
                self.ridge_fitted = True
                print(f"Ridge fitted on {len(price_y_all)} samples!")
                
            if not self.elasticNet_fitted:
                self.elasticNet.fit(x_all, direction_y_all)
                self.elasticNet_fitted = True
                print(f"ElasticNet fitted on {len(direction_y_all)} samples!")
                
    
    def forward(self, x):
        """Forward pass combining all model predictions"""
        logger.debug("MultiHeadEnsemble forward pass with input shape: %s", x.shape)
        
        # CNN predictions (both heads)
        cnn_price, cnn_direction_logits, price_features, direction_features = self.cnn(x)
        
        if self.use_meta_learning:
            z_price, z_dir = self.build_meta_features(x, cnn_price, cnn_direction_logits)
            
            meta_price = self.meta_price(z_price)
            meta_direction_logits = self.meta_dir(z_dir)
            
            final_price = torch.clamp(meta_price, min=-1e6, max=1e6)
            final_direction = meta_direction_logits
        else:
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
            elasticNet_direction = None
            if self.elasticNet_fitted and self.elasticNet.is_fitted:
                try:
                    elasticNet_direction = self.elasticNet(x)
                    if elasticNet_direction.device != cnn_direction_logits.device:
                        elasticNet_direction = elasticNet_direction.to(cnn_direction_logits.device)
                    if elasticNet_direction.shape != cnn_direction_logits.shape:
                        elasticNet_direction = elasticNet_direction.view_as(cnn_direction_logits)
                except Exception as e:
                    logger.warning(f"ElasticNet prediction failed: {e}, using CNN only")
                    elasticNet_direction = None
            
            # Combine predictions
            if ridge_price is not None:
                final_price = (self.price_cnn_weight * cnn_price + 
                            self.price_ridge_weight * ridge_price)
            else:
                final_price = cnn_price
            
            if elasticNet_direction is not None:
                final_direction = (self.direction_cnn_weight * cnn_direction_logits + 
                                self.direction_elasticNet_weight * elasticNet_direction)
            else:
                final_direction = cnn_direction_logits
            
            # Clamp price predictions to prevent extreme values
            final_price = torch.clamp(final_price, min=-1e6, max=1e6)
        
        return final_price, final_direction, price_features, direction_features
        
    def training_step(self, batch, batch_idx):
        logger.debug(f"Training step: batch_idx={batch_idx}")
        
        x, price_y, direction_y = batch
        
        price_pred, direction_pred, price_features, direction_features = self(x)
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self(x)

        
        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)
        
        meta_price_loss = self.price_criterion(price_pred, price_y)
        meta_direction_loss = self.direction_criterion(direction_pred, direction_y)
        
        base_losses = 0.0
        if self.include_base_losses:
            if cnn_price.dim() != price_y.dim():
                cnn_price = cnn_price.view_as(price_y)
            base_price_loss = self.price_criterion(cnn_price, price_y)
            base_direction_loss = self.direction_criterion(cnn_direction_logits, direction_y)
            base_losses = self.price_loss_weight * base_price_loss + self.direction_loss_weight * base_direction_loss
        
        orthogonal_penalty = self.orthogonal_loss(price_features, direction_features)
        
        total_loss = (self.price_loss_weight * meta_price_loss + 
                        self.direction_loss_weight * meta_direction_loss + base_losses +
                        orthogonal_penalty * self.cfg.model.orthogonal_lambda
                    )
        
        self.mae_train.update(price_pred, price_y)
        self.rmse_train.update(price_pred, price_y)
        self.r2_train.update(price_pred, price_y)
        
        direction_preds_class = torch.argmax(direction_pred, dim=1)
        self.direction_acc_train.update(direction_preds_class, direction_y)
        self.direction_f1_train.update(direction_preds_class, direction_y)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_price_loss', meta_price_loss, on_step=True, on_epoch=True)
        self.log('train_direction_loss', meta_direction_loss, on_step=True, on_epoch=True)
        if self.include_base_losses:
            self.log('train_base_losses', base_losses, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step: batch_idx={batch_idx}")
        
        x, price_y, direction_y = batch
        price_pred, direction_pred, price_features, direction_features = self(x)
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self(x)
        
        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)
        
        meta_price_loss = self.price_criterion(price_pred, price_y)
        meta_direction_loss = self.direction_criterion(direction_pred, direction_y)
        
        base_losses = 0.0
        if self.include_base_losses:
            if cnn_price.dim() != price_y.dim():
                cnn_price = cnn_price.view_as(price_y)
            base_price_loss = self.price_criterion(cnn_price, price_y)
            base_direction_loss = self.direction_criterion(cnn_direction_logits, direction_y)
            base_losses = self.price_loss_weight * base_price_loss + self.direction_loss_weight * base_direction_loss
        
        orthogonal_penalty = self.orthogonal_loss(price_features, direction_features)
        
        total_loss = (self.price_loss_weight * meta_price_loss + 
                        self.direction_loss_weight * meta_direction_loss + base_losses +
                        orthogonal_penalty * self.cfg.model.orthogonal_lambda
                    )
        
        self.mae_val.update(price_pred, price_y)
        self.rmse_val.update(price_pred, price_y)
        self.r2_val.update(price_pred, price_y)
        
        direction_preds_class = torch.argmax(direction_pred, dim=1)
        self.direction_acc_val.update(direction_preds_class, direction_y)
        self.direction_f1_val.update(direction_preds_class, direction_y)
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_price_loss', meta_price_loss, on_step=False, on_epoch=True)
        self.log('val_direction_loss', meta_direction_loss, on_step=False, on_epoch=True)
        if self.include_base_losses:
            self.log('val_base_losses', base_losses, on_step=False, on_epoch=True)
        
        return total_loss
        
    def on_train_epoch_end(self):
        logger.info("Training epoch ended. Computing training metrics.")
        
        # Price metrics
        price_mae = self.mae_train.compute()
        price_rmse = torch.sqrt(self.rmse_train.compute())
        price_r2 = self.r2_train.compute()
        
        # Direction metrics
        direction_acc = self.direction_acc_train.compute()
        direction_f1 = self.direction_f1_train.compute()
        
        # Log metrics
        self.log('train_price_mae', price_mae, prog_bar=True)
        self.log('train_price_rmse', price_rmse, prog_bar=True)
        self.log('train_price_r2', price_r2, prog_bar=True)
        self.log('train_direction_acc', direction_acc, prog_bar=True)
        self.log('train_direction_f1', direction_f1, prog_bar=True)
        
        meta_status = "enabled" if self.use_meta_learning else "disabled"
        
        logger.info(f"\n --------- Training Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        logger.info(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        logger.info(f"Models Fitted - Ridge: {self.ridge_fitted}, ElasticNet: {self.elasticNet_fitted}")
        logger.info(f"--------- Training Complete ---------\n")
        
        # Reset metrics
        self.mae_train.reset()
        self.rmse_train.reset()
        self.r2_train.reset()
        self.direction_acc_train.reset()
        self.direction_f1_train.reset()
    
    def on_validation_epoch_end(self):
        logger.info("Validation epoch ended. Computing validation metrics.")
        
        # Price metrics
        price_mae = self.mae_val.compute()
        price_rmse = torch.sqrt(self.rmse_val.compute())
        price_r2 = self.r2_val.compute()
        
        # Direction metrics
        direction_acc = self.direction_acc_val.compute()
        direction_f1 = self.direction_f1_val.compute()
        
        # Log metrics
        self.log('val_price_mae', price_mae, prog_bar=True)
        self.log('val_price_rmse', price_rmse, prog_bar=True)
        self.log('val_price_r2', price_r2, prog_bar=True)
        self.log('val_direction_acc', direction_acc, prog_bar=True)
        self.log('val_direction_f1', direction_f1, prog_bar=True)
        
        meta_status = "enabled" if self.use_meta_learning else "disabled"
        
        logger.info(f"\n --------- Validation Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        logger.info(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        logger.info(f"--------- Validation Complete ---------\n")
        
        # Reset metrics
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
        self.direction_acc_val.reset()
        self.direction_f1_val.reset()
    
    def test_step(self, batch, batch_idx):
        logger.debug("Test step: batch_idx=%d", batch_idx)
        return self.validation_step(batch, batch_idx)  # Reuse validation logic
    
    def on_test_epoch_end(self):
        logger.info("Test epoch ended. Computing test metrics.")
        
        # Price metrics
        price_mae = self.mae_val.compute()
        price_rmse = torch.sqrt(self.rmse_val.compute())
        price_r2 = self.r2_val.compute()
        
        # Direction metrics
        direction_acc = self.direction_acc_val.compute()
        direction_f1 = self.direction_f1_val.compute()
        
        # Log metrics
        self.log('val_price_mae', price_mae, prog_bar=True)
        self.log('val_price_rmse', price_rmse, prog_bar=True)
        self.log('val_price_r2', price_r2, prog_bar=True)
        self.log('val_direction_acc', direction_acc, prog_bar=True)
        self.log('val_direction_f1', direction_f1, prog_bar=True)
        
        meta_status = "enabled" if self.use_meta_learning else "disabled"
        
        logger.info(f"\n --------- Test Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        logger.info(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        logger.info(f"--------- Test Complete ---------\n")
        
        # Reset metrics
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
        self.direction_acc_val.reset()
        self.direction_f1_val.reset()
    
    def get_base_predictions(self, x):
        """
        Get individual predictions from all base models.
        Useful for analysis and debugging.
        """
        with torch.no_grad():
            # CNN predictions
            cnn_price, cnn_direction_logits, price_features, direction_features = self.cnn(x)
            cnn_direction_probs = F.softmax(cnn_direction_logits, dim=1)
            
            # Ridge predictions
            ridge_price = None
            if self.ridge_fitted and self.ridge.is_fitted:
                ridge_price = self.ridge(x)
            
            # ElasticNet predictions
            elasticNet_logits = None
            elasticNet_probs = None
            if self.elasticNet_fitted and self.elasticNet.is_fitted:
                elasticNet_logits = self.elasticNet(x)
                elasticNet_probs = F.softmax(elasticNet_logits, dim=1)
            
            return {
                'cnn_price': cnn_price,
                'cnn_direction_logits': cnn_direction_logits,
                'cnn_direction_probs': cnn_direction_probs,
                'ridge_price': ridge_price,
                'elasticNet_logits': elasticNet_logits,
                'elasticNet_probs': elasticNet_probs,
                'price_features': price_features,
                'direction_features': direction_features
            }
    
    def get_meta_predictions(self, x):
        """
        Get meta-learner predictions.
        """
        if not self.use_meta_learning:
            raise ValueError("Meta-learning is disabled. Use get_base_predictions() instead.")
        
        with torch.no_grad():
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
            z_price, z_dir = self.build_meta_features(x, cnn_price, cnn_direction_logits)
            
            meta_price = self.meta_price(z_price)
            meta_direction_logits = self.meta_dir(z_dir)
            meta_direction_probs = F.softmax(meta_direction_logits, dim=1)
            
            return {
                'meta_price': meta_price,
                'meta_direction_logits': meta_direction_logits,
                'meta_direction_probs': meta_direction_probs,
                'meta_features_price': z_price,
                'meta_features_direction': z_dir
            }
    
    def save_components(self):
        logger.info("Saving CNN and Elasticnet model components to disk.")
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        cnnPath = Path(repo_root / self.cfg.model.cnnPath).resolve()
        elasticNetPath = Path(repo_root / self.cfg.model.elasticNetPath).resolve()
        ridgePath = Path(repo_root / self.cfg.model.ridgePath).resolve()
        meta_price_path = Path(repo_root / self.cfg.model.meta_price_path).resolve()
        meta_direction_path = Path(repo_root / self.cfg.model.meta_direction_path).resolve()
        
        for path in [cnnPath, elasticNetPath, ridgePath, meta_price_path, meta_direction_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.cnn.state_dict(), cnnPath)
        torch.save(self.elasticNet.state_dict(), elasticNetPath)
        torch.save(self.ridge.state_dict(), ridgePath)
        torch.save(self.meta_price.state_dict(), meta_price_path)
        torch.save(self.meta_direction.state_dict(), meta_direction_path)
        print(f"CNN model saved to {cnnPath}.")
        print(f"Ridge model saved to {ridgePath}, ElasticNet model saved to {elasticNetPath}.")
        print(f"Meta price model saved to {meta_price_path}, Meta direction model saved to {meta_direction_path}.")
    
    def configure_optimizers(self):
        logger.info("Configuring optimizers and learning rate scheduler.")
        
        # Separate parameter groups for different learning rates if desired
        base_params = list(self.cnn.parameters())
        meta_params = []
        
        if self.use_meta_learning:
            meta_params = list(self.meta_price.parameters()) + list(self.meta_dir.parameters())
        
        # You can set different learning rates for base and meta models
        base_lr = self.cfg.optimiser.base_lr
        meta_lr = self.cfg.optimiser.meta_lr
        
        param_groups = [{'params': base_params, 'lr': base_lr}]
        
        if meta_params:
            param_groups.append({'params': meta_params, 'lr': meta_lr})
            
        optimizer = torch.optim.Adam(
            param_groups, 
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
        
        gradient_clip_val = self.cfg.optimiser.gradient_clip_val
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
            'gradient_clip_val': gradient_clip_val
        }