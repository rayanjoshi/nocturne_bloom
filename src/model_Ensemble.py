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
            nn.Linear(cnnChannels[2], 1)
        )
        
        num_classes = cfg.cnn.num_classes
        self.directionhead = nn.Sequential(
            nn.Linear(cnnChannels[1], cnnChannels[2]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[0]),
            nn.Linear(cnnChannels[2], num_classes)
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
        logger.info(f"Initializing RidgeRegressor with alpha={cfg.Ridge.alpha:.4f}, fit_intercept={cfg.Ridge.get('fit_intercept', True)}")
        super().__init__()
        self.alpha = cfg.Ridge.alpha
        self.fit_intercept = cfg.Ridge.get('fit_intercept', True)
        self.eps = cfg.Ridge.get('eps', 1e-8)
        
        # Ridge parameters (will be set during fit)
        self.register_buffer('weight', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('bias', torch.tensor(0.0, requires_grad=False))
        self.is_fitted = False
        
    def fit(self, X, y):
        logger.info("Fitting RidgeRegressor on data: X shape %s, y shape %s", X.shape, y.shape)
        
        device = X.device
        y = y.to(device).float()
        
        # Flatten X to 2D if it's 3D
        if len(X.shape) == 3:
            X = X.view(X.shape[0], -1)
        
        X = X.float()
        
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
        
        # Solve the system with better error handling
        try:
            L = torch.linalg.cholesky(XtX_reg)
            theta = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            try:
                theta = torch.linalg.solve(XtX_reg, Xty)
            except RuntimeError:
                logger.warning("Using pseudo-inverse due to singular matrix")
                theta = torch.linalg.pinv(XtX_reg) @ Xty
        
        # Split weight and bias
        if self.fit_intercept:
            self.weight.data = theta[:-1].clone().detach()
            self.bias.data = theta[-1].clone().detach()
        else:
            self.weight.data = theta.clone().detach()
            self.bias.data = torch.tensor(0.0, device=device)
        
        self.is_fitted = True
        
    def forward(self, x):
        logger.debug("RidgeRegressor forward pass with input shape: %s", x.shape)
        
        if not self.is_fitted:
            raise RuntimeError("Ridge model must be fitted before forward pass. Call .fit() first.")
        
        # Flatten to 2D if needed
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        
        x = x.to(self.weight.device).float()
        predictions = torch.mv(x, self.weight)
        
        if self.fit_intercept:
            predictions = predictions + self.bias
        
        return predictions

class ElasticNetClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        logger.info(f"Initializing ElasticNetClassifier with alpha={cfg.ElasticNet.alpha:.4f}, l1_ratio={cfg.ElasticNet.l1_ratio:.4f}, fit_intercept={cfg.ElasticNet.get('fit_intercept', True)}")
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
        X_standardised = X.float()
        
        if self.fit_intercept:
            y_mean = y_binary.float().mean()
            y_centred = y_binary.float() - y_mean
        else:
            y_centred = y_binary.float()
            y_mean = torch.tensor(0.0, device=device)
        
        beta = torch.zeros(numFeatures, device=device, dtype=X.dtype)
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1 - self.l1_ratio)
        
        # Just check and warn if zero variance exists
        X_var = torch.var(X_standardised, dim=0, unbiased=False)
        if (X_var < self.eps).any():
            logger.warning("Zero variance features detected")
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            beta_old = beta.clone()
            
            # Vectorized computation of residuals for all features
            residuals = y_centred - X_standardised @ beta
            
            # Vectorized computation of rho_j for all features
            rho = torch.matmul(X_standardised.t(), residuals) / numSamples
            
            denominator = 1.0 + (l2_reg / numSamples)
            beta_new = self.soft_threshold(rho, l1_reg / numSamples) / denominator
            
            # Clip beta to prevent explosion
            beta = torch.clamp(beta_new, min=-1e6, max=1e6)
            
            # Check for convergence
            if torch.norm(beta - beta_old) < self.tol:
                logger.info("Converged after %d iterations", iteration + 1)
                break
                
            if torch.abs(beta).max() > 1e5:
                logger.warning(f"Beta coefficients too large, using zero weights")
                weight = torch.zeros_like(beta)
                bias = torch.tensor(0.0, device=device)
                return weight, bias
        
        weight = beta
        if self.fit_intercept:
            bias = y_mean
        else:
            bias = torch.tensor(0.0, device=device)
            
        return weight, bias
    
    def fit(self, X, y):
        """
        Fit ElasticNet using one-vs-rest for multi-class classification
        """
        logger.info("Fitting ElasticNetClassifier on data: X shape %s, y shape %s", X.shape, y.shape)
        
        device = X.device
        y = y.to(device)
        
        if len(X.shape) == 3:
            X = X.view(X.shape[0], -1)
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        for class_idx in range(self.num_classes):
            y_binary = (y == class_idx).int()
            weight, bias = self.fit_single_class(X, y_binary, device)
            self.weights.append(nn.Parameter(weight.clone(), requires_grad=False))
            self.biases.append(nn.Parameter(bias.clone(), requires_grad=False))
        
        self.is_fitted = True
        logger.info(f"ElasticNetClassifier fitted successfully with {self.num_classes} classes")
        
    def forward(self, x):
        logger.debug(f"ElasticNetClassifier forward pass with input shape: {x.shape}")
        
        if not self.is_fitted:
            raise RuntimeError("ElasticNet classifier must be fitted before forward pass. Call .fit() first.")
        
        # Flatten to 2D if needed
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        
        device = self.weights[0].device
        x = x.to(device).float()
        
        scores = []
        for class_idx in range(self.num_classes):
            score = torch.mv(x, self.weights[class_idx])
            if self.fit_intercept:
                score = score + self.biases[class_idx]
            scores.append(score.unsqueeze(1))
        
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
    
    def forward(self, z):
        return self.linear(z).squeeze(-1)

class MetaDirectionClassifier(nn.Module):
    """
    Linear classifier over base outputs (CNN logits/probs, Elastic logits/probs, optional prices).
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    
    def forward(self, z):
        return self.linear(z)

class EnsembleModule(L.LightningModule):
    """
    Multi-headed CNN-Ridge-ElasticNet Ensemble Lightning Module
    - CNN provides both price and direction predictions
    - Ridge complements price predictions
    - ElasticNet complements direction predictions
    - Meta-learners to blend all predictions
    """
    def __init__(self, cfg: DictConfig):
        logger.info(f"Initializing EnsembleModule with config: {cfg.model}")
        super().__init__()
        self.cfg = cfg
        
        # NN components
        self.cnn = MultiHeadCNN(cfg)
        self.ridge = RidgeRegressor(cfg)
        self.elasticNet = ElasticNetClassifier(cfg)
        
        self.num_classes = cfg.cnn.num_classes
        
        # Meta-learning setup
        self.use_meta_learning = cfg.model.use_meta_learning
        if self.use_meta_learning:
            meta_price_dim = 3  # cnn_price + ridge_price + dir_entropy
            meta_dir_dim = 2 + 2 * self.num_classes  # cnn_price + ridge_price + cnn_dir_probs + en_probs
            self.meta_price = MetaPriceRegressor(in_dim=meta_price_dim)
            self.meta_dir = MetaDirectionClassifier(in_dim=meta_dir_dim, num_classes=self.num_classes)
        
        # Configuration
        self.include_base_losses = self.cfg.model.include_base_losses
        self.price_cnn_weight = cfg.model.price_cnn_weight
        self.price_ridge_weight = cfg.model.ridge_weight
        self.direction_cnn_weight = cfg.model.direction_cnn_weight
        self.direction_elasticNet_weight = cfg.model.elasticNet_weight
        self.price_loss_weight = cfg.model.price_loss_weight
        self.direction_loss_weight = cfg.model.direction_loss_weight
        
        # Loss functions
        self.price_criterion = nn.HuberLoss(delta=cfg.model.huber_delta)
        self.direction_criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self._setup_metrics()
        
        # Flags
        self.ridge_fitted = False
        self.elasticNet_fitted = False
        self.direction_threshold = cfg.model.direction_threshold
        
    def _setup_metrics(self):
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
            z_price: (B, 3)  # cnn_price, ridge_price, dir_entropy
            z_dir:   (B, 2 + 2 * num_classes)  # cnn_price, ridge_price, cnn_dir_probs, en_probs
        """
        B = cnn_price.shape[0]
        device = cnn_price.device
        
        cnn_dir_probs = F.softmax(cnn_dir_logits, dim=1)
        
        if self.ridge_fitted and self.ridge.is_fitted:
            ridge_price = self.ridge(x)
            ridge_price = ridge_price.view(-1, 1)
        else:
            ridge_price = torch.zeros(B, 1, device=device)
            
        if self.elasticNet_fitted and self.elasticNet.is_fitted:
            en_logits = self.elasticNet(x)
            en_probs = F.softmax(en_logits, dim=1)
        else:
            en_probs = torch.zeros(B, self.num_classes, device=device)
        
        # Assemble meta inputs
        dir_entropy = -torch.sum(cnn_dir_probs * torch.log(cnn_dir_probs + 1e-10), dim=1, keepdim=True)
        z_price = torch.cat([
            cnn_price.view(-1, 1),
            ridge_price,
            dir_entropy
        ], dim=1)
        
        z_dir = torch.cat([
            cnn_price.view(-1, 1),
            ridge_price,
            cnn_dir_probs,
            en_probs
        ], dim=1)
        
        return z_price, z_dir
    
    def forward(self, x):
        logger.debug("EnsembleModule forward pass with input shape: %s", x.shape)
        
        # Always get CNN predictions first
        cnn_price, cnn_direction_logits, price_features, direction_features = self.cnn(x)
        
        if self.use_meta_learning:
            z_price, z_dir = self.build_meta_features(x, cnn_price, cnn_direction_logits)
            
            meta_price = self.meta_price(z_price)
            meta_direction_logits = self.meta_dir(z_dir)
            
            final_price = torch.clamp(meta_price, min=-1e6, max=1e6)
            final_direction = meta_direction_logits
        else:
            # Traditional ensemble
            ridge_price = None
            if self.ridge_fitted and self.ridge.is_fitted:
                try:
                    ridge_price = self.ridge(x)
                    if ridge_price.device != cnn_price.device:
                        ridge_price = ridge_price.to(cnn_price.device)
                    if ridge_price.shape != cnn_price.shape:
                        ridge_price = ridge_price.view_as(cnn_price)
                except Exception as e:
                    logger.warning(f"Ridge prediction failed: {e}")
                    ridge_price = None
            
            elasticNet_direction = None
            if self.elasticNet_fitted and self.elasticNet.is_fitted:
                try:
                    elasticNet_direction = self.elasticNet(x)
                    if elasticNet_direction.device != cnn_direction_logits.device:
                        elasticNet_direction = elasticNet_direction.to(cnn_direction_logits.device)
                    if elasticNet_direction.shape != cnn_direction_logits.shape:
                        elasticNet_direction = elasticNet_direction.view_as(cnn_direction_logits)
                except Exception as e:
                    logger.warning(f"ElasticNet prediction failed: {e}")
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
        
        # Clamp predictions
        final_price = torch.clamp(final_price, min=-1e6, max=1e6)
        
        return final_price, final_direction, price_features, direction_features
    
    def on_train_start(self):
        logger.info("on_train_start: Fitting Ridge and ElasticNet on training data...")
        
        if not (self.ridge_fitted and self.elasticNet_fitted):
            train_dataloader = self.trainer.datamodule.train_dataloader()
            device = next(self.parameters()).device
            
            x_all = []
            price_y_all = []
            direction_y_all = []
            
            with torch.no_grad():
                for batch in train_dataloader:
                    x, price_y, direction_y = batch
                    x_all.append(x.to(device))
                    price_y_all.append(price_y.to(device))
                    direction_y_all.append(direction_y.to(device))
                    
                x_all = torch.cat(x_all, dim=0)
                price_y_all = torch.cat(price_y_all, dim=0)
                direction_y_all = torch.cat(direction_y_all, dim=0)
                
            if not self.ridge_fitted:
                self.ridge.fit(x_all, price_y_all)
                self.ridge_fitted = True
                logger.info(f"Ridge fitted on {len(price_y_all)} samples!")
                
            if not self.elasticNet_fitted:
                self.elasticNet.fit(x_all, direction_y_all)
                self.elasticNet_fitted = True
                logger.info(f"ElasticNet fitted on {len(direction_y_all)} samples!")
    
    def training_step(self, batch, batch_idx):
        logger.debug(f"Training step: batch_idx={batch_idx}")
        
        x, price_y, direction_y = batch
        
        # Always get final ensemble predictions
        price_pred, direction_pred, price_features, direction_features = self(x)
        
        # Ensure dimensions match
        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)
        
        # Calculate main losses
        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)
        
        # Calculate base losses if needed
        base_losses = 0.0
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
            if cnn_price.dim() != price_y.dim():
                cnn_price = cnn_price.view_as(price_y)
            base_price_loss = self.price_criterion(cnn_price, price_y)
            base_direction_loss = self.direction_criterion(cnn_direction_logits, direction_y)
            base_losses = (self.price_loss_weight * base_price_loss + 
                          self.direction_loss_weight * base_direction_loss)
        
        # Orthogonal penalty
        orthogonal_penalty = self.orthogonal_loss(price_features, direction_features)
        
        # Total loss
        total_loss = (self.price_loss_weight * price_loss + 
                     self.direction_loss_weight * direction_loss + base_losses +
                     orthogonal_penalty * self.cfg.model.orthogonal_lambda)
        
        # Update metrics
        self.mae_train.update(price_pred, price_y)
        self.rmse_train.update(price_pred, price_y)
        self.r2_train.update(price_pred, price_y)
        
        direction_preds_class = torch.argmax(direction_pred, dim=1)
        self.direction_acc_train.update(direction_preds_class, direction_y)
        self.direction_f1_train.update(direction_preds_class, direction_y)
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_price_loss', price_loss, on_step=True, on_epoch=True)
        self.log('train_direction_loss', direction_loss, on_step=True, on_epoch=True)
        if self.include_base_losses:
            self.log('train_base_losses', base_losses, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step: batch_idx={batch_idx}")
        
        x, price_y, direction_y = batch
        
        # Always get final ensemble predictions
        price_pred, direction_pred, price_features, direction_features = self(x)
        
        # Ensure dimensions match
        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)
        
        # Calculate main losses
        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)
        
        # Calculate base losses if needed
        base_losses = 0.0
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
            if cnn_price.dim() != price_y.dim():
                cnn_price = cnn_price.view_as(price_y)
            base_price_loss = self.price_criterion(cnn_price, price_y)
            base_direction_loss = self.direction_criterion(cnn_direction_logits, direction_y)
            base_losses = (self.price_loss_weight * base_price_loss + 
                          self.direction_loss_weight * base_direction_loss)
        
        # Orthogonal penalty
        orthogonal_penalty = self.orthogonal_loss(price_features, direction_features)
        
        # Total loss
        total_loss = (self.price_loss_weight * price_loss + 
                     self.direction_loss_weight * direction_loss + 
                        base_losses +
                     orthogonal_penalty * self.cfg.model.orthogonal_lambda)
        
        # Update metrics
        self.mae_val.update(price_pred, price_y)
        self.rmse_val.update(price_pred, price_y)
        self.r2_val.update(price_pred, price_y)
        
        direction_preds_class = torch.argmax(direction_pred, dim=1)
        self.direction_acc_val.update(direction_preds_class, direction_y)
        self.direction_f1_val.update(direction_preds_class, direction_y)
        
        # Log losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_price_loss', price_loss, on_step=False, on_epoch=True)
        self.log('val_direction_loss', direction_loss, on_step=False, on_epoch=True)
        if self.include_base_losses:
            self.log('val_base_losses', base_losses, on_step=False, on_epoch=True)
        
        return total_loss
    
    def on_train_epoch_end(self):
        logger.info("Training epoch ended. Computing training metrics.")
        
        price_mae = self.mae_train.compute()
        price_rmse = torch.sqrt(self.rmse_train.compute())
        price_r2 = self.r2_train.compute()
        direction_acc = self.direction_acc_train.compute()
        direction_f1 = self.direction_f1_train.compute()
        
        self.log('train_price_mae', price_mae, prog_bar=True)
        self.log('train_price_rmse', price_rmse, prog_bar=True)
        self.log('train_price_r2', price_r2, prog_bar=True)
        self.log('train_direction_acc', direction_acc, prog_bar=True)
        self.log('train_direction_f1', direction_f1, prog_bar=True)
        
        meta_status = "enabled" if self.use_meta_learning else "disabled"
        logger.info(f"--------- Training Results ---------")
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
        
        price_mae = self.mae_val.compute()
        price_rmse = torch.sqrt(self.rmse_val.compute())
        price_r2 = self.r2_val.compute()
        direction_acc = self.direction_acc_val.compute()
        direction_f1 = self.direction_f1_val.compute()
        
        self.log('val_price_mae', price_mae, prog_bar=True)
        self.log('val_price_rmse', price_rmse, prog_bar=True)
        self.log('val_price_r2', price_r2, prog_bar=True)
        self.log('val_direction_acc', direction_acc, prog_bar=True)
        self.log('val_direction_f1', direction_f1, prog_bar=True)
        
        meta_status = "enabled" if self.use_meta_learning else "disabled"
        logger.info(f"--------- Validation Results ---------")
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
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        logger.info("Test epoch ended. Computing test metrics.")
        
        price_mae = self.mae_val.compute()
        price_rmse = torch.sqrt(self.rmse_val.compute())
        price_r2 = self.r2_val.compute()
        direction_acc = self.direction_acc_val.compute()
        direction_f1 = self.direction_f1_val.compute()
        
        self.log('test_price_mae', price_mae, prog_bar=True)
        self.log('test_price_rmse', price_rmse, prog_bar=True)
        self.log('test_price_r2', price_r2, prog_bar=True)
        self.log('test_direction_acc', direction_acc, prog_bar=True)
        self.log('test_direction_f1', direction_f1, prog_bar=True)
        
        meta_status = "enabled" if self.use_meta_learning else "disabled"
        logger.info(f"--------- Test Results ---------")
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
                try:
                    ridge_price = self.ridge(x)
                except Exception as e:
                    logger.warning(f"Ridge prediction failed: {e}")
            
            # ElasticNet predictions
            elasticNet_logits = None
            elasticNet_probs = None
            if self.elasticNet_fitted and self.elasticNet.is_fitted:
                try:
                    elasticNet_logits = self.elasticNet(x)
                    elasticNet_probs = F.softmax(elasticNet_logits, dim=1)
                except Exception as e:
                    logger.warning(f"ElasticNet prediction failed: {e}")
            
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
        logger.info("Saving model components to disk.")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        
        # Define paths
        cnnPath = Path(repo_root / self.cfg.model.cnnPath).resolve()
        elasticNetPath = Path(repo_root / self.cfg.model.elasticNetPath).resolve()
        ridgePath = Path(repo_root / self.cfg.model.ridgePath).resolve()
        
        # Create directories
        for path in [cnnPath, elasticNetPath, ridgePath]:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(self.cnn.state_dict(), cnnPath)
        torch.save(self.elasticNet.state_dict(), elasticNetPath)
        torch.save(self.ridge.state_dict(), ridgePath)
        
        # Save meta-learners if they exist
        if self.use_meta_learning:
            meta_price_path = Path(repo_root / self.cfg.model.meta_price_path).resolve()
            meta_direction_path = Path(repo_root / self.cfg.model.meta_direction_path).resolve()
            
            meta_price_path.parent.mkdir(parents=True, exist_ok=True)
            meta_direction_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(self.meta_price.state_dict(), meta_price_path)
            torch.save(self.meta_dir.state_dict(), meta_direction_path)
            
            logger.info(f"Meta models saved to {meta_price_path} and {meta_direction_path}")
        
        logger.info(f"CNN saved to {cnnPath}")
        logger.info(f"Ridge saved to {ridgePath}")
        logger.info(f"ElasticNet saved to {elasticNetPath}")
    
    def configure_optimizers(self):
        logger.info("Configuring optimizers and learning rate scheduler.")
        
        # Separate parameter groups for different learning rates
        base_params = list(self.cnn.parameters())
        meta_params = []
        
        if self.use_meta_learning:
            meta_params = list(self.meta_price.parameters()) + list(self.meta_dir.parameters())
        
        # Set learning rates
        base_lr = self.cfg.optimiser.base_lr
        param_groups = [{'params': base_params, 'lr': base_lr}]
        
        if meta_params:
            meta_lr = self.cfg.optimiser.meta_lr
            param_groups.append({'params': meta_params, 'lr': meta_lr})
            
        optimizer = torch.optim.Adam(
            param_groups, 
            weight_decay=self.cfg.optimiser.weightDecay,
            eps=self.cfg.optimiser.eps
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