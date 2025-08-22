"""Ensemble module for time series prediction with CNN, Ridge, and LSTM models.

This module implements a PyTorch Lightning-based ensemble model that combines
a multi-head 1D Convolutional Neural Network (CNN), a Ridge regression model,
and a bidirectional Long Short-Term Memory (LSTM) classifier for joint price
regression and market direction classification tasks. It supports optional
meta-learners to fuse base model predictions and incorporates custom loss
functions like Focal Loss and Orthogonal Loss to handle class imbalance and
feature orthogonality.

The module is designed for time series data, leveraging 1D convolutions for
feature extraction, Ridge regression for linear corrections, and LSTMs for
sequential modeling. It includes comprehensive logging, metric tracking, and
model checkpointing.

Key components:
- `MultiHeadCNN`: A 1D CNN with separate heads for price regression and
    direction classification.
- `RidgeRegressor`: A Ridge regression model with SVD-based fitting.
- `LSTMClassifier`: A bidirectional LSTM for direction classification.
- `MetaPriceRegressor` and `MetaDirectionClassifier`: Optional meta-learners
    for ensembling base model outputs.
- `FocalLoss`: A class-balanced loss function focusing on hard examples.
- `OrthogonalLoss`: A penalty to encourage feature orthogonality.
- `EnsembleModule`: The main PyTorch Lightning module orchestrating the ensemble.

The module adheres to PEP 8 style guidelines and PEP 257 docstring conventions
for improved readability and maintainability.

Dependencies:
    - torch: Core PyTorch library for tensor operations.
    - torch.nn: Neural network modules.
    - torch.nn.functional: Functional operations for neural networks.
    - torchmetrics: Metrics for evaluation (MAE, RMSE, R2, Accuracy, F1).
    - lightning: PyTorch Lightning for scalable training.
    - omegaconf: For configuration management.
    - scripts.logging_config: Custom logging utilities.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Accuracy, F1Score
from torchmetrics.regression import R2Score
import lightning as L
from omegaconf import DictConfig
from scripts.logging_config import get_logger, setup_logging

setup_logging(log_level="INFO", console_output=True, file_output=True)
logger = get_logger("model_ensemble")

class MultiHeadCNN(nn.Module):
    """
    A multi-head 1D CNN model for joint regression and classification tasks.
    
    This model processes input time series using stacked 1D convolutional layers 
    followed by two separate heads:
        - A regression head for predicting price.
        - A classification head for predicting market direction.
    
    The architecture uses batch normalization, dropout, and ReLU activations 
    to stabilize training and reduce overfitting.
    
    Args:
        cfg (DictConfig): Configuration object specifying CNN architecture parameters,
            including input channels, number of classes, layer sizes, dropout rates, 
            kernel sizes, paddings, and strides.
    """
    def __init__(self, cfg: DictConfig):
        logger.info(f"Initializing CNN model with config: {cfg.cnn}")
        super().__init__()
        input_channels = cfg.cnn.inputChannels
        cnn_channels = [cfg.cnn.cnnChannels[0], cfg.cnn.cnnChannels[1], cfg.cnn.cnnChannels[2]]

        self.cnn = nn.Sequential(
            nn.Conv1d(
            input_channels,
            cnn_channels[0],
            kernel_size=cfg.cnn.kernelSize[0],
            padding=cfg.cnn.padding[0],
            ),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
            kernel_size=cfg.cnn.poolSize[0],
            stride=cfg.cnn.stride,
            padding=cfg.cnn.poolPadding[0],
            ),
            nn.Dropout(cfg.cnn.dropout[0] * 0.5),

            nn.Conv1d(
            cnn_channels[0],
            cnn_channels[1],
            kernel_size=cfg.cnn.kernelSize[1],
            padding=cfg.cnn.padding[1],
            ),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
            kernel_size=cfg.cnn.poolSize[1],
            stride=cfg.cnn.stride,
            padding=cfg.cnn.poolPadding[1],
            ),
            nn.Dropout(cfg.cnn.dropout[0]),

            nn.Conv1d(
            cnn_channels[1],
            cnn_channels[2],
            kernel_size=cfg.cnn.kernelSize[2],
            padding=cfg.cnn.padding[2],
            ),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(inplace=True),

            nn.Conv1d(
            cnn_channels[2],
            cnn_channels[2],
            kernel_size=cfg.cnn.kernelSize[2],
            padding=cfg.cnn.padding[2],
            ),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(cfg.cnn.dropout[0]),
        )

        self.price_features = nn.Sequential(
            nn.Linear(cnn_channels[2], cnn_channels[1]),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[1]),
        )

        self.direction_features = nn.Sequential(
            nn.Linear(cnn_channels[2], cnn_channels[1]),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[1] * 0.5),

            nn.Linear(cnn_channels[1], cnn_channels[1] * 2),
            nn.BatchNorm1d(cnn_channels[1] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[1]),

            nn.Linear(cnn_channels[1] * 2, cnn_channels[1]),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(inplace=True),
        )

        self.pricehead = nn.Sequential(
            nn.Linear(cnn_channels[1], cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0]),
            nn.Linear(cnn_channels[2], 1)
        )

        num_classes = cfg.cnn.num_classes
        self.directionhead = nn.Sequential(
            nn.Linear(cnn_channels[1], cnn_channels[2] * 2),
            nn.BatchNorm1d(cnn_channels[2] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0] * 0.5),

            nn.Linear(cnn_channels[2] * 2, cnn_channels[2] * 2),
            nn.BatchNorm1d(cnn_channels[2] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0]),

            nn.Linear(cnn_channels[2] * 2, cnn_channels[2]),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0]),

            nn.Linear(cnn_channels[2], num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of convolutional, linear, and batch norm layers.
        
        - Convolutional and non-head linear layers use Kaiming initialization.
        - Classification head linear layers use Xavier initialization.
        - BatchNorm layers are initialized to preserve scale and shift.
        
        This promotes better convergence during training.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Check if the module is part of self.directionhead
                if m in self.directionhead.modules():
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Perform the forward pass of the MultiHeadCNN.

        Applies convolutional layers to extract features, then splits into
        two heads: one for price regression and another for direction
        classification.

        Residual connections and feature alignment are applied to improve
        directional representations.

        Args:
            x (torch.Tensor): Input tensor of shape
            (batch_size, time_steps, input_channels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - price_pred (torch.Tensor): Predicted price, shape
                (batch_size,).
            - direction_pred (torch.Tensor): Class logits, shape
                (batch_size, num_classes).
            - price_features (torch.Tensor): Features used for price.
            - direction_features (torch.Tensor): Features used for
                classification.
        """
        logger.debug(f"MultiHeadCNN forward pass with input shape: {x.shape}")
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.squeeze(-1)

        price_features = self.price_features(x)
        direction_features_raw = self.direction_features(x)

        # Add pooled CNN activations to learned direction features.
        # Project pooled activations to match direction_features_raw dim when necessary.
        pooled = F.adaptive_avg_pool1d(x.unsqueeze(-1), 1).squeeze(-1)  # pylint: disable=not-callable
        if pooled.size(1) != direction_features_raw.size(1):
            pooled = F.linear(pooled,  # pylint: disable=not-callable
                        torch.eye(direction_features_raw.size(1), pooled.size(1),
                        device=pooled.device, dtype=pooled.dtype))

        direction_features = direction_features_raw + pooled

        # Ensure direction_features and price_features have same dim; project if needed
        if direction_features.size(1) != price_features.size(1):
            # Project direction_features to match price_features dim
            eye = torch.eye(
                price_features.size(1),
                direction_features.size(1),
                device=direction_features.device,
                dtype=direction_features.dtype,
            )
            direction_features = F.linear(direction_features, eye)  # pylint: disable=not-callable

        price_pred = self.pricehead(price_features).squeeze(-1)
        direction_pred = self.directionhead(direction_features_raw)

        return price_pred, direction_pred, price_features, direction_features

class RidgeRegressor(nn.Module):
    """
    Ridge regression model implemented using PyTorch.
    
    This class supports fitting a linear model with L2 regularization (Ridge)
    using either Singular Value Decomposition (SVD) or a normal equations fallback.
    It handles optional intercept fitting and can be used in GPU-accelerated pipelines.
    
    Args:
        cfg (DictConfig): Configuration object containing the following Ridge parameters:
            - alpha (float): Regularization strength.
            - fit_intercept (bool, optional): Whether to fit an intercept. Defaults to True.
            - eps (float, optional): Small value to prevent division by zero. Defaults to 1e-8.
    """
    def __init__(self, cfg: DictConfig):
        logger.info(
            "Initializing RidgeRegressor with alpha=%0.4f, fit_intercept=%s",
            cfg.Ridge.alpha,
            cfg.Ridge.get("fit_intercept", True),
        )
        super().__init__()
        self.alpha = cfg.Ridge.alpha
        self.fit_intercept = cfg.Ridge.get('fit_intercept', True)
        self.eps = cfg.Ridge.get('eps', 1e-8)

        # Ridge parameters (will be set during fit)
        self.register_buffer('weight', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('bias', torch.tensor(0.0, requires_grad=False))
        self.is_fitted = False

        self.fallback_count = 0  # Track fallback attempts

    def fit(self, x, y, rank=None):
        """
        Fit the Ridge regression model using input features and targets.

        Uses SVD for a numerically stable solution. Falls back to solving the
        regularized normal equations if SVD fails. Learns weights and an
        optional intercept.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, ...).
            y (torch.Tensor): Target values, shape (batch_size,).
            rank (int, optional): Truncate SVD to this rank for low-rank fit.

        Raises:
            RuntimeError: If fitting fails due to CUDA or numerical issues.
        """
        logger.info(f"Fitting RidgeRegressor on data: x shape {x.shape}, y shape {y.shape}")

        device = x.device
        y = y.to(device).float()

        # Flatten x to 2D if it's 3D
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)

        x = x.float()

        # Add intercept column if needed
        if self.fit_intercept:
            ones = torch.ones(x.shape[0], 1, device=device)
            x_with_intercept = torch.cat([x, ones], dim=1)
        else:
            x_with_intercept = x

        try:
            with torch.cuda.amp.autocast():
                u_matrix, singular_values, v_transpose = torch.linalg.svd(   # pylint: disable=not-callable
                    x_with_intercept, full_matrices=False
                )
                if rank is not None:
                    u_matrix = u_matrix[:, :rank]
                    singular_values = singular_values[:rank]
                    v_transpose = v_transpose[:rank, :]

                # Log condition number
                max_sv = torch.max(singular_values)
                min_sv = torch.min(singular_values)
                condition_number = max_sv / (min_sv + self.eps)
                logger.info(f"Matrix condition number: {condition_number:.4f}")

                # Ridge solution using SVD
                s_reg = singular_values / (singular_values**2 + self.alpha + self.eps)
                y_col = y.unsqueeze(1)
                u_ty = u_matrix.t() @ y_col
                s_reg_col = s_reg.unsqueeze(1)
                theta_col = v_transpose.t() @ (s_reg_col * u_ty)
                theta = theta_col.squeeze(1)

        except torch.cuda.CudaError as e:
            logger.error(f"GPU memory error during SVD: {e}")
            raise
        except RuntimeError as e:
            self.fallback_count = getattr(self, 'fallback_count', 0) + 1
            count = self.fallback_count
            logger.warning(
                f"SVD failed (count: {count}, error: {e}). "
                "Falling back to normal equations"
            )
            num_features = x_with_intercept.shape[1]
            identity = torch.eye(num_features, device=device)

            xtx_mat = torch.mm(x_with_intercept.t(), x_with_intercept)
            xtx_reg = xtx_mat + (self.alpha + self.eps) * identity
            x_ty = torch.mv(x_with_intercept.t(), y)

            try:
                theta = torch.linalg.solve(xtx_reg, x_ty)  # pylint: disable=not-callable
            except RuntimeError:
                logger.warning("Using pseudo-inverse due to singular matrix")
                theta = torch.linalg.pinv(xtx_reg) @ x_ty  # pylint: disable=not-callable

        # Validate coefficients
        if torch.any(torch.abs(theta) > 1e5):
            logger.warning(f"Large coefficients detected: max |theta| = {torch.abs(theta).max()}")

        # Split weight and bias
        if self.fit_intercept:
            self.weight.data = theta[:-1].clone().detach()
            self.bias.data = theta[-1].clone().detach()
        else:
            self.weight.data = theta.clone().detach()
            self.bias.data = torch.tensor(0.0, device=device)

        # Log training MAE
        with torch.no_grad():
            y_pred = torch.matmul(x_with_intercept, theta)
            mae = torch.mean(torch.abs(y_pred - y))
            logger.info(f"Training MAE after fitting: {mae:.4f}")

        self.is_fitted = True

    def forward(self, x):
        """
        Perform a forward pass using the fitted Ridge regression model.
        
        Predicts outputs using the learned weights (and optional intercept).
        Input is flattened to 2D if needed.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *) or (batch_size, features).
        
        Returns:
            torch.Tensor: Predicted values of shape (batch_size,).
        
        Raises:
            RuntimeError: If called before the model has been fitted using `.fit()`.
        """
        logger.debug(f"RidgeRegressor forward pass with input shape: {x.shape}")

        if not self.is_fitted:
            raise RuntimeError("Ridge model must be fitted before forward pass. Call .fit() first.")

        # Flatten to 2D if needed
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)

        x = x.to(self.weight.device).float()
        predictions = torch.matmul(x, self.weight)

        if self.fit_intercept:
            predictions = predictions + self.bias

        return predictions

class LSTMClassifier(nn.Module):
    """
    A bidirectional LSTM-based classifier for time series data.
    
    This model projects input features, applies a multi-layer bidirectional LSTM, 
    and feeds the output through a fully connected network to perform classification.
    
    Includes:
        - Input projection to match LSTM dimensionality.
        - Layer normalization and residual connections.
        - Xavier initialization for stability.
        - Final dense layers for class prediction.
    
    Args:
        cfg (DictConfig): Configuration object containing model parameters:
            - cfg.LSTM.hidden_size (int): Hidden dimension size of the LSTM.
            - cfg.LSTM.num_layers (int): Number of LSTM layers.
            - cfg.LSTM.dropout (float): Dropout rate.
            - cfg.cnn.inputChannels (int): Dimensionality of input features.
            - cfg.cnn.num_classes (int): Number of output classes.
    """
    def __init__(self, cfg: DictConfig):
        logger.info(
            "Initializing Improved LSTMClassifier with hidden_size=%s, num_layers=%s",
            cfg.LSTM.hidden_size,
            cfg.LSTM.num_layers,
        )
        super().__init__()
        self.input_size = cfg.cnn.inputChannels
        self.hidden_size = cfg.LSTM.hidden_size
        self.num_layers = cfg.LSTM.num_layers
        self.num_classes = cfg.cnn.num_classes
        self.dropout = cfg.LSTM.dropout

        # Input projection to align CNN features
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=True  # Enable bidirectional processing
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_size * 2)  # *2 for bidirectional

        # Fully connected classifier (simplified)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of all submodules in the network.
        
        - Linear and LSTM weights are initialized using Xavier uniform.
        - Biases are initialized to zero.
        - LayerNorm weights initialized to 1.0 and biases to 0.0.
        
        Improves training stability and convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Perform a forward pass through the LSTM classifier.
        
        Steps:
            1. Project input features to LSTM hidden size.
            2. Run input through a bidirectional LSTM.
            3. Apply layer normalization and a residual connection.
            4. Extract the last time step output.
            5. Feed through a fully connected classifier to get logits.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
        
        Returns:
            torch.Tensor: Output class logits of shape (batch_size, num_classes).
        """
        logger.debug(f"LSTMClassifier forward pass with input shape: {x.shape}")
        # Input shape: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Project input to hidden_size
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)

        # Initialize hidden state (*2 for bidirectional)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_size * 2)

        # Apply layer normalization
        out = self.norm(out)

        # Residual connection (project input to match LSTM output size)
        identity = torch.eye(
            self.hidden_size * 2,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype
        )
        residual = F.linear(x, identity)  # pylint: disable=not-callable
        out = out + residual  # (batch, seq_len, hidden_size * 2)

        # Take the last time step
        out = out[:, -1, :]  # (batch, hidden_size * 2)

        # Fully connected layer
        logits = self.fc(out)  # (batch, num_classes)

        return logits


class MetaPriceRegressor(nn.Module):
    """
    A linear regressor that combines outputs from multiple base models.
    
    This model acts as a learned ensemble for price prediction, typically
    blending features such as:
        - CNN price predictions
        - Ridge regression outputs
        - Directional classification probabilities
        - LSTM-based predictions
    
    Can behave like Ridge regression when combined with weight decay in the optimizer.
    
    Args:
        in_dim (int): Dimensionality of the input feature vector.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, z):
        """
        Forward pass for the price regressor.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, in_dim) containing
                features from base models.
        
        Returns:
            torch.Tensor: Predicted price values of shape (batch_size,).
        """
        return self.linear(z).squeeze(-1)

class MetaDirectionClassifier(nn.Module):
    """
    A linear classifier that ensembles multiple model outputs for direction prediction.
    
    Inputs may include:
        - CNN logits or probabilities
        - LSTM logits or probabilities
        - Price predictions (optional)
    
    Args:
        in_dim (int): Number of input features.
        num_classes (int): Number of output classes.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        """
        Forward pass for the direction classifier.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, in_dim).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.linear(z)

class FocalLoss(nn.Module):
    """
    Focal loss with class-balancing using effective number of samples.

    This loss reduces the relative loss for well-classified examples and focuses
    training on the hard examples. It also supports class imbalance via adaptive
    weighting.

    Args:
        num_classes (int): Number of output classes.
        class_counts (torch.Tensor, optional): Raw class counts used to compute
            balancing weights.
        gamma (float): Focusing parameter (default: 1.5).
        beta (float): Beta parameter for effective number calculation
            (default: 0.9999).
        reduction (str): Reduction method: 'mean', 'sum', or 'none'.
        eps (float): Small epsilon for numerical stability.
    """
    def __init__(self, num_classes: int, class_counts: torch.Tensor = None,
                    gamma: float = 1.5, beta: float = 0.9999,
                    reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        # Validate inputs
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        # Allow None class_counts (fallback to uniform)
        if class_counts is None:
            class_counts = torch.ones(num_classes, dtype=torch.float32)
        elif not isinstance(class_counts, torch.Tensor):
            class_counts = torch.tensor(class_counts, dtype=torch.float32)

        if len(class_counts) != num_classes:
            raise ValueError(
                f"class_counts length ({len(class_counts)}) must match "
                f"num_classes ({num_classes})"
            )
        # If counts are all zeros or negative values sneaked in, fallback to uniform
        if (class_counts <= 0).all():
            class_counts = torch.ones(num_classes, dtype=torch.float32)
        if (class_counts < 0).any():
            raise ValueError("class_counts must be non-negative")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

        # Calculate effective weights using effective number of samples
        effective_num = 1.0 - torch.pow(beta, class_counts)
        # Avoid divide-by-zero by clamping effective_num
        effective_num = torch.clamp(effective_num, min=eps)
        weights = (1.0 - beta) / (effective_num)
        weights = weights / (weights.sum() + eps) * num_classes

        # Log weights for debugging
        logger.info(f"Calculated class weights: {weights.tolist()}")

        # Register weights as buffer to ensure device consistency
        self.register_buffer('alpha', weights)
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss with balanced weights.
        
        Args:
            inputs (torch.Tensor): Raw logits of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size,).
        
        Returns:
            torch.Tensor: Computed focal loss based on reduction type.
        
        Raises:
        ValueError: If input shapes or target ranges are invalid.
        """
        # Validate inputs
        if inputs.dim() != 2 or inputs.size(1) != self.num_classes:
            raise ValueError(
                "Expected inputs of shape "
                f"(batch_size, {self.num_classes}), got {inputs.shape}"
            )
        if targets.dim() != 1 or targets.size(0) != inputs.size(0):
            raise ValueError(f"Expected targets of shape (batch_size,), got {targets.shape}")
        if (targets < 0).any() or (targets >= self.num_classes).any():
            raise ValueError(f"Targets must be in range [0, {self.num_classes-1}]")

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss).clamp(min=self.eps, max=1.0)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class OrthogonalLoss(nn.Module):
    """
    Encourages orthogonality between feature representations.
    
    Designed to reduce correlation between price and directional feature embeddings
    by penalizing cosine similarity.
    
    Args:
        lambda_ortho (float): Scaling factor for the orthogonality penalty (default: 1.0).
    """
    def __init__(self, lambda_ortho=1.0):
        super().__init__()
        self.lambda_ortho = lambda_ortho

    def forward(self, price_features, direction_features):
        """
        Compute the orthogonality penalty between two feature vectors.
        
        Args:
            price_features (torch.Tensor): Tensor of shape (batch_size, feature_dim).
            direction_features (torch.Tensor): Tensor of shape (batch_size, feature_dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - penalty (torch.Tensor): Scalar loss encouraging orthogonality.
                - cosine_sim (torch.Tensor): Tensor of cosine similarities per sample.
        """
        price_norm = F.normalize(price_features, p=2, dim=1, eps=1e-8)
        direction_norm = F.normalize(direction_features, p=2, dim=1, eps=1e-8)
        cosine_sim = torch.sum(price_norm * direction_norm, dim=1)
        penalty = torch.mean(torch.abs(cosine_sim))
        return self.lambda_ortho * penalty, cosine_sim

class EnsembleModule(L.LightningModule):
    """
    PyTorch Lightning module for an ensemble of CNN, Ridge, and LSTM models.

    Supports multi-headed outputs for price regression and direction
    classification. Optionally uses meta-learners to fuse base model
    predictions.

    Components:
        - cnn: MultiHeadCNN predicting price and direction.
        - ridge: RidgeRegressor for linear price correction.
        - lstm: LSTMClassifier for directional signals.
        - meta_price / meta_dir: Optional meta-learners when enabled.

    Attributes:
        cfg: Hydra DictConfig for model and optimizer settings.
        use_meta_learning: Bool indicating use of meta-learners.
        include_base_losses: Bool to include base-model losses in total.
        price_criterion: Loss used for price regression.
        direction_criterion: Loss used for classification.
        orthogonal_loss: Penalty to encourage orthogonality of features.
        ridge_fitted: Flag set after fitting ridge on training data.
        class_counts: Tensor of label frequencies for class weighting.
    """
    def __init__(self, cfg: DictConfig):
        logger.info(f"Initializing EnsembleModule with config: {cfg.model}")
        super().__init__()
        self.cfg = cfg

        # NN components
        self.cnn = MultiHeadCNN(cfg)
        self.ridge = RidgeRegressor(cfg)
        self.lstm = LSTMClassifier(cfg)

        self.num_classes = cfg.cnn.num_classes

        # Meta-learning setup
        self.use_meta_learning = cfg.model.use_meta_learning
        if self.use_meta_learning:
            meta_price_dim = 5
            meta_dir_dim = 2 * self.num_classes + 6
            self.meta_price = MetaPriceRegressor(in_dim=meta_price_dim)
            self.meta_dir = MetaDirectionClassifier(
                in_dim=meta_dir_dim,
                num_classes=self.num_classes,
            )

        # Configuration
        self.include_base_losses = self.cfg.model.include_base_losses
        self.price_cnn_weight = cfg.model.price_cnn_weight
        self.price_ridge_weight = cfg.model.ridge_weight
        self.direction_cnn_weight = cfg.model.direction_cnn_weight
        self.direction_lstm_weight = cfg.model.lstm_weight
        self.price_loss_weight = cfg.model.price_loss_weight
        self.direction_loss_weight = cfg.model.direction_loss_weight

        # Loss functions
        self.price_criterion = nn.HuberLoss(delta=cfg.model.huber_delta)
        self.direction_criterion = FocalLoss(num_classes=self.num_classes,
                                                class_counts=None,
                                                gamma=cfg.model.focal_gamma,
                                                beta=cfg.model.focal_beta,
                                                reduction='mean',
                                                eps=1e-8
                                                )
        self.orthogonal_loss = OrthogonalLoss(lambda_ortho=cfg.model.orthogonal_lambda)

        self.class_counts = None

        # Metrics
        self._setup_metrics()

        # Flags
        self.ridge_fitted = False

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
        self.direction_f1_train = F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average="weighted",
        )
        self.direction_acc_val = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.direction_f1_val = F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average='weighted')

    def build_meta_features(self, x, cnn_price, cnn_dir_logits):
        """
        Construct meta-features for meta-learners from base model predictions.
        
        Args:
            x (torch.Tensor): Input features.
            cnn_price (torch.Tensor): Price prediction from CNN.
            cnn_dir_logits (torch.Tensor): Direction logits from CNN.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Meta-features for price regression.
                - Meta-features for direction classification.
        """
        batch_size = cnn_price.shape[0]
        device = cnn_price.device

        cnn_dir_probs = F.softmax(cnn_dir_logits, dim=1)

        # Get base model predictions with error handling
        if self.ridge_fitted and self.ridge.is_fitted:
            try:
                ridge_price = self.ridge(x)
                if ridge_price.dim() > 1:
                    ridge_price = ridge_price.squeeze(-1)
            except (
                RuntimeError,
                ValueError,
                TypeError,
                IndexError,
                torch.cuda.OutOfMemoryError,
            ) as e:
                logger.warning(f"Ridge prediction failed in meta-features: {e}")
                ridge_price = torch.zeros(batch_size, device=device)
        else:
            ridge_price = torch.zeros(batch_size, device=device)

        lstm_logits = None
        try:
            lstm_logits = self.lstm(x)
            lstm_probs = F.softmax(lstm_logits, dim=1)
        except (RuntimeError, ValueError, TypeError, IndexError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"LSTM prediction failed in meta-features: {e}")
            lstm_probs = torch.zeros(batch_size, self.num_classes, device=device)

        dir_entropy = -torch.sum(cnn_dir_probs * torch.log(cnn_dir_probs + 1e-10), dim=1)

        non_zero_ridge = False
        if ridge_price is not None:
            non_zero_ridge = not torch.allclose(
            ridge_price, torch.zeros_like(ridge_price)
            )
        if non_zero_ridge:
            price_diff = torch.abs(cnn_price.squeeze() - ridge_price)
            price_mean = (cnn_price.squeeze() + ridge_price) / 2
            price_disagreement = price_diff / (torch.abs(price_mean) + 1e-6)
        else:
            price_disagreement = torch.zeros(batch_size, device=device)

        if not torch.allclose(lstm_probs, torch.zeros_like(lstm_probs)):
            dir_kl_div = torch.sum(cnn_dir_probs * torch.log(
                (cnn_dir_probs + 1e-10) / (lstm_probs + 1e-10)
            ), dim=1)
        else:
            dir_kl_div = torch.zeros(batch_size, device=device)

        cnn_confidence = torch.max(cnn_dir_probs, dim=1)[0]
        # Compute LSTM confidence, handle missing LSTM probs
        if torch.allclose(lstm_probs, torch.zeros_like(lstm_probs)):
            lstm_confidence = torch.zeros(batch_size, device=device)
        else:
            lstm_confidence = torch.max(lstm_probs, dim=1)[0]

        z_price = torch.stack([
            cnn_price.squeeze(),
            ridge_price,
            dir_entropy,
            price_disagreement,
            cnn_confidence
        ], dim=1)

        z_dir = torch.cat([
            cnn_price.squeeze().unsqueeze(1),
            ridge_price.unsqueeze(1),
            dir_entropy.unsqueeze(1),
            dir_kl_div.unsqueeze(1),
            cnn_confidence.unsqueeze(1),
            lstm_confidence.unsqueeze(1),
            cnn_dir_probs,
            lstm_probs
        ], dim=1)

        return z_price, z_dir

    def forward(self, *args, **kwargs):
        """
        Forward pass of the ensemble module.
        
        Depending on whether meta-learning is enabled, either base predictions are blended
        or meta-models are used for final outputs.
        
        Args:
            x (torch.Tensor): Input features.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Final price prediction.
                - Final direction logits.
                - Price feature vector.
                - Direction feature vector.
        """
        x = args[0]
        logger.debug(f"EnsembleModule forward pass with input shape: {x.shape}")

        cnn_price, cnn_direction_logits, price_features, direction_features = self.cnn(x)

        if self.use_meta_learning:
            z_price, z_dir = self.build_meta_features(x, cnn_price, cnn_direction_logits)

            meta_price = self.meta_price(z_price)
            meta_direction_logits = self.meta_dir(z_dir)

            final_price = torch.clamp(meta_price, min=-1e6, max=1e6)
            final_direction = meta_direction_logits
        else:
            ridge_price = None
            if self.ridge_fitted and self.ridge.is_fitted:
                try:
                    ridge_price = self.ridge(x)
                    if ridge_price.device != cnn_price.device:
                        ridge_price = ridge_price.to(cnn_price.device)
                    if ridge_price.shape != cnn_price.shape:
                        ridge_price = ridge_price.view_as(cnn_price)
                except (RuntimeError,
                        ValueError,
                        TypeError,
                        IndexError,
                        torch.cuda.OutOfMemoryError) as e:
                    logger.warning(f"Ridge prediction failed: {e}")
                    ridge_price = None

            lstm_direction = None
            try:
                lstm_direction = self.lstm(x)
                if lstm_direction.device != cnn_direction_logits.device:
                    lstm_direction = lstm_direction.to(cnn_direction_logits.device)
                if lstm_direction.shape != cnn_direction_logits.shape:
                    lstm_direction = lstm_direction.view_as(cnn_direction_logits)
            except (RuntimeError,
                    ValueError,
                    TypeError,
                    IndexError,
                    torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"LSTM prediction failed: {e}")
                lstm_direction = None

            if ridge_price is not None:
                final_price = (self.price_cnn_weight * cnn_price +
                             self.price_ridge_weight * ridge_price)
            else:
                final_price = cnn_price

            if lstm_direction is not None:
                final_direction = (self.direction_cnn_weight * cnn_direction_logits +
                                 self.direction_lstm_weight * lstm_direction)
            else:
                final_direction = cnn_direction_logits

        final_price = torch.clamp(final_price, min=-1e6, max=1e6)

        return final_price, final_direction, price_features, direction_features

    def on_train_start(self):
        """
        Operations to perform at the beginning of training.
        
        - Computes class frequencies for Focal Loss.
        - Fits the Ridge model on the full training set.
        """
        logger.info("Setting up training with class balancing...")

        train_dataloader = self.trainer.datamodule.train_dataloader()
        device = next(self.parameters()).device
        all_labels = []

        for batch in train_dataloader:
            _, _, direction_y = batch
            all_labels.append(direction_y.to(device))

        all_labels = torch.cat(all_labels, dim=0)
        unique, counts = torch.unique(all_labels, return_counts=True)
        self.class_counts = counts.to(device)

        logger.info(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
        logger.info(f"Class imbalance ratio: {float(counts.max() / counts.min()):.2f}")

        self.direction_criterion = FocalLoss(
            num_classes=self.num_classes,
            class_counts=self.class_counts,
            gamma=self.cfg.model.focal_gamma,
            beta=self.cfg.model.get('focal_beta', 0.9999),
            reduction='mean',
            eps=1e-8
        ).to(device)

        logger.info("Fitting Ridge on training data...")

        if not self.ridge_fitted:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            device = next(self.parameters()).device

            x_all = []
            price_y_all = []

            with torch.no_grad():
                for batch in train_dataloader:
                    x, price_y, _ = batch
                    x_all.append(x.to(device))
                    price_y_all.append(price_y.to(device))

                x_all = torch.cat(x_all, dim=0)
                price_y_all = torch.cat(price_y_all, dim=0)

            self.ridge.fit(x_all, price_y_all)
            self.ridge_fitted = True
            logger.info(f"Ridge fitted on {len(price_y_all)} samples!")

    def training_step(self, *args, **kwargs):
        """
        Perform a single training step.
        
        Computes predictions, losses, and updates metrics.
        
        Args:
            batch (Tuple[torch.Tensor]): A batch containing (x, price_y, direction_y).
            batch_idx (int): Index of the current batch.
        
        Returns:
            torch.Tensor: Total loss for the batch.
        """
        batch = args[0]
        batch_idx = args[1]

        logger.debug(f"Training step: batch_idx={batch_idx}")

        x, price_y, direction_y = batch

        price_pred, direction_pred, price_features, direction_features = self(x)

        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)

        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)

        base_losses = 0.0
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
            if cnn_price.dim() != price_y.dim():
                cnn_price = cnn_price.view_as(price_y)
            base_price_loss = self.price_criterion(cnn_price, price_y)
            base_direction_loss = self.direction_criterion(cnn_direction_logits, direction_y)
            base_losses = (self.price_loss_weight * base_price_loss +
                         self.direction_loss_weight * base_direction_loss)

        orthogonal_penalty, cosine_sim = self.orthogonal_loss(price_features, direction_features)

        total_loss = (self.price_loss_weight * price_loss +
                        self.direction_loss_weight * direction_loss +
                        base_losses + orthogonal_penalty
                    )

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
        self.log('train_cosine_sim', torch.mean(cosine_sim), on_step=True, on_epoch=True)
        if self.include_base_losses:
            self.log('train_base_losses', base_losses, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, *args, **kwargs):
        """
        Perform a single validation step.
        
        Args:
            batch (Tuple[torch.Tensor]): A batch containing (x, price_y, direction_y).
            batch_idx (int): Index of the current batch.
        
        Returns:
            torch.Tensor: Total validation loss.
        """
        batch = args[0]
        batch_idx = args[1]

        logger.debug(f"Validation step: batch_idx={batch_idx}")

        x, price_y, direction_y = batch

        price_pred, direction_pred, price_features, direction_features = self(x)

        if price_pred.dim() != price_y.dim():
            price_pred = price_pred.view_as(price_y)

        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)

        base_losses = 0.0
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
            if cnn_price.dim() != price_y.dim():
                cnn_price = cnn_price.view_as(price_y)
            base_price_loss = self.price_criterion(cnn_price, price_y)
            base_direction_loss = self.direction_criterion(cnn_direction_logits, direction_y)
            base_losses = (self.price_loss_weight * base_price_loss +
                         self.direction_loss_weight * base_direction_loss)

        orthogonal_penalty, cosine_sim = self.orthogonal_loss(price_features, direction_features)

        # Total loss
        total_loss = (self.price_loss_weight * price_loss +
                        self.direction_loss_weight * direction_loss +
                        base_losses + orthogonal_penalty
                    )

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
        self.log('val_cosine_sim', torch.mean(cosine_sim), on_step=False, on_epoch=True)
        if self.include_base_losses:
            self.log('val_base_losses', base_losses, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        """
        Computes and logs training metrics at the end of each training epoch.
        
        Resets metric objects after logging.
        """
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
        logger.info("--------- Training Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        logger.info(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        logger.info(f"Models Fitted - Ridge: {self.ridge_fitted}")
        logger.info("--------- Training Complete ---------\n")

        # Reset metrics
        self.mae_train.reset()
        self.rmse_train.reset()
        self.r2_train.reset()
        self.direction_acc_train.reset()
        self.direction_f1_train.reset()

    def on_validation_epoch_end(self):
        """
        Computes and logs validation metrics at the end of each validation epoch.
        
        Resets metric objects after logging.
        """
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
        logger.info("--------- Validation Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        logger.info(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        logger.info("--------- Validation Complete ---------\n")

        # Reset metrics
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
        self.direction_acc_val.reset()
        self.direction_f1_val.reset()

    def test_step(self, *args, **kwargs):
        """
        Run a test step (delegated to the validation step).
        
        Args:
            batch (Tuple[torch.Tensor]): A batch from the test set.
            batch_idx (int): Index of the current test batch.
        
        Returns:
            torch.Tensor: Total test loss.
        """
        batch = args[0]
        batch_idx = args[1]

        logger.debug("Test step: batch_idx=%d", batch_idx)
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """
        Computes and logs test metrics at the end of the test epoch.
        
        Resets metric objects after logging.
        """
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
        logger.info("--------- Test Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: {price_mae:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
        logger.info(f"Direction - Acc: {direction_acc:.4f}, F1: {direction_f1:.4f}")
        logger.info("--------- Test Complete ---------\n")

        # Reset metrics
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
        self.direction_acc_val.reset()
        self.direction_f1_val.reset()

    def get_base_predictions(self, x):
        """
        Retrieve predictions from all base models.
        
        Args:
            x (torch.Tensor): Input features.
        
        Returns:
            dict: Dictionary containing predictions and features from CNN, Ridge, and LSTM.
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
                except (RuntimeError,
                        ValueError,
                        TypeError,
                        IndexError,
                        torch.cuda.OutOfMemoryError) as e:
                    logger.warning(f"Ridge prediction failed: {e}")

            lstm_logits = None
            lstm_probs = None
            try:
                lstm_logits = self.lstm(x)
                lstm_probs = F.softmax(lstm_logits, dim=1)
            except (RuntimeError,
                    ValueError,
                    TypeError,
                    IndexError,
                    torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"LSTM prediction failed: {e}")

            return {
                'cnn_price': cnn_price,
                'cnn_direction_logits': cnn_direction_logits,
                'cnn_direction_probs': cnn_direction_probs,
                'ridge_price': ridge_price,
                'lstm_logits': lstm_logits,
                'lstm_probs': lstm_probs,
                'price_features': price_features,
                'direction_features': direction_features
            }

    def get_meta_predictions(self, x):
        """
        Retrieve predictions from the meta-learners.
        
        Args:
            x (torch.Tensor): Input features.
        
        Returns:
            dict: Dictionary containing meta predictions and corresponding features.
        
        Raises:
            ValueError: If meta-learning is disabled.
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
        """
        Save the model components (CNN, Ridge, LSTM, and meta-learners if used) to disk.
        
        File paths are defined in the configuration.
        """
        logger.info("Saving model components to disk.")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent

        # Define paths
        cnn_path = Path(repo_root / self.cfg.model.cnnPath).resolve()
        lstm_path = Path(repo_root / self.cfg.model.lstmPath).resolve()
        ridge_path = Path(repo_root / self.cfg.model.ridgePath).resolve()

        # Create directories
        for path in [cnn_path, lstm_path, ridge_path]:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Save models
        torch.save(self.cnn.state_dict(), cnn_path)
        torch.save(self.lstm.state_dict(), lstm_path)
        torch.save(self.ridge.state_dict(), ridge_path)

        # Save meta-learners if they exist
        if self.use_meta_learning:
            meta_price_path = Path(repo_root / self.cfg.model.meta_price_path).resolve()
            meta_direction_path = Path(repo_root / self.cfg.model.meta_direction_path).resolve()

            meta_price_path.parent.mkdir(parents=True, exist_ok=True)
            meta_direction_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(self.meta_price.state_dict(), meta_price_path)
            torch.save(self.meta_dir.state_dict(), meta_direction_path)

            logger.info(f"Meta models saved to {meta_price_path} and {meta_direction_path}")

        logger.info(f"CNN saved to {cnn_path}")
        logger.info(f"Ridge saved to {ridge_path}")
        logger.info(f"LSTM saved to {lstm_path}")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.
        
        Returns:
            dict: Optimizer and LR scheduler configuration for PyTorch Lightning.
        """
        logger.info("Configuring optimizers and learning rate scheduler.")

        base_params = list(self.cnn.parameters()) + list(self.lstm.parameters())
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

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }
