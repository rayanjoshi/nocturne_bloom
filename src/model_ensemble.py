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
    - torchmetrics: Metrics for evaluation (MAE, WWMAPE, R2).
    - lightning: PyTorch Lightning for scalable training.
    - omegaconf: For configuration management.
    - scripts.logging_config: Custom logging utilities.
"""
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, Metric
from torchmetrics.regression import R2Score, WeightedMeanAbsolutePercentageError
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
        output_seq_len = cfg.cnn.output_seq_len

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
        )

        self.price_features = nn.Sequential(
            nn.Conv1d(cnn_channels[2], cnn_channels[1], kernel_size=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[1]),
        )

        self.direction_features = nn.Sequential(
            nn.Conv1d(cnn_channels[2], cnn_channels[1], kernel_size=1),
            nn.ReLU(),
            nn.Dropout(cfg.cnn.dropout[1] * 0.5),

            nn.Conv1d(cnn_channels[1], cnn_channels[1] * 2, kernel_size=1),
            nn.BatchNorm1d(cnn_channels[1] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[1]),

            nn.Conv1d(cnn_channels[1] * 2, cnn_channels[1], kernel_size=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(inplace=True),
        )

        self.pricehead = nn.Sequential(
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0]),
            nn.Conv1d(cnn_channels[2], output_seq_len, kernel_size=1)
        )

        num_classes = cfg.cnn.num_classes
        self.directionhead = nn.Sequential(
            nn.Conv1d(cnn_channels[1], cnn_channels[2] * 2, kernel_size=1),
            nn.BatchNorm1d(cnn_channels[2] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0] * 0.5),

            nn.Conv1d(cnn_channels[2] * 2, cnn_channels[2] * 2, kernel_size=1),
            nn.BatchNorm1d(cnn_channels[2] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0]),

            nn.Conv1d(cnn_channels[2] * 2, cnn_channels[2], kernel_size=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cnn.dropout[0]),

            nn.Conv1d(cnn_channels[2], num_classes, kernel_size=1)
        )

        self.pooled_to_dir = nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=1)
        self.dir_to_price = nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for convolutional, linear, and batch normalization layers.

        This method applies specific initialization techniques to different layer types to
        promote better convergence during training. Convolutional and non-head linear layers
        use Kaiming initialization, classification head linear layers use Xavier initialization,
        and batch normalization layers are initialized to preserve scale and shift.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Check if this is an output head
                if m in self.pricehead.modules() or m in self.directionhead.modules():
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Perform the forward pass of the MultiHeadCNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_channels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - price_pred (torch.Tensor): Predicted price, shape (batch_size, output_seq_len).
            - direction_pred (torch.Tensor): Class logits, shape (batch_size, num_classes).
            - price_features (torch.Tensor): Features used for price.
            - direction_features (torch.Tensor): Features used for classification.
        """
        logger.debug(f"MultiHeadCNN forward pass with input shape: {x.shape}")
        x = x.transpose(1, 2)  # (batch_size, input_channels, window_size)
        x = self.cnn(x)

        price_features = self.price_features(x)
        direction_features_raw = self.direction_features(x)

        # Residual connections
        pooled = price_features
        pooled_proj = self.pooled_to_dir(pooled)
        direction_features = direction_features_raw + pooled_proj

        if direction_features.size(1) != price_features.size(1):
            direction_features = self.dir_to_price(direction_features)

        price_pred = self.pricehead(price_features)
        # price_pred should be (batch_size, output_seq_len, remaining_seq_len)
        # We want (batch_size, output_seq_len), so take the mean or last timestep
        price_pred = price_pred.mean(dim=2)  # Average across time dimension

        # Direction head - final classification, not per-timestep
        direction_pred = self.directionhead(direction_features)
        direction_pred = direction_pred.mean(dim=2)

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
            f"Initializing RidgeRegressor with alpha={cfg.ridge.alpha:.4f}, "
            f"fit_intercept={cfg.ridge.fit_intercept}"
        )
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.ridge.alpha
        self.fit_intercept = cfg.ridge.fit_intercept
        self.eps = cfg.ridge.eps
        self.output_seq_len = cfg.cnn.output_seq_len

        # Ridge parameters (will be set during fit). Use placeholders so buffers
        # exist in the module and can be replaced with properly-shaped tensors
        # when `.fit()` computes the coefficients.
        self.register_buffer('weight', torch.empty(0))
        self.register_buffer('bias', torch.tensor(0.0))
        self.is_fitted = False
        self.register_buffer('condition_number', torch.tensor(0.0, requires_grad=False))
        self.fallback_count = 0  # Track fallback attempts

    def fit(self, x, y, rank=None):
        """
        Fit the Ridge regression model using input features and targets.

        Uses SVD for a numerically stable solution. Falls back to solving the
        regularized normal equations if SVD fails. Learns weights and an
        optional intercept.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, ...).
            y (torch.Tensor): Target values, shape (batch_size,) or (batch_size, output_seq_len).
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

        # Handle y dimensions - flatten if 2D but keep track of output dimension
        output_dim = 1
        if y.dim() == 2:
            output_dim = y.shape[1]
            y = y.view(-1, output_dim)  # Keep as 2D: [batch_size, output_dim]
        elif y.dim() == 1:
            y = y.view(-1, 1)  # Make 2D: [batch_size, 1]

        x = x.float()

        # Add intercept column if needed
        if self.fit_intercept:
            ones = torch.ones(x.shape[0], 1, device=device)
            x_with_intercept = torch.cat([x, ones], dim=1)
        else:
            x_with_intercept = x

        try:
            # Compute SVD in float32 explicitly for numerical stability
            orig_dtype = x_with_intercept.dtype
            x_svd = x_with_intercept.to(torch.float32)

            u_matrix, singular_values, v_transpose = torch.linalg.svd(  # pylint: disable=not-callable
                x_svd, full_matrices=False
            )
            if rank is not None:
                u_matrix = u_matrix[:, :rank]
                singular_values = singular_values[:rank]
                v_transpose = v_transpose[:rank, :]

            # Log condition number (computed in float32)
            max_sv = torch.max(singular_values)
            min_sv = torch.min(singular_values)
            condition_number = (max_sv / (min_sv + self.eps)).to(x_with_intercept.dtype)
            logger.info(f"Matrix condition number: {condition_number:.4f}")

            # Ridge solution using SVD (all float32 math)
            s_reg = singular_values / (singular_values**2 + self.alpha + self.eps)
            y_float32 = y.to(torch.float32)
            u_ty = torch.mm(u_matrix.t(), y_float32)
            s_reg_expanded = s_reg.unsqueeze(1)
            regularized_u_ty = s_reg_expanded * u_ty
            theta = torch.mm(v_transpose.t(), regularized_u_ty)
            theta = theta.to(device).to(orig_dtype)

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

            # x_with_intercept.t() @ y
            xty = torch.mm(x_with_intercept.t(), y)  # [n_features, output_dim]

            try:
                theta = torch.linalg.solve(xtx_reg, xty)  # pylint: disable=not-callable
            except RuntimeError:
                logger.warning("Using pseudo-inverse due to singular matrix")
                theta = torch.linalg.pinv(xtx_reg) @ xty  # pylint: disable=not-callable

        # Validate coefficients
        if torch.any(torch.abs(theta) > 1e5):
            logger.warning(f"Large coefficients detected: max |theta| = {torch.abs(theta).max()}")

        # Split weight and bias - theta has shape [n_features, output_dim]
        if self.fit_intercept:
            self.register_buffer('weight', theta[:-1, :].clone().detach())
            self.register_buffer('bias', theta[-1, :].clone().detach())
        else:
            self.register_buffer('weight', theta.clone().detach())
            self.register_buffer('bias', torch.zeros(output_dim, device=device))

        # Log training MAE
        with torch.no_grad():
            y_pred = torch.mm(x_with_intercept, theta)  # [batch_size, output_dim]
            mae = torch.mean(torch.abs(y_pred - y))
            script_dir = Path(__file__).parent
            repo_root = script_dir.parent
            scaled_rel_path = Path(self.cfg.data_module.price_y_scaled_save_path)
            scaler_path = (repo_root / scaled_rel_path).resolve()
            target_scaler = joblib.load(scaler_path)
            mae_np = np.array([[mae]])
            mae = target_scaler.inverse_transform(mae_np)[0, 0]
            logger.info(f"Training MAE after fitting: ${mae:.2f}")

        self.is_fitted = True


    def forward(self, x):
        """
        Perform a forward pass using the fitted Ridge regression model.
        
        Predicts outputs using the learned weights (and optional intercept).
        Input is flattened to 2D if needed.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *) or (batch_size, features).
        
        Returns:
            torch.Tensor: Predicted values of shape (batch_size, output_dim).
        
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

        predictions = torch.mm(x, self.weight)

        if self.fit_intercept:
            predictions = predictions + self.bias.unsqueeze(0)
        if predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)

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
            - cfg.lstm.hidden_size (int): Hidden dimension size of the LSTM.
            - cfg.lstm.num_layers (int): Number of LSTM layers.
            - cfg.lstm.dropout (float): Dropout rate.
            - cfg.cnn.inputChannels (int): Dimensionality of input features.
            - cfg.cnn.num_classes (int): Number of output classes.
    """
    def __init__(self, cfg: DictConfig):
        logger.info(
            f"Initializing Improved LSTMClassifier with hidden_size={cfg.lstm.hidden_size}, "
            f"num_layers={cfg.lstm.num_layers}"
        )
        super().__init__()
        self.input_size = cfg.cnn.inputChannels
        self.hidden_size = cfg.lstm.hidden_size
        self.num_layers = cfg.lstm.num_layers
        self.num_classes = cfg.cnn.num_classes
        self.dropout = cfg.lstm.dropout

        # Input projection to align CNN features
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        self.res_proj = nn.Linear(self.hidden_size, self.hidden_size * 2)

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
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    # Leave LSTM weights with default initialization
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Perform a forward pass through the LSTM classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
        
        Returns:
            torch.Tensor: Output class logits of shape (batch_size, num_classes).
                        Note: Returns single prediction per batch, not per timestep.
        """
        logger.debug(f"LSTMClassifier forward pass with input shape: {x.shape}")
        batch_size = x.size(0)
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_size * 2)
        out = self.norm(out)
        residual = self.res_proj(x)  # (batch_size, seq_len, hidden_size * 2)
        out = out + residual  # (batch_size, seq_len, hidden_size * 2)

        # Take the last timestep output for final classification
        # This gives us one prediction per sequence, not per timestep
        out = out[:, -1, :]  # (batch_size, hidden_size * 2)
        logits = self.fc(out)  # (batch_size, num_classes)

        logger.debug(f"LSTMClassifier output logits shape: {logits.shape}")

        return logits


class MetaPriceRegressor(nn.Module):
    """
    A linear regressor that combines outputs from multiple base models.
    """
    def __init__(self, in_dim: int, output_seq_len: int):
        super().__init__()
        self.output_seq_len = output_seq_len
        self.linear = nn.Linear(in_dim, output_seq_len)

    def forward(self, z):
        """
        Forward pass for the price regressor.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, in_dim).
        
        Returns:
            torch.Tensor: Predicted price values of shape (batch_size, output_seq_len).
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


class DirectionalAccuracy(Metric):
    """
    Calculate directional accuracy of price predictions.

    This metric computes the ratio of correct direction predictions (up, down, or flat)
    to the total number of valid cases (non-zero price changes) across batches.
    It compares the direction of predicted price changes against actual price changes.

    Attributes:
        higher_is_better (bool): Indicates that higher metric values are better. Set to True.
        correct (torch.Tensor): Running sum of correct direction predictions.
        total (torch.Tensor): Total number of valid predictions.
        prev_pred (list or torch.Tensor): Previous batch predictions for calculating changes.
        prev_target (list or torch.Tensor): Previous batch targets for calculating changes.
    """
    higher_is_better = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize attributes so static analyzers and type checkers can see them
        self.correct: torch.Tensor = torch.tensor(0)
        self.total: torch.Tensor = torch.tensor(0)
        self.prev_pred = []
        self.prev_target = []
        # Register states with torchmetrics
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("prev_pred", default=[], persistent=False)
        self.add_state("prev_target", default=[], persistent=False)

    def update(self, *args, **kwargs):
        """
        Update metric with current batch predictions and targets.

        Args:
            *args: Positional arguments where:
                - args[0] (torch.Tensor): Predicted prices for the current batch.
                - args[1] (torch.Tensor): Target prices for the current batch.
            **kwargs: Additional keyword arguments (ignored).

        Notes:
            - Skips the first batch as it requires previous values to compute changes.
            - Handles variable batch sizes by truncating to the minimum length.
        """
        price_pred = args[0]
        price_target = args[1]
        _unused = kwargs
        # Flatten if needed
        pred = price_pred.flatten()
        target = price_target.flatten()

        # Skip first batch (need previous values)
        if isinstance(self.prev_pred, list) and len(self.prev_pred) == 0:
            self.prev_pred = pred.clone()
            self.prev_target = target.clone()
            return

        # Ensure consistent batch sizes by taking the minimum length
        min_len = min(len(pred), len(self.prev_pred))
        pred = pred[:min_len]
        target = target[:min_len]
        prev_pred = self.prev_pred[:min_len]
        prev_target = self.prev_target[:min_len]

        # Compute price changes
        pred_change = pred - prev_pred
        target_change = target - prev_target

        # Check if directions match: both up, both down, or both flat
        same_direction = (pred_change > 0) == (target_change > 0)

        # Count correct predictions
        correct = same_direction.sum()
        total = len(same_direction)

        self.correct += correct
        self.total += total

        # Store for next batch
        self.prev_pred = pred.clone()
        self.prev_target = target.clone()

    def compute(self):
        """
        Compute the directional accuracy ratio.

        Returns:
            torch.Tensor: The ratio of correct direction predictions to total valid cases.
                Returns 0.0 if no valid cases exist (total is 0).
        """
        if self.total > 0:
            return self.correct.float() / self.total.float()
        return torch.tensor(0.0)

    def reset(self) -> None:
        """
        Reset internal metric state safely.

        We implement a custom reset to avoid the base Metric.reset trying to call
        `.clear()` on tensor objects (which raises AttributeError). This method
        re-initializes tensor counters and sequence states to empty lists.
        """
        # Reset scalar/tensor states
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

        # Reset sequence/previous-value states to empty lists so subsequent
        # updates behave as the first batch logic expects.
        self.prev_pred = []
        self.prev_target = []

class EnsembleModule(L.LightningModule):
    """
    A PyTorch Lightning module that ensembles CNN, Ridge, and LSTM models.
    
    Combines a multi-head 1D CNN, a Ridge regressor, and a bidirectional LSTM
    for price regression and direction classification. Supports optional
    meta-learning for price prediction and custom losses (Focal, Orthogonal).
    
    Attributes:
        cfg (DictConfig): Model and optimizer configuration.
        use_meta_learning (bool): Whether meta-learning is enabled.
        include_base_losses (bool): Whether to include base model losses.
        price_criterion (nn.Module): Loss for price regression.
        direction_criterion (nn.Module): Loss for direction classification.
        orthogonal_loss (nn.Module): Penalty enforcing feature orthogonality.
        ridge_fitted (bool): Flag set when Ridge model has been fitted.
        class_counts (torch.Tensor): Label frequencies for class balancing.
        cnn (MultiHeadCNN): CNN for feature extraction and prediction.
        ridge (RidgeRegressor): Ridge regressor for linear corrections.
        lstm (LSTMClassifier): LSTM classifier for direction.
        meta_price (MetaPriceRegressor): Meta-learner for price (optional).
        num_classes (int): Number of direction classes.
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
            meta_price_dim = 9
            self.meta_price = MetaPriceRegressor(
                in_dim=meta_price_dim,
                output_seq_len=self.cfg.cnn.output_seq_len,
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
        self.wmape_train = WeightedMeanAbsolutePercentageError()
        self.r2_train = R2Score()
        self.mae_val = MeanAbsoluteError()
        self.wmape_val = WeightedMeanAbsolutePercentageError()
        self.r2_val = R2Score()
        # Direction Metrics
        self.dir_acc_train = DirectionalAccuracy()
        self.dir_acc_val = DirectionalAccuracy()

    def build_meta_features(self, x, cnn_price, cnn_dir_logits):
        """
        Construct meta-features for price meta-learner from base outputs.
        
        Generates features including CNN and Ridge predictions,
        direction entropy, model disagreement, and directional signals.
        
        Args:
            x (torch.Tensor): Input, shape (batch, seq_len, input_channels).
            cnn_price (torch.Tensor): CNN price preds,
            shape (batch, output_seq_len).
            cnn_dir_logits (torch.Tensor): CNN direction logits,
            shape (batch, num_classes).
        
        Returns:
            torch.Tensor: Meta-features for price regression,
            shape (batch, 9).
        """
        batch_size = cnn_price.shape[0]
        device = cnn_price.device
        cnn_dir_probs = F.softmax(cnn_dir_logits, dim=1)  # [batch_size, num_classes]

        ridge_price = torch.zeros(batch_size, device=device)
        if self.ridge_fitted and self.ridge.is_fitted:
            try:
                ridge_price = self.ridge(x)
                if ridge_price.dim() > 1:
                    ridge_price = ridge_price.squeeze(-1)  # Ensure 1D
            except (RuntimeError,
                    ValueError,
                    TypeError,
                    IndexError,
                    torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"Ridge prediction failed in meta-features: {e}")

        lstm_logits = torch.zeros(batch_size, self.num_classes, device=device)
        try:
            lstm_logits = self.lstm(x)  # Should be (batch_size, num_classes)
            lstm_probs = F.softmax(lstm_logits, dim=1)
        except (RuntimeError,
                ValueError,
                TypeError,
                IndexError,
                torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"LSTM prediction failed in meta-features: {e}")
            lstm_probs = torch.zeros(batch_size, self.num_classes, device=device)

        # Ensure all tensors are 1D for price calculations
        if cnn_price.dim() > 1:
            cnn_price = cnn_price.squeeze(-1)

        dir_entropy = -torch.sum(cnn_dir_probs * torch.log(cnn_dir_probs + 1e-10), dim=1)

        price_diff = torch.abs(cnn_price - ridge_price)
        price_mean = (cnn_price + ridge_price) / 2
        price_disagreement = price_diff / (torch.abs(price_mean) + 1e-6)

        eps = 1e-10
        ratio = (cnn_dir_probs + eps) / (lstm_probs + eps)
        dir_kl_div = torch.sum(cnn_dir_probs * torch.log(ratio), dim=1)

        direction_confidence = torch.max(cnn_dir_probs, dim=1)[0]

        direction_signal = direction_confidence - 0.5
        weighted_direction_signal = direction_signal * direction_confidence

        cosine_similarity = F.cosine_similarity(cnn_dir_probs, lstm_probs, dim=1)  # pylint: disable=not-callable

        price_direction_confidence = direction_confidence * torch.abs(cnn_price)

        z_price = torch.stack([
            cnn_price.squeeze(-1) if cnn_price.dim() > 1 else cnn_price,
            ridge_price.squeeze(-1) if ridge_price.dim() > 1 else ridge_price,
            dir_entropy,
            price_disagreement,
            dir_kl_div,
            direction_confidence,
            weighted_direction_signal,
            cosine_similarity,
            price_direction_confidence,
        ], dim=1)
        return z_price

    def forward(self, *args, **kwargs):
        """
        Perform forward pass of the ensemble module.
        
        Combines base model predictions (CNN, Ridge, LSTM) or uses a
        meta-learner for price prediction when enabled. Returns price
        predictions, direction logits, and feature embeddings.
        
        Args:
            x (torch.Tensor): Input features with shape
            (batch_size, seq_len, input_channels).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_price: Final price prediction with shape
                (batch_size, output_seq_len).
            - final_direction: Direction logits with shape
                (batch_size, num_classes).
            - price_features: Price feature embeddings with shape
                (batch_size, feature_dim, seq_len).
            - direction_features: Direction feature embeddings with shape
                (batch_size, feature_dim, seq_len).
        """
        x = args[0]
        logger.debug(f"EnsembleModule forward pass with input shape: {x.shape}")

        cnn_price, cnn_direction_logits, price_features, direction_features = self.cnn(x)

        if self.use_meta_learning:
            z_price = self.build_meta_features(x, cnn_price, cnn_direction_logits)

            meta_price = self.meta_price(z_price)

            final_price = torch.clamp(meta_price, min=-1e6, max=1e6)
            final_direction = cnn_direction_logits # for supervised learning only
        else:
            ridge_price = torch.zeros_like(cnn_price)
            if self.ridge_fitted and self.ridge.is_fitted:
                try:
                    ridge_price = self.ridge(x)
                    if ridge_price.device != cnn_price.device:
                        ridge_price = ridge_price.to(cnn_price.device)
                except (RuntimeError,
                        ValueError,
                        TypeError,
                        IndexError,
                        torch.cuda.OutOfMemoryError) as e:
                    logger.warning(f"Ridge prediction failed: {e}")

            lstm_direction = torch.zeros_like(cnn_direction_logits)
            try:
                lstm_direction = self.lstm(x)
                if lstm_direction.device != cnn_direction_logits.device:
                    lstm_direction = lstm_direction.to(cnn_direction_logits.device)
            except (RuntimeError,
                    ValueError,
                    TypeError,
                    IndexError,
                    torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"LSTM prediction failed: {e}")

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

        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)

        base_losses = 0.0
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
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
        self.wmape_train.update(price_pred, price_y)
        self.r2_train.update(price_pred, price_y)
        self.dir_acc_train.update(price_pred, price_y)

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

        price_loss = self.price_criterion(price_pred, price_y)
        direction_loss = self.direction_criterion(direction_pred, direction_y)

        base_losses = 0.0
        if self.include_base_losses:
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
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
        self.wmape_val.update(price_pred, price_y)
        self.r2_val.update(price_pred, price_y)
        self.dir_acc_val.update(price_pred, price_y)

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
        price_wmape = self.wmape_train.compute()
        price_r2 = self.r2_train.compute()
        dir_acc = self.dir_acc_train.compute()

        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        scaler_path = (repo_root / Path(self.cfg.data_module.price_y_scaled_save_path)).resolve()
        target_scaler = joblib.load(scaler_path)
        price_mae_np = np.array([[price_mae]])
        price_mae = target_scaler.inverse_transform(price_mae_np)[0, 0]
        price_wmape = price_wmape * 100

        self.log('train_price_mae', price_mae, prog_bar=True)
        self.log('train_price_wmape', price_wmape, prog_bar=True)
        self.log('train_price_r2', price_r2, prog_bar=True)
        self.log('train_dir_acc', dir_acc, prog_bar=True)

        meta_status = "enabled" if self.use_meta_learning else "disabled"
        logger.info("--------- Training Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: ${price_mae:.2f}, WMAPE: {price_wmape:.2f}%, R²: {price_r2:.4f}")
        logger.info(f"Direction - Accuracy: {dir_acc:.4f}")
        logger.info(f"Models Fitted - Ridge: {self.ridge_fitted}")
        logger.info("--------- Training Complete ---------\n")

        # Reset metrics
        self.mae_train.reset()
        self.wmape_train.reset()
        self.r2_train.reset()
        self.dir_acc_train.reset()

    def on_validation_epoch_end(self):
        """
        Computes and logs validation metrics at the end of each validation epoch.
        
        Resets metric objects after logging.
        """
        logger.info("Validation epoch ended. Computing validation metrics.")

        price_mae = self.mae_val.compute()
        price_wmape = self.wmape_val.compute()
        price_r2 = self.r2_val.compute()
        dir_acc = self.dir_acc_val.compute()

        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        scaler_path = (repo_root / Path(self.cfg.data_module.price_y_scaled_save_path)).resolve()
        target_scaler = joblib.load(scaler_path)
        price_mae_np = np.array([[price_mae]])
        price_mae = target_scaler.inverse_transform(price_mae_np)[0, 0]
        price_wmape = price_wmape * 100

        self.log('val_price_mae', price_mae, prog_bar=True)
        self.log('val_price_wmape', price_wmape, prog_bar=True)
        self.log('val_price_r2', price_r2, prog_bar=True)
        self.log('val_dir_acc', dir_acc, prog_bar=True)

        meta_status = "enabled" if self.use_meta_learning else "disabled"
        logger.info("--------- Validation Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: ${price_mae:.2f}, WMAPE: {price_wmape:.2f}%, R²: {price_r2:.4f}")
        logger.info(f"Direction - Accuracy: {dir_acc:.4f}")
        logger.info("--------- Validation Complete ---------\n")

        # Reset metrics
        self.mae_val.reset()
        self.wmape_val.reset()
        self.r2_val.reset()
        self.dir_acc_val.reset()

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

        logger.debug(f"Test step: batch_idx={batch_idx}")
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """
        Computes and logs test metrics at the end of the test epoch.
        
        Resets metric objects after logging.
        """
        logger.info("Test epoch ended. Computing test metrics.")

        price_mae = self.mae_val.compute()
        price_wmape = self.wmape_val.compute()
        price_r2 = self.r2_val.compute()

        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        scaler_path = (repo_root / Path(self.cfg.data_module.price_y_scaled_save_path)).resolve()
        target_scaler = joblib.load(scaler_path)
        price_mae_np = np.array([[price_mae]])
        price_mae = target_scaler.inverse_transform(price_mae_np)[0, 0]
        price_wmape = price_wmape * 100

        self.log('test_price_mae', price_mae, prog_bar=True)
        self.log('test_price_wmape', price_wmape, prog_bar=True)
        self.log('test_price_r2', price_r2, prog_bar=True)

        meta_status = "enabled" if self.use_meta_learning else "disabled"
        logger.info("--------- Test Results ---------")
        logger.info(f"Meta-learning: {meta_status}")
        logger.info(f"Price - MAE: ${price_mae:.2f}, WMAPE: {price_wmape:.2f}%, R²: {price_r2:.4f}")
        logger.info("--------- Test Complete ---------\n")

        # Reset metrics
        self.mae_val.reset()
        self.wmape_val.reset()
        self.r2_val.reset()
        self.dir_acc_val.reset()

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
        Retrieve predictions from the price meta-learner.
        
        Args:
            x (torch.Tensor): Input features, shape (batch_size, seq_len, input_channels).
        
        Returns:
            dict: Dictionary containing meta-learner price predictions and features, with keys:
                - 'meta_price': Price predictions, shape (batch_size, output_seq_len).
                - 'meta_features_price': Meta-features for price, shape (batch_size, 9).
        """
        if not self.use_meta_learning:
            raise ValueError("Meta-learning is disabled. Use get_base_predictions() instead.")

        with torch.no_grad():
            cnn_price, cnn_direction_logits, _, _ = self.cnn(x)
            z_price= self.build_meta_features(x, cnn_price, cnn_direction_logits)

            meta_price = self.meta_price(z_price)

            return {
                'meta_price': meta_price,
                'meta_features_price': z_price,
            }

    def save_components(self):
        """
        Save model components (CNN, Ridge, LSTM, and meta-learner if used) to disk.
        
        Saves model state dictionaries to paths specified in the configuration.
        Ensures parent directories exist before saving.
        """
        logger.info("Saving model components to disk.")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent

        # Define paths
        cnn_path = Path(repo_root / self.cfg.model.cnn_path).resolve()
        lstm_path = Path(repo_root / self.cfg.model.lstm_path).resolve()
        ridge_path = Path(repo_root / self.cfg.model.ridge_path).resolve()

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

            meta_price_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(self.meta_price.state_dict(), meta_price_path)

            logger.info(f"Meta models saved to {meta_price_path}")

        logger.info(f"CNN saved to {cnn_path}")
        logger.info(f"Ridge saved to {ridge_path}")
        logger.info(f"LSTM saved to {lstm_path}")

    def configure_optimizers(self):
        """
        Configure optimizer and LR scheduler.
        
        Uses Adam with separate learning rates for base models and an optional
        meta-learner. Scheduler is ReduceLROnPlateau monitoring 'val_loss'.
        
        Returns:
            dict: {'optimizer', 'lr_scheduler', 'monitor'}.
        """
        logger.info("Configuring optimizers and learning rate scheduler.")

        base_params = list(self.cnn.parameters()) + list(self.lstm.parameters())
        meta_params = []

        if self.use_meta_learning:
            meta_params = list(self.meta_price.parameters())

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
