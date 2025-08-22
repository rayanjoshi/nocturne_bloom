"""
Hyperparameter optimization for stock price prediction using Ray Tune and Optuna.

This module defines functions and classes to perform hyperparameter tuning for a stock
prediction model using an ensemble of CNN, LSTM, and Ridge regression components. It
integrates Ray Tune with Optuna for search space exploration, supports multi-objective
optimization, and handles training with PyTorch Lightning.

The main components include:
- Definition of search spaces for hyperparameter tuning.
- Configuration updates for the model, optimizer, and data module.
- Training and evaluation of the model with custom metrics.
- Integration with logging and checkpointing for reproducibility.
"""
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path
import hydra
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CheckpointConfig, RunConfig
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback

# Try to import OptunaSearch and handle if not available
try:
    from ray.tune.search.optuna import OptunaSearch
    OPTUNA_AVAILABLE = True
    print("✅ OptunaSearch imported successfully")
except ImportError as e:
    print(f"❌ OptunaSearch import failed: {e}")
    print("🔧 Falling back to basic search algorithm")
    OPTUNA_AVAILABLE = False

from src.model_ensemble import EnsembleModule
from src.data_module import StockDataModule
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

class MultiObjectiveMetric:
    """
    Custom metric combining MAE and direction accuracy with configurable weights.

    This class computes a weighted combination of Mean Absolute Error (MAE) and
    directional accuracy for multi-objective optimization in stock price prediction.

    Attributes:
        mae_weight (float): Weight for the MAE component in the combined score.
        acc_weight (float): Weight for the directional accuracy component.
        mae_scale (float): Scaling factor for normalizing MAE.

    Args:
        mae_weight (float, optional): Weight for MAE. Defaults to 0.5.
        acc_weight (float, optional): Weight for directional accuracy. Defaults to 0.5.
        mae_scale (float, optional): Scale for MAE normalization. Defaults to 1.0.
    """
    def __init__(self, mae_weight=0.5, acc_weight=0.5, mae_scale=1.0):
        self.mae_weight = mae_weight
        self.acc_weight = acc_weight
        self.mae_scale = mae_scale

    def __call__(self, mae, direction_acc):
        """
        Compute the combined metric score.

        Args:
            mae (float): Mean Absolute Error value.
            direction_acc (float): Directional accuracy value.

        Returns:
            float: Combined score based on weighted MAE and directional accuracy.
        """
        # Normalize MAE to 0-1 range (lower is better, so we use 1-normalized_mae)
        mae_normalized = 1.0 - min(mae / self.mae_scale, 1.0)
        # Direction accuracy is already in 0-1 range (higher is better)
        combined_score = (self.mae_weight * mae_normalized +
                         self.acc_weight * direction_acc)
        return combined_score

def optuna_search_space(trial):
    """
    Define the hyperparameter search space for Optuna.

    This function creates an exhaustive search space for tuning hyperparameters
    of a stock prediction model, optimizing for low MAE and high directional accuracy.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for suggesting parameters.

    Returns:
        dict: Dictionary of hyperparameter names and their suggested values.
    """
    if not OPTUNA_AVAILABLE:
        return {}

    return {
        # CNN Architecture parameters - comprehensive ranges
        "cnn_ch1": trial.suggest_categorical(
            "cnn_ch1",
            [8, 16, 24, 32, 48, 64, 96, 128],
        ),
        "cnn_ch2": trial.suggest_categorical(
            "cnn_ch2",
            [16, 32, 48, 64, 96, 128, 192, 256],
        ),
        "cnn_ch3": trial.suggest_categorical(
            "cnn_ch3",
            [8, 16, 24, 32, 48, 64, 96, 128],
        ),

        # Kernel sizes - expanded range
        "kernel1": trial.suggest_categorical(
            "kernel1", [1, 2, 3, 4, 5, 7]
        ),
        "kernel2": trial.suggest_categorical(
            "kernel2", [1, 2, 3, 4, 5, 7]
        ),
        "kernel3": trial.suggest_categorical(
            "kernel3", [1, 2, 3, 4, 5]
        ),

        # Padding configurations
        "padding1": trial.suggest_categorical(
            "padding1", [0, 1, 2, 3]
        ),
        "padding2": trial.suggest_categorical(
            "padding2", [0, 1, 2, 3]
        ),
        "padding3": trial.suggest_categorical(
            "padding3", [0, 1, 2, 3]
        ),

        # Pooling configurations
        "pool1": trial.suggest_categorical(
            "pool1", [1, 2, 3, 4, 5]
        ),
        "pool2": trial.suggest_categorical(
            "pool2", [1, 2, 3, 4, 5]
        ),
        "pool3": trial.suggest_categorical(
            "pool3", [1, 2, 3, 4, 5]
        ),

        # CNN stride
        "cnn_stride": trial.suggest_categorical(
            "cnn_stride", [1, 2]
        ),

        # Dropout rates - fine-grained control
        "dropout1": trial.suggest_float(
            "dropout1", 0.05, 0.7, step=0.05
        ),
        "dropout2": trial.suggest_float(
            "dropout2", 0.1, 0.8, step=0.05
        ),

        # Ridge parameters - expanded range
        "ridge_alpha": trial.suggest_float(
            "ridge_alpha", 1e-4, 1e2, log=True
        ),
        "ridge_eps": trial.suggest_float(
            "ridge_eps", 1e-10, 1e-6, log=True
        ),

        # LSTM parameters
        "lstm_hidden_size": trial.suggest_int(
            "lstm_hidden_size", 64, 256, step=16
        ),
        "lstm_num_layers": trial.suggest_int(
            "lstm_num_layers", 1, 3
        ),
        "lstm_dropout": trial.suggest_float(
            "lstm_dropout", 0.1, 0.5, step=0.1
        ),

        # Ensemble weights - fine-grained control for balance
        "price_cnn_weight": trial.suggest_float(
            "price_cnn_weight", 0.1, 2.0, step=0.01
        ),
        "ridge_weight": trial.suggest_float(
            "ridge_weight", 0.1, 2.0, step=0.01
        ),
        "direction_cnn_weight": trial.suggest_float(
            "direction_cnn_weight", 0.1, 2.0, step=0.01
        ),
        "lstm_weight": trial.suggest_float(
            "lstm_weight", 0.1, 2.0, step=0.01
        ),

        # Loss weights - critical for MAE vs accuracy balance
        "price_loss_weight": trial.suggest_float(
            "price_loss_weight", 0.1, 0.9, step=0.01
        ),
        "direction_loss_weight": trial.suggest_float(
            "direction_loss_weight", 0.1, 0.9, step=0.01
        ),
        "orthogonal_lambda": trial.suggest_float(
            "orthogonal_lambda", 1e-6, 1e2, step=0.1
        ),

        # Huber loss parameter - important for price prediction robustness
        "huber_delta": trial.suggest_float(
            "huber_delta", 0.01, 5.0, step=0.01
        ),

        # Focal loss parameters - important for class imbalance
        "focal_gamma": trial.suggest_float(
            "focal_gamma", 0.5, 5.0, step=0.1
        ),
        "focal_alpha": trial.suggest_float(
            "focal_alpha", 0.0, 1.0, step=0.01
        ),
        "focal_beta": trial.suggest_float(
            "focal_beta", 0.9, 0.9999, step=0.001
        ),

        # Optimizer parameters - comprehensive tuning
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-8, 1e-1, log=True
        ),
        "optimizer_eps": trial.suggest_float(
            "optimizer_eps", 1e-12, 1e-4, log=True
        ),

        # Scheduler parameters - learning rate control
        "scheduler_mode": trial.suggest_categorical(
            "scheduler_mode", ["min", "max"]
        ),
        "scheduler_factor": trial.suggest_float(
            "scheduler_factor", 0.1, 0.9, step=0.01
        ),
        "scheduler_patience": trial.suggest_categorical(
            "scheduler_patience", [2, 3, 5, 7, 10, 15, 20]
        ),
        "scheduler_min_lr": trial.suggest_float(
            "scheduler_min_lr", 1e-10, 1e-4, log=True
        ),

        # Training parameters - comprehensive control
        "max_epochs": trial.suggest_categorical(
            "max_epochs",
            [30, 40, 50, 60, 70, 80, 100, 120],
        ),
        "gradient_clip_val": trial.suggest_float(
            "gradient_clip_val", 0.1, 5.0, step=0.1
        ),
        "early_stopping_patience": trial.suggest_categorical(
            "early_stopping_patience",
            [5, 8, 10, 12, 15, 20, 25, 30],
        ),
        "early_stopping_delta": trial.suggest_float(
            "early_stopping_delta", 1e-8, 1e-1, log=True
        ),

        # Data module parameters - affect model input
        "batch_size": trial.suggest_categorical(
            "batch_size", [16, 32, 48, 64, 96, 128]
        ),
        "window_size": trial.suggest_categorical(
            "window_size", [5, 7, 10, 15, 20, 25]
        ),

        # CNN architecture choices
        "num_classes": trial.suggest_categorical(
            "num_classes", [3]
        ),  # Keep as 3 for your setup
        "output_size": trial.suggest_categorical(
            "output_size", [1]
        ),  # Keep as 1 for regression

        # Meta-learning and optimizer params
        "use_meta_learning": trial.suggest_categorical(
            "use_meta_learning", [True, False]
        ),
        "include_base_losses": trial.suggest_categorical(
            "include_base_losses", [True, False]
        ),
        "meta_price_loss_weight": trial.suggest_float(
            "meta_price_loss_weight", 0.5, 2.0, step=0.05
        ),
        "meta_direction_loss_weight": trial.suggest_float(
            "meta_direction_loss_weight", 0.5, 2.0, step=0.05
        ),
        "base_lr": trial.suggest_float(
            "base_lr", 1e-7, 1e-2, log=True
        ),
        "meta_lr": trial.suggest_float(
            "meta_lr", 1e-7, 1e-2, log=True
        ),
    }

def get_ray_tune_search_space():
    """
    Define the hyperparameter search space for Ray Tune.

    This function provides a fallback search space for Ray Tune when Optuna is unavailable,
    matching the structure of the Optuna search space for consistency.

    Returns:
        dict: Dictionary of hyperparameter names and their search distributions.
    """
    return {
        # CNN Architecture parameters - comprehensive ranges
        "cnn_ch1": tune.choice([8, 16, 24, 32, 48, 64, 96, 128]),
        "cnn_ch2": tune.choice([16, 32, 48, 64, 96, 128, 192, 256]),
        "cnn_ch3": tune.choice([8, 16, 24, 32, 48, 64, 96, 128]),

        # Kernel sizes - expanded range
        "kernel1": tune.choice([1, 2, 3, 4, 5, 7]),
        "kernel2": tune.choice([1, 2, 3, 4, 5, 7]),
        "kernel3": tune.choice([1, 2, 3, 4, 5]),

        # Padding configurations
        "padding1": tune.choice([0, 1, 2, 3]),
        "padding2": tune.choice([0, 1, 2, 3]),
        "padding3": tune.choice([0, 1, 2, 3]),

        # Pooling configurations
        "pool1": tune.choice([1, 2, 3, 4, 5]),
        "pool2": tune.choice([1, 2, 3, 4, 5]),
        "pool3": tune.choice([1, 2, 3, 4, 5]),

        # CNN stride
        "cnn_stride": tune.choice([1, 2]),

        # Dropout rates - fine-grained control
        "dropout1": tune.uniform(0.05, 0.7),
        "dropout2": tune.uniform(0.1, 0.8),

        # Ridge parameters - expanded range
        "ridge_alpha": tune.loguniform(1e-4, 1e2),
        "ridge_eps": tune.loguniform(1e-10, 1e-6),

        # LSTM parameters
        "lstm_hidden_size": tune.choice([64, 128, 256]),
        "lstm_num_layers": tune.choice([1, 2, 3]),
        "lstm_dropout": tune.uniform(0.1, 0.5),

        # Ensemble weights - fine-grained control for balance
        "price_cnn_weight": tune.uniform(0.1, 2.0),
        "ridge_weight": tune.uniform(0.1, 2.0),
        "direction_cnn_weight": tune.uniform(0.1, 2.0),
        "lstm_weight": tune.uniform(0.1, 2.0),
        "orthogonal_lambda": tune.loguniform(1e-6,1e2),
        # Loss weights - critical for MAE vs accuracy balance
        "price_loss_weight": tune.uniform(0.1, 0.9),
        "direction_loss_weight": tune.uniform(0.1, 0.9),

        # Huber loss parameter - important for price prediction robustness
        "huber_delta": tune.uniform(0.01, 5.0),

        # Focal loss parameters - important for class imbalance
        "focal_gamma": tune.uniform(0.5, 5.0),
        "focal_alpha": tune.uniform( 0.0, 1.0),
        "focal_beta": tune.uniform(0.9, 0.9999),

        # Optimizer parameters - comprehensive tuning
        "weight_decay": tune.loguniform(1e-8, 1e-1),
        "optimizer_eps": tune.loguniform(1e-12, 1e-4),

        # Scheduler parameters - learning rate control
        "scheduler_mode": tune.choice(["min", "max"]),
        "scheduler_factor": tune.uniform(0.1, 0.9),
        "scheduler_patience": tune.choice([2, 3, 5, 7, 10, 15, 20]),
        "scheduler_min_lr": tune.loguniform(1e-10, 1e-4),

        # Training parameters - comprehensive control
        "max_epochs": tune.choice([30, 40, 50, 60, 70, 80, 100, 120]),
        "gradient_clip_val": tune.uniform(0.1, 5.0),
        "early_stopping_patience": tune.choice([5, 8, 10, 12, 15, 20, 25, 30]),
        "early_stopping_delta": tune.loguniform(1e-8, 1e-1),

        # Data module parameters - affect model input
        "batch_size": tune.choice([16, 32, 48, 64, 96, 128]),
        "window_size": tune.choice([5, 7, 10, 15, 20, 25]),

        # CNN architecture choices
        "num_classes": tune.choice([3]),  # Keep as 3 for your setup
        "output_size": tune.choice([1]),  # Keep as 1 for regression
        # Meta-learning and optimizer params
        "meta_price_loss_weight": tune.uniform(0.5, 2.0),
        "meta_direction_loss_weight": tune.uniform(0.5, 2.0),
        "base_lr": tune.loguniform(1e-7, 1e-2),
        "meta_lr": tune.loguniform(1e-7, 1e-2),
    }

def update_config_from_trial_params(
    base_cfg: DictConfig, trial_params: Dict[str, Any]
) -> DictConfig:
    """
    Update the base configuration with trial parameters.

        This function modifies the base configuration with hyperparameters from a trial,
        updating parameters for CNN, Ridge, LSTM, ensemble weights, losses, optimizer,
        trainer, and data module.

        Args:
            base_cfg (DictConfig): Base configuration to update.
            trial_params (Dict[str, Any]): Trial parameters to apply.

        Returns:
            DictConfig: Updated configuration object.
        """
    # Create a copy of base config
    cfg = OmegaConf.create(OmegaConf.to_yaml(base_cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Update CNN parameters
    cfg_dict["cnn"]["cnnChannels"] = [
        trial_params["cnn_ch1"],
        trial_params["cnn_ch2"],
        trial_params["cnn_ch3"],
    ]
    cfg_dict["cnn"]["kernelSize"] = [
        trial_params["kernel1"],
        trial_params["kernel2"],
        trial_params["kernel3"],
    ]
    cfg_dict["cnn"]["padding"] = [
        trial_params.get("padding1", 1),
        trial_params.get("padding2", 1),
        trial_params.get("padding3", 1),
    ]
    cfg_dict["cnn"]["poolSize"] = [
        trial_params["pool1"],
        trial_params["pool2"],
        trial_params.get("pool3", 3),
    ]
    cfg_dict["cnn"]["stride"] = trial_params.get("cnn_stride", 1)
    cfg_dict["cnn"]["dropout"] = [
        trial_params["dropout1"],
        trial_params["dropout2"],
    ]
    cfg_dict["cnn"]["num_classes"] = trial_params.get("num_classes", 3)
    cfg_dict["cnn"]["outputSize"] = trial_params.get("output_size", 1)

    # Update Ridge parameters
    cfg_dict["Ridge"]["alpha"] = trial_params["ridge_alpha"]
    cfg_dict["Ridge"]["eps"] = trial_params.get("ridge_eps", 1e-8)

    # Update LSTM parameters
    cfg_dict["LSTM"]["hidden_size"] = trial_params.get("lstm_hidden_size", 128)
    cfg_dict["LSTM"]["num_layers"] = trial_params.get("lstm_num_layers", 2)
    cfg_dict["LSTM"]["dropout"] = trial_params.get("lstm_dropout", 0.3)

    # Normalize and update ensemble weights
    price_cnn_w = trial_params["price_cnn_weight"]
    ridge_w = trial_params["ridge_weight"]
    price_total = price_cnn_w + ridge_w
    cfg_dict["model"]["price_cnn_weight"] = price_cnn_w / price_total
    cfg_dict["model"]["ridge_weight"] = ridge_w / price_total

    dir_cnn_w = trial_params["direction_cnn_weight"]
    lstm_w = trial_params["lstm_weight"]
    direction_total = dir_cnn_w + lstm_w
    cfg_dict["model"]["direction_cnn_weight"] = dir_cnn_w / direction_total
    cfg_dict["model"]["lstm_weight"] = lstm_w / direction_total

    # Normalize and update loss weights
    price_loss_w = trial_params["price_loss_weight"]
    direction_loss_w = trial_params["direction_loss_weight"]
    loss_total = price_loss_w + direction_loss_w
    cfg_dict["model"]["price_loss_weight"] = price_loss_w / loss_total
    cfg_dict["model"]["direction_loss_weight"] = direction_loss_w / loss_total

    cfg_dict["model"]["orthogonal_lambda"] = trial_params.get(
        "orthogonal_lambda", 0.1
    )
    cfg_dict["model"]["focal_gamma"] = trial_params.get("focal_gamma", 2.0)
    cfg_dict["model"]["focal_alpha"] = trial_params.get("focal_alpha", 0.25)
    cfg_dict["model"]["focal_beta"] = trial_params.get("focal_beta", 0.9999)

    # Update model Huber parameter
    cfg_dict["model"]["huber_delta"] = trial_params["huber_delta"]

    # Meta-learning and optimizer params
    default_meta_price = cfg_dict["model"].get("meta_price_loss_weight", 1.0)
    cfg_dict["model"]["meta_price_loss_weight"] = trial_params.get(
        "meta_price_loss_weight", default_meta_price
    )

    meta_dir_default = cfg_dict["model"].get("meta_direction_loss_weight", 1.0)
    cfg_dict["model"]["meta_direction_loss_weight"] = trial_params.get(
        "meta_direction_loss_weight", meta_dir_default
    )

    default_base_lr = cfg_dict["optimiser"].get("base_lr", 1e-3)
    cfg_dict["optimiser"]["base_lr"] = trial_params.get("base_lr", default_base_lr)

    meta_lr_default = cfg_dict["optimiser"].get("meta_lr", 1e-3)
    cfg_dict["optimiser"]["meta_lr"] = trial_params.get(
        "meta_lr", meta_lr_default
    )

    # Update optimiser parameters
    cfg_dict["optimiser"]["weightDecay"] = trial_params["weight_decay"]
    cfg_dict["optimiser"]["eps"] = trial_params["optimizer_eps"]
    cfg_dict["optimiser"]["schedulerMode"] = trial_params.get(
        "scheduler_mode", "min"
    )
    cfg_dict["optimiser"]["schedulerFactor"] = trial_params["scheduler_factor"]
    cfg_dict["optimiser"]["schedulerPatience"] = trial_params[
        "scheduler_patience"
    ]
    cfg_dict["optimiser"]["schedulerMinLR"] = trial_params["scheduler_min_lr"]

    # Update trainer parameters
    cfg_dict["trainer"]["max_epochs"] = trial_params["max_epochs"]
    cfg_dict["trainer"]["gradient_clip_val"] = trial_params[
        "gradient_clip_val"
    ]
    cfg_dict["trainer"]["early_stopping_patience"] = trial_params[
        "early_stopping_patience"
    ]
    cfg_dict["trainer"]["early_stopping_delta"] = trial_params[
        "early_stopping_delta"
    ]

    # Update data module parameters
    cfg_dict["data_module"]["batch_size"] = trial_params.get("batch_size", 64)
    cfg_dict["data_module"]["window_size"] = trial_params.get(
        "window_size", 10
    )

    return OmegaConf.create(cfg_dict)

def train_model(config, base_cfg=None, optimization_target="multi_objective"):
    """
    Train the stock prediction model with given hyperparameters.

    This function configures and trains the ensemble model using PyTorch Lightning,
    reporting validation metrics to Ray Tune for hyperparameter optimization.

    Args:
        config (dict): Hyperparameters for the trial.
        base_cfg (DictConfig, optional): Base configuration to update. Defaults to None.
        optimization_target (str, optional): Metric to optimize. Defaults to "multi_objective".

    Raises:
        RuntimeError: If model training fails due to computational issues.
        ValueError: If configuration parameters are invalid.
        OSError: If file operations fail.
        KeyboardInterrupt: If training is interrupted by the user.
    """
    logger = log_function_start("train_model", optimization_target=optimization_target)

    # Update config with trial parameters
    cfg = update_config_from_trial_params(base_cfg, config)

    # Setup temporary directory for this trial
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=temp_dir,
            filename="model-{epoch:02d}-{val_mae:.4f}",
            save_top_k=1,
            mode="min",
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.trainer.early_stopping_patience,
            mode="min",
            min_delta=cfg.trainer.early_stopping_delta,
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=False,  # Disable logger for tune trials
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            deterministic=True,
            gradient_clip_val=cfg.trainer.gradient_clip_val,
        )

        try:
            # Initialize model and data
            torch.manual_seed(cfg.trainer.seed)
            logger.debug("Seed set to %d", cfg.trainer.seed)
            model = EnsembleModule(cfg)
            data_module = StockDataModule(cfg)

            # Train the model
            trainer.fit(model, datamodule=data_module)

            # Get final validation metrics
            val_results = trainer.validate(model, datamodule=data_module, verbose=False)

            if val_results and len(val_results) > 0:
                val_mae = val_results[0].get("val_price_mae", float('inf'))
                val_direction_acc = val_results[0].get("val_direction_acc", 0.0)
                val_r2 = val_results[0].get("val_price_r2", -float('inf'))
                val_loss = val_results[0].get("val_loss", float('inf'))

                # Calculate multi-objective score
                metric_calculator = MultiObjectiveMetric(
                    mae_weight=0.4,
                    acc_weight=0.6,
                    mae_scale=1.0,
                )
                multi_objective_score = metric_calculator(val_mae, val_direction_acc)

                # Report all metrics
                metrics = {
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                    "val_direction_acc": val_direction_acc,
                    "multi_objective_score": multi_objective_score,
                    "epoch": trainer.current_epoch
                }
            else:
                # Report poor metrics if validation failed
                metrics = {
                    "val_loss": float('inf'),
                    "val_mae": float('inf'),
                    "val_r2": -float('inf'),
                    "val_direction_acc": 0.0,
                    "multi_objective_score": 0.0,
                    "epoch": 0
                }

            session.report(metrics)
            log_function_end("train_model", success=True)

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Trial failed with error: {e}")
            # Report failure metrics
            session.report({
                "val_loss": float('inf'),
                "val_mae": float('inf'),
                "val_r2": -float('inf'),
                "val_direction_acc": 0.0,
                "multi_objective_score": 0.0,
                "epoch": 0
            })
            log_function_end("train_model", success=False)
        except KeyboardInterrupt:
            # Allow user interruption to propagate cleanly
            logger.warning("Trial interrupted by user (KeyboardInterrupt)")
            raise

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: Optional[DictConfig] = None):
    """
    Perform hyperparameter optimization for stock prediction model.

    This function sets up and runs a hyperparameter search using Ray Tune with Optuna
    or random search, optimizing for a multi-objective metric combining MAE and
    directional accuracy. Results are logged and the best configuration is saved.

    Args:
        cfg (DictConfig, optional): Hydra configuration object. Defaults to None.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    logger.info("=== NVDA Stock Predictor Hyperparameter Optimization with Optuna ===")

    # Configuration for optimization
    metric = "multi_objective_score"
    mode = "max"
    num_samples = 100
    max_concurrent = 5

    logger.info("Optimizing for combined MAE and Direction Accuracy")
    logger.info(f"Number of trials: {num_samples}")
    logger.info(f"Max concurrent trials: {max_concurrent}")

    # Configure scheduler
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=metric,
        mode=mode,
        max_t=100,
        grace_period=5,
        reduction_factor=2,
        brackets=1
    )
    logger.debug("ASHA Scheduler configured")

    # Configure search algorithm - try OptunaSearch first, fallback to basic search
    if OPTUNA_AVAILABLE:
        try:
            # Get the Ray Tune search space
            search_space = get_ray_tune_search_space()

            # Try OptunaSearch without the space parameter
            # OptunaSearch can work with the search space passed to Tuner
            search_alg = OptunaSearch(
                metric=metric,
                mode=mode,
            )

            logger.info("✅ Using OptunaSearch (Optuna TPE sampler)")

        except (ValueError, RuntimeError, TypeError) as e:
            # Handle known initialization errors and fall back to random search
            logger.warning(f"❌ OptunaSearch failed to initialize: {e}")
            logger.info("🔧 Falling back to basic RandomSearch")
            search_alg = None
            search_space = get_ray_tune_search_space()
    else:
        logger.info("🔧 Using basic RandomSearch (Optuna not available)")
        search_alg = None
        search_space = get_ray_tune_search_space()

    logger.debug("Search algorithm configured")

    # Debug information about the search algorithm
    if search_alg is not None:
        logger.info(f"Search algorithm type: {type(search_alg).__name__}")
        logger.info(f"Search algorithm module: {type(search_alg).__module__}")
        # Log the Optuna study name only if it is exposed via a public attribute
        study_name = getattr(search_alg, "study_name", None)
        if study_name is not None:
            logger.info(f"Optuna study: {study_name}")
        else:
            logger.debug(
                "OptunaSearch does not expose 'study_name' publicly; "
                "skipping study name log."
            )
    else:
        logger.info("Using default RandomSearch (no custom search algorithm)")

    # Configure reporter
    reporter = CLIReporter(
        metric_columns=[
            "val_loss", 
            "val_mae", 
            "val_r2", 
            "val_direction_acc",
            "multi_objective_score",
            "epoch"
        ],
        parameter_columns=[
            "base_lr",
            "meta_lr",
            "ridge_alpha",
            "price_loss_weight_raw",
            "direction_loss_weight_raw",
        ],
        max_report_frequency=30
    )
    logger.debug("CLI Reporter configured")

    # Configure run
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    outputs_dir = (repo_root / "outputs/ray_results_optuna").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory for results: %s", outputs_dir)

    run_config = RunConfig(
        name=f"nvda_optuna_{metric}",
        storage_path=str(outputs_dir),
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute=metric,
            checkpoint_score_order=mode,
        ),
        callbacks=[
            WandbLoggerCallback(
                project=f"{cfg.trainer.project_name}_optuna",
                group=f"optuna_{metric}",
                api_key_file=None,
                log_config=True,
                save_checkpoints=False,
            )
        ],
        progress_reporter=reporter,
        stop={"epoch": 100},
    )

    logger.info("\nStarting Optuna hyperparameter search:")
    logger.info(f"   - Target metric: {metric}")
    logger.info(f"   - Optimization mode: {mode}")
    logger.info(f"   - Number of trials: {num_samples}")
    logger.info(f"   - Max concurrent: {max_concurrent}")
    logger.info(f"   - Results will be saved to: {outputs_dir}")

    # Create the trainable function with base_cfg
    trainable = tune.with_parameters(
        train_model,
        base_cfg=cfg,
        optimization_target=metric
    )

    # Run hyperparameter search - always use param_space with search_space
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,  # Always pass the search space here
        tune_config=tune.TuneConfig(
            search_alg=search_alg,  # This can be OptunaSearch or None
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=run_config,
    )

    results = tuner.fit()

    # Get best results
    best_result = results.get_best_result(metric=metric, mode=mode)

    logger.info("\n" + "="*60)
    logger.info("🏆 OPTUNA OPTIMIZATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best {metric}: {best_result.metrics[metric]:.6f}")
    logger.info(f"Best validation loss: {best_result.metrics['val_loss']:.6f}")
    logger.info(f"Best validation MAE: {best_result.metrics['val_mae']:.6f}")
    logger.info(f"Best validation R²: {best_result.metrics['val_r2']:.6f}")
    logger.info(f"Best direction accuracy: {best_result.metrics['val_direction_acc']:.6f}")
    logger.info(f"Training completed at epoch: {best_result.metrics['epoch']}")

    logger.info("\n🔧 Best hyperparameters:")
    logger.info("-" * 40)
    for key, value in best_result.config.items():
        logger.info(f"{key}: {value}")

    # Save best configuration
    best_config_path = f"outputs/best_config_optuna_{metric}.yaml"

    # Create updated config with best parameters
    best_cfg = update_config_from_trial_params(cfg, best_result.config)

    # Save the configuration
    Path(best_config_path).parent.mkdir(parents=True, exist_ok=True)
    best_config_path = Path(repo_root / best_config_path).resolve()
    OmegaConf.save(best_cfg, best_config_path)

    logger.info(f"\nBest configuration saved to: {best_config_path}")
    # Cleanup Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
