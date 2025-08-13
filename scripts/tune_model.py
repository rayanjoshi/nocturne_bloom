import hydra
import torch
import tempfile
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.air import CheckpointConfig, RunConfig, session
from ray.air.integrations.wandb import WandbLoggerCallback
from pathlib import Path

from src.model_Ensemble import EnsembleModule
from src.data_module import StockDataModule
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

def define_search_space_r2():
    """Define search space optimized for R2 score"""
    return {
        # CNN Architecture - More complex for R2 optimization
        "cnn.cnnChannels": tune.choice([
            (16, 32, 16),
            (32, 64, 32), 
            (64, 128, 64),
            (32, 64, 128),
            (16, 64, 32)
        ]),
        "cnn.kernelSize": tune.choice([
            (3, 3, 2), 
            (3, 3, 3),
            (5, 3, 2),
            (3, 5, 3),
            (5, 5, 3)
        ]),
        "cnn.dropout": tune.choice([
            (0.2, 0.3),
            (0.3, 0.4),
            (0.3, 0.5),
            (0.4, 0.5),
            (0.5, 0.6)
        ]),
        
        "cnn.poolSize": tune.choice([
            (2, 2, 2),
            (3, 3, 2),
            (2, 3, 2),
            (3, 2, 2),
            (2, 2, 3)
        ]),
        
        "cnn.poolPadding": tune.choice([
            (0, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1)
        ]),
        
        # Model weights (balanced for R2)
        "model.cnnWeight": tune.uniform(0.1, 0.5),
        "model.ridgeWeight": tune.uniform(0.5, 0.9),
        "model.huber_delta": tune.uniform(0.1, 1.0),
        
        # Ridge regularization
        "Ridge.alpha": tune.loguniform(0.001, 100.0),
        
        # Optimizer settings
        "optimiser.name": tune.choice(["adam"]),
        "optimiser.lr": tune.loguniform(1e-6, 1e-2),
        "optimiser.weightDecay": tune.loguniform(1e-7, 1e-2),
        "optimiser.schedulerFactor": tune.uniform(0.2, 0.7),
        "optimiser.schedulerPatience": tune.choice([3, 5, 7, 10, 15]),
        
        # Training settings
        "trainer.max_epochs": tune.choice([50, 70, 100]),
        "trainer.early_stopping_patience": tune.choice([10, 15, 20]),
        "trainer.early_stopping_delta": tune.loguniform(1e-6, 1e-2),
        "trainer.gradient_clip_val": tune.uniform(0.5, 2.0),
    }


def train_model(config, base_cfg, optimization_target="val_mae"):
    """Training function for Ray Tune"""
    logger = log_function_start("train_model", optimization_target=optimization_target)
    # Create a copy of base config and update with tune parameters
    cfg = OmegaConf.create(OmegaConf.to_yaml(base_cfg))
    
    # Update config with tuned parameters using a different approach
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Update nested parameters
    for key, value in config.items():
        keys = key.split('.')
        current_dict = cfg_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    
    cfg = OmegaConf.create(cfg_dict)
    
    # Setup temporary directory for this trial
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=optimization_target,
            dirpath=temp_dir,
            filename="model-{epoch:02d}-{val_mae:.2f}",
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
            gradient_clip_val=getattr(cfg.trainer, 'gradient_clip_val', 1.0),
        )
        
        # Initialize model and data
        torch.manual_seed(cfg.trainer.seed)
        logger.debug("Seed set to %d", cfg.trainer.seed)
        model = EnsembleModule(cfg)
        data_module = StockDataModule(cfg)
        
        # Train the model
        trainer.fit(model, datamodule=data_module)
        
        # Get final validation metrics
        val_results = trainer.validate(model, datamodule=data_module, verbose=False)
        
        # Report metrics based on optimization target
        
        metrics = {
            "val_r2": val_results[0]["val_r2"],
            "val_loss": val_results[0]["val_loss"],
            "val_mae": val_results[0]["val_mae"],
            "val_rmse": val_results[0]["val_rmse"],
            "epoch": trainer.current_epoch
        }
        
    session.report(metrics)
    log_function_end("train_model", success=True)

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    """Main function with Ray Tune integration"""
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    logger.info("=== NVDA Stock Predictor Hyperparameter Optimization ===")
    search_space = define_search_space_r2()
    logger.info("Search space defined with %d parameters", len(search_space))
    metric = "val_mae"
    mode = "min"
    num_samples = 1
    max_concurrent = 5
    logger.info("Optimizing for MAE")
    
    
    # Configure scheduler
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=metric,
        mode=mode,
        max_t=cfg.trainer.max_epochs,
        grace_period=5,
        reduction_factor=2,
        brackets=3
    )
    logger.debug("Scheduler configured with max epochs %d", cfg.trainer.max_epochs)
    
    # Configure search algorithm
    search_alg = OptunaSearch(
        metric=metric,
        mode=mode,
    )
    logger.debug("Search algorithm configured with metric %s", metric)
    
    # Configure reporter
    
    reporter = CLIReporter(
        metric_columns=["val_r2", "val_loss", "val_mae", "epoch"],
        max_report_frequency=30
    )
    logger.debug("Reporter configured with columns: %s", reporter._metric_columns)
    
    # Configure run
    script_dir = Path(__file__).parent  # /path/to/repo/src
    repo_root = script_dir.parent  # /path/to/repo/
    outputs_dir = (repo_root / "outputs/ray_results").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory for results: %s", outputs_dir)
    
    run_config = RunConfig(
        name=f"nvda_tune_{metric}",
        storage_path=outputs_dir,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute=metric,
            checkpoint_score_order=mode,
        ),
        callbacks=[
            WandbLoggerCallback(
                project=f"{cfg.trainer.project_name}_tune",
            group=f"tune_{metric}",
                api_key_file=None,  # Will use WANDB_API_KEY env var
                log_config=True,
                save_checkpoints=False,
            )
        ],
        progress_reporter=reporter,
        stop={"epoch": cfg.trainer.max_epochs},
    )
    
    # Determine optimization target
    
    logger.info(f"\n🚀 Starting hyperparameter search:")
    logger.info(f"   - Search space size: {len(search_space)} parameters")
    logger.info(f"   - Target metric: {metric}")
    logger.info(f"   - Optimization target: {"mae" if metric == "val_mae" else "r2"}")
    logger.info(f"   - Number of trials: {num_samples}")
    logger.info(f"   - Max concurrent: {max_concurrent}")
    logger.info(f"   - Results will be saved to: {outputs_dir}")
    
    # Run hyperparameter search
    tuner = tune.Tuner(
        tune.with_parameters(train_model, base_cfg=cfg, optimization_target="val_mae" if metric == "val_mae" else "val_r2"),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
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
    logger.info("🏆 OPTIMIZATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best {metric}: {best_result.metrics[metric]:.6f}")
    logger.info(f"Best validation loss: {best_result.metrics['val_loss']:.6f}")
    logger.info(f"Best validation MAE: {best_result.metrics['val_mae']:.6f}")
    logger.info(f"Training completed at epoch: {best_result.metrics['epoch']}")
    logger.info("\n🔧 Best hyperparameters:")
    logger.info("-" * 40)
    for key, value in best_result.config.items():
        logger.info(f"{key}: {value}")
    
    # Save best configuration
    best_config_path = f"../outputs/best_config_{metric}.yaml"
    
    # Create updated config with best parameters
    best_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(best_cfg, resolve=True)
    
    # Update nested parameters
    for key, value in best_result.config.items():
        keys = key.split('.')
        current_dict = cfg_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    
    best_cfg = OmegaConf.create(cfg_dict)
    
    # Save the configuration
    Path(best_config_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(best_cfg, best_config_path)
    
    logger.info(f"\n💾 Best configuration saved to: {best_config_path}")
    logger.info("\n📊 To train the final model with these parameters:")
    logger.info(f"   python train_final_model.py --config-path=../outputs --config-name=best_config_{metric}")

if __name__ == "__main__":
    main()