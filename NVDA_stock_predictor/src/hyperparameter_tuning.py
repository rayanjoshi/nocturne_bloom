import hydra
import torch
import os
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
from ray.tune import CheckpointConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from model_Ensemble import EnsembleModule
from data_module import StockDataModule


def define_search_space_r2():
    """Define search space optimized for R2 score"""
    return {
        # CNN Architecture - More complex for R2 optimization
        "cnn.cnnChannels": tune.choice([
            [16, 32, 16],
            [32, 64, 32], 
            [64, 128, 64],
            [32, 64, 128],
            [16, 64, 32]
        ]),
        "cnn.kernelSize": tune.choice([
            [3, 3, 2], 
            [3, 3, 3],
            [5, 3, 2],
            [3, 5, 3],
            [5, 5, 3]
        ]),
        "cnn.dropout": tune.choice([
            [0.2, 0.3],
            [0.3, 0.4],
            [0.3, 0.5],
            [0.4, 0.5],
            [0.5, 0.6]
        ]),
        
        "cnn.poolSize": tune.choice([
            [2, 2, 2],
            [3, 3, 2],
            [2, 3, 2],
            [3, 2, 2],
            [2, 2, 3]
        ]),
        
        "cnn.poolPadding": tune.choice([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
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
            monitor="val_mae",
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
        
        tune.report(metrics=metrics)

@hydra.main(version_base=None, config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    """Main function with Ray Tune integration"""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    print("=== NVDA Stock Predictor Hyperparameter Optimization ===")
    search_space = define_search_space_r2()
    metric = "val_mae"
    mode = "min"
    num_samples = 100
    max_concurrent = 5
    print("\n📈 Optimizing for MAE")

    
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
    
    # Configure search algorithm
    search_alg = OptunaSearch(
        metric=metric,
        mode=mode,
    )
    
    # Configure reporter

    reporter = CLIReporter(
        metric_columns=["val_r2", "val_loss", "val_mae", "epoch"],
        max_report_frequency=30
    )
    
    # Configure run
    outputs_dir = os.path.abspath("../outputs/ray_results")
    os.makedirs(outputs_dir, exist_ok=True)
    
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
    
    print(f"\n🚀 Starting hyperparameter search:")
    print(f"   - Search space size: {len(search_space)} parameters")
    print(f"   - Target metric: {metric}")
    print(f"   - Optimization target: {"mae" if metric == "val_mae" else "r2"}")
    print(f"   - Number of trials: {num_samples}")
    print(f"   - Max concurrent: {max_concurrent}")
    print(f"   - Results will be saved to: {outputs_dir}")
    
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
    
    print("\n" + "="*60)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Best {metric}: {best_result.metrics[metric]:.6f}")
    print(f"Best validation loss: {best_result.metrics['val_loss']:.6f}")
    print(f"Best validation MAE: {best_result.metrics['val_mae']:.6f}")
    print(f"Training completed at epoch: {best_result.metrics['epoch']}")
    
    print("\n🔧 Best hyperparameters:")
    print("-" * 40)
    for key, value in best_result.config.items():
        print(f"{key}: {value}")
    
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
    
    # Normalize classifier weights
    total_weight = (best_cfg.classifiers.GBWEIGHT + best_cfg.classifiers.RFWEIGHT + 
                    best_cfg.classifiers.LOGISTICWEIGHT + best_cfg.classifiers.SVMWEIGHT)
    best_cfg.classifiers.GBWEIGHT /= total_weight
    best_cfg.classifiers.RFWEIGHT /= total_weight
    best_cfg.classifiers.LOGISTICWEIGHT /= total_weight
    best_cfg.classifiers.SVMWEIGHT /= total_weight
    
    # Save the configuration
    os.makedirs("../outputs", exist_ok=True)
    OmegaConf.save(best_cfg, best_config_path)
    
    print(f"\n💾 Best configuration saved to: {best_config_path}")
    print("\n📊 To train the final model with these parameters:")
    print(f"   python train_final_model.py --config-path=../outputs --config-name=best_config_{metric}")
    
    # Optionally train final model
    print("\n" + "="*60)
    train_final = input("Would you like to train the final model now? (y/n): ").strip().lower()
    
    if train_final == 'y':
        print("\n🔥 Training final model with best hyperparameters...")
        train_final_model(best_cfg)
    
    ray.shutdown()


def train_final_model(cfg: DictConfig):
    """Train the final model with best hyperparameters"""
    
    # Setup WandB logger for final training
    wandb_logger = WandbLogger(
        project=f"{cfg.trainer.project_name}_final",
        name=f"final_model_{cfg.trainer.run_name}",
        save_dir="../logs",
        tags=["final_model", "best_hyperparams"]
    )
    
    # Configure callbacks for final training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_r2",
        dirpath="../models/final",
        filename="final-model-{epoch:02d}-{val_r2:.2f}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.early_stopping_patience,
        mode="min",
        min_delta=cfg.trainer.early_stopping_delta,
    )
    
    # Create trainer for final model
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        deterministic=True,
        gradient_clip_val=getattr(cfg.trainer, 'gradient_clip_val', 1.0),
    )
    
    # Initialize model and data
    torch.manual_seed(cfg.trainer.seed)
    model = EnsembleModule(cfg)
    data_module = StockDataModule(cfg)
    
    # Train the final model
    print("🚂 Starting final model training...")
    trainer.fit(model, datamodule=data_module)
    
    # Test the final model
    test_results = trainer.test(model, datamodule=data_module)
    
    print("\n🎉 Final model training complete!")
    print(f"Final test R2: {test_results[0]['test_r2']:.6f}")
    print(f"Final test loss: {test_results[0]['test_loss']:.6f}")
    print(f"Final test MAE: {test_results[0]['test_mae']:.6f}")

    
    return trainer, model
    
if __name__ == "__main__":
    main()