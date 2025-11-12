"""Feature importance analysis for stock price prediction models.

This module performs feature importance analysis on preprocessed financial data
using Ridge Regression from an ensemble model.
It computes and visualizes feature importance using SHAP values for Ridge Regression.

Dependencies:
    - pathlib: For file path handling.
    - pandas: For data manipulation and CSV handling.
    - numpy: For numerical operations.
    - sklearn.linear_model.Ridge: For Ridge Regression modeling.
    - shap: For SHAP value computation and visualization.
    - matplotlib.pyplot: For plotting SHAP and saliency results.
    - torch: For PyTorch-based CNN operations.
    - captum.attr.Saliency: For computing saliency maps.
    - omegaconf: For configuration file handling.
    - src.model_ensemble: Custom module containing the EnsembleModule class.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import shap
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="model_ensemble")
def main(cfg: Optional[DictConfig] = None):
    """Run feature importance analysis on preprocessed NVDA stock data.

    This function loads the CSV data, fits a Ridge Regression model, computes SHAP
    values for feature importance, and generates a summary bar plot. It then
    instantiates an EnsembleModule, loads the CNN weights, prepares windowed inputs,
    computes vanilla saliency gradients for the price prediction output, and prints
    and plots the mean absolute gradient scores per feature.

    Args:
        cfg (Optional[DictConfig]): Hydra configuration object. If None, defaults
            to hardcoded target columns ("Price_Target", "Direction_Target").

    Raises:
        KeyError: If no CNN path is found in cfg.model.
        RuntimeError: If insufficient data for windowed inputs.
        ValueError: If output unpacking fails due to unexpected model outputs.
    """

    # Load preprocessed features
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    preprocessed_path = repo_root / "data/preprocessing/nvda_processed_data.csv"
    df = pd.read_csv(preprocessed_path, header=0, index_col=0, parse_dates=True)

    # Use configured target column if available
    try:
        price_col = cfg.data_module.price_target_col
        direction_col = cfg.data_module.direction_target_col
    except (AttributeError, KeyError):
        price_col = "Price_Target"
        direction_col = "Direction_Target"

    # Drop target column and any non-feature columns
    x = df.drop(columns=[price_col, direction_col])
    y = df[price_col]

    # Use features as-is for next-day prediction
    x_lagged = x
    y_lagged = y

    # Ridge Regression Coefficients
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x_lagged, y_lagged)
    print("Ridge Regression Coefficients:")
    for i, col in enumerate(x_lagged.columns):
        print(f"{col}: {ridge.coef_[i]:.4f}")

    # SHAP values for Ridge Regression
    explainer = shap.Explainer(ridge, x_lagged)
    shap_values = explainer(x_lagged)
    print("\nSHAP Feature Importance (mean absolute value):")

    # Support both Explanation objects and legacy/list outputs from SHAP
    if hasattr(shap_values, "values"):
        # modern Explanation object
        shap_array = shap_values.values
        shap_to_plot = shap_values
    else:
        # shap_values might be a list of Explanation objects or raw arrays
        try:
            arrs = []
            for sv in shap_values:
                if hasattr(sv, "values"):
                    arrs.append(sv.values)
                else:
                    arrs.append(np.array(sv))
            shap_array = np.array(arrs)
        except (TypeError, ValueError):
            shap_array = np.array(shap_values)
        if shap_array.ndim == 3 and shap_array.shape[0] == 1:
            shap_array = shap_array[0]
        if isinstance(shap_values, (list, tuple)) and len(shap_values) > 0:
            shap_to_plot = shap_values[0]
        else:
            shap_to_plot = shap_values

    shap_importance = np.abs(shap_array).mean(axis=0)
    shap_indices = np.argsort(shap_importance)[::-1]
    for i in range(len(x_lagged.columns)):
        print(f"{i+1}. {x_lagged.columns[shap_indices[i]]}: {shap_importance[shap_indices[i]]:.4f}")

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_to_plot, x_lagged, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
