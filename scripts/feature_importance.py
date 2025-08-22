"""Feature importance analysis for stock price prediction models.

This module performs feature importance analysis on preprocessed financial data
using Ridge Regression and a Convolutional Neural Network (CNN) from an ensemble model.
It computes and visualizes feature importance using SHAP values for Ridge Regression
and saliency maps for the CNN.

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
    - src.model_Ensemble: Custom module containing the EnsembleModule class.

The module loads preprocessed data, fits a Ridge Regression model, computes SHAP values,
and generates saliency maps for the CNN component of an ensemble model. Results are
printed and visualized as bar plots.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import shap
import matplotlib.pyplot as plt
import torch
from captum.attr import Saliency
from omegaconf import OmegaConf
from src.model_ensemble import EnsembleModule

# Load preprocessed features
script_dir = Path(__file__).parent
repo_root = script_dir.parent
# Automatically select the correct file with Target column
preprocessed_path = repo_root / 'data/preprocessing/nvda_processed_data.csv'
df = pd.read_csv(preprocessed_path, header=0, index_col=0, parse_dates=True)

# Drop target column and any non-feature columns
x = df.drop(columns=['Price_Target'])
y = df['Price_Target']


# Use features as-is for next-day prediction
X_lagged = x
y_lagged = y


# Ridge Regression Coefficients
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_lagged, y_lagged)
print('Ridge Regression Coefficients:')
for i, col in enumerate(X_lagged.columns):
    print(f"{col}: {ridge.coef_[i]:.4f}")

# SHAP values for Ridge Regression
explainer = shap.Explainer(ridge, X_lagged)
shap_values = explainer(X_lagged)
print('\nSHAP Feature Importance (mean absolute value):')

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
for i in range(len(X_lagged.columns)):
    print(f"{i+1}. {X_lagged.columns[shap_indices[i]]}: {shap_importance[shap_indices[i]]:.4f}")

# SHAP summary plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_to_plot, X_lagged, plot_type="bar", show=False)
plt.tight_layout()
plt.show()



# --- Saliency Maps for CNN from model_Ensemble.py ---
# Load config
config_path = repo_root / 'configs/model_Ensemble.yaml'
cfg = OmegaConf.load(str(config_path))

# Instantiate ensemble and load CNN weights
ensemble = EnsembleModule(cfg)
cnn_weights_path = repo_root / cfg.model.cnnPath
ensemble.cnn.load_state_dict(torch.load(cnn_weights_path, map_location='cpu'))
ensemble.cnn.eval()

# Prepare a batch of features for attribution (first N rows)
WINDOW_SIZE = 10
N = 32  # Number of samples for attribution
# Build rolling windows: shape [N, num_features, window_size]
X_arr = X_lagged.values
num_features = X_arr.shape[1]
windows = []
for i in range(N):
    if i + WINDOW_SIZE > len(X_arr):
        break
    # window: [WINDOW_SIZE, num_features]
    window = X_arr[i:i+WINDOW_SIZE, :]  # shape [WINDOW_SIZE, num_features]
    windows.append(window)
inputs = torch.tensor(np.stack(windows), dtype=torch.float32)
inputs.requires_grad_()

# Debug: print input shape and assert
print(f"CNN input shape: {inputs.shape} (expected [batch,window,features] = {inputs.shape[:3]})")

# Saliency attribution on CNN
saliency = Saliency(ensemble.cnn)
attributions = saliency.attribute(inputs)
# attributions: [N, window_size, num_features]
saliency_scores = attributions.abs().mean(dim=(0,1)).detach().cpu().numpy().squeeze()
saliency_indices = np.argsort(saliency_scores)[::-1]

print('\nSaliency Map Feature Importance (CNN, mean abs gradient):')
for i in range(len(X_lagged.columns)):
    col = X_lagged.columns[saliency_indices[i]]
    score = saliency_scores[saliency_indices[i]]
    print(f"{i+1}. {col}: {score:.4f}")

# Plot saliency scores
plt.figure(figsize=(10,6))
plt.title('Saliency Map Feature Importances (CNN)')
plt.bar(range(len(X_lagged.columns)), saliency_scores[saliency_indices], align='center')
plt.xticks(range(len(X_lagged.columns)), X_lagged.columns[saliency_indices], rotation=90)
plt.tight_layout()
plt.show()
