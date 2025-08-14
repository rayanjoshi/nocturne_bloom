import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from captum.attr import Saliency
from omegaconf import OmegaConf
from src.model_Ensemble import EnsembleModule

# Load preprocessed features
script_dir = Path(__file__).parent
repo_root = script_dir.parent
# Automatically select the correct file with Target column
preprocessed_path = repo_root / 'data/preprocessing/nvda_processed_data.csv'
df = pd.read_csv(preprocessed_path, header=0, index_col=0, parse_dates=True)

# Drop target column and any non-feature columns
X = df.drop(columns=['Target'])
y = df['Target']


# Use features as-is for next-day prediction
X_lagged = X
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
shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_indices = np.argsort(shap_importance)[::-1]
for i in range(len(X_lagged.columns)):
    print(f"{i+1}. {X_lagged.columns[shap_indices[i]]}: {shap_importance[shap_indices[i]]:.4f}")

# SHAP summary plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_lagged, plot_type="bar", show=False)
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
window_size = 10
N = 32  # Number of samples for attribution
# Build rolling windows: shape [N, num_features, window_size]
X_arr = X_lagged.values
num_features = X_arr.shape[1]
windows = []
for i in range(N):
    if i + window_size > len(X_arr):
        break
    # window: [window_size, num_features]
    window = X_arr[i:i+window_size, :]  # shape [window_size, num_features]
    windows.append(window)
inputs = torch.tensor(np.stack(windows), dtype=torch.float32)  # shape [N, window_size, num_features]
inputs.requires_grad_()

# Debug: print input shape and assert
print(f"CNN input shape: {inputs.shape} (should be [batch, window_size, num_features] = [{inputs.shape[0]}, {inputs.shape[1]}, {inputs.shape[2]}])")

# Saliency attribution on CNN
saliency = Saliency(ensemble.cnn)
attributions = saliency.attribute(inputs)
# attributions: [N, window_size, num_features]
saliency_scores = attributions.abs().mean(dim=(0,1)).detach().cpu().numpy().squeeze()  # mean over batch and window
saliency_indices = np.argsort(saliency_scores)[::-1]

print('\nSaliency Map Feature Importance (CNN, mean abs gradient):')
for i in range(len(X_lagged.columns)):
    print(f"{i+1}. {X_lagged.columns[saliency_indices[i]]}: {saliency_scores[saliency_indices[i]]:.4f}")

# Plot saliency scores
plt.figure(figsize=(10,6))
plt.title('Saliency Map Feature Importances (CNN)')
plt.bar(range(len(X_lagged.columns)), saliency_scores[saliency_indices], align='center')
plt.xticks(range(len(X_lagged.columns)), X_lagged.columns[saliency_indices], rotation=90)
plt.tight_layout()
plt.show()
