"""Module for visualizing feature correlations in preprocessed NVIDIA stock data.

This module provides functionality to create a correlation heatmap and identify
strong correlations between features in a dataset. It uses a preprocessed CSV file
containing NVIDIA stock data to generate a heatmap visualization and print the top
correlations with a target variable and between feature pairs.

Dependencies:
    matplotlib.pyplot: For creating the heatmap visualization
    pandas: For data manipulation and correlation calculations
    seaborn: For enhanced visualization of the correlation matrix

The module assumes the input CSV file has a datetime index and contains relevant
financial features for correlation analysis.
"""
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# Load preprocessed features
script_dir = Path(__file__).parent
repo_root = script_dir.parent
# Automatically select the correct file with Target column
preprocessed_path = repo_root / 'data/preprocessing/nvda_processed_data.csv'
df = pd.read_csv(preprocessed_path, header=0, index_col=0, parse_dates=True)
corr = df.corr()

# Create a much larger figure to accommodate all features
plt.figure(dpi=100, figsize=(20, 16))

# Use smaller font size and remove annotations for better readability
sns.heatmap(corr,
    annot=False, # Remove numbers to reduce clutter
    fmt=".2f",
    cmap='RdBu_r',  # Better color scheme
    center=0,  # Center colormap at 0
    square=True,  # Make cells square
    cbar_kws={"shrink": 0.8})

plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

# Alternative: Show only the strongest correlations
print("\n=== STRONGEST CORRELATIONS ===")
# Get correlations with target variable (assuming 'close' or similar)
target_cols = [
    col
    for col in corr.columns
    if 'close' in col.lower() or 'price_target' in col.lower()
]

if target_cols:
    target_col = target_cols[0]
    target_corr = corr[target_col].abs().sort_values(ascending=False)
    print(f"\nTop 15 features most correlated with {target_col}:")
    print(target_corr.head(15))

# Show pairs with highest absolute correlation (excluding diagonal)
print("\n=== HIGHEST FEATURE-TO-FEATURE CORRELATIONS ===")
corr_pairs = []
for i, feature1 in enumerate(corr.columns):
    for j, feature2 in enumerate(corr.columns[i+1:], start=i+1):
        corr_pairs.append({
            'feature1': feature1,
            'feature2': feature2,
            'correlation': corr.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs)
corr_df['abs_correlation'] = corr_df['correlation'].abs()
top_correlations = corr_df.nlargest(15, 'abs_correlation')
print("\nTop 15 strongest feature pairs:")
for _, row in top_correlations.iterrows():
    print(f"{row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")
