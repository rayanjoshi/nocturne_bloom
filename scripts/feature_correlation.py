from matplotlib import pyplot as plt
import pandas as pd
import scipy
import numpy as np
import seaborn as sns

df = pd.read_csv('../data/preprocessing/nvda_processed_data.csv', index_col=0, parse_dates=True)
corr = df.corr()

# Create a much larger figure to accommodate all features
plt.figure(dpi=100, figsize=(20, 16))

# Use smaller font size and remove annotations for better readability
sns.heatmap(corr, 
    annot=False,  # Remove numbers to reduce clutter
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
target_cols = [col for col in corr.columns if 'close' in col.lower() or 'Price_Target' in col.lower()]

if target_cols:
    target_col = target_cols[0]
    target_corr = corr[target_col].abs().sort_values(ascending=False)
    print(f"\nTop 15 features most correlated with {target_col}:")
    print(target_corr.head(15))

# Show pairs with highest absolute correlation (excluding diagonal)
print("\n=== HIGHEST FEATURE-TO-FEATURE CORRELATIONS ===")
corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        corr_pairs.append({
            'feature1': corr.columns[i],
            'feature2': corr.columns[j],
            'correlation': corr.iloc[i, j]
        })

corr_df = pd.df(corr_pairs)
corr_df['abs_correlation'] = corr_df['correlation'].abs()
top_correlations = corr_df.nlargest(15, 'abs_correlation')
print("\nTop 15 strongest feature pairs:")
for _, row in top_correlations.iterrows():
    print(f"{row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")