import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model-ready dataset used by the XGBoost pipeline.
df = pd.read_csv('final_training_dataset.csv')
sns.set_theme(style="whitegrid")

# 1. FEATURE CORRELATION HEATMAP
# Compares numeric features to see which ones move together.
# The upper triangle is hidden to avoid showing duplicate correlations.
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title("Updated Feature Correlation (Cleaned)")
plt.show()

# 2. POPULARITY COMPARISON
# Compares track popularity distributions for performers vs. non-performers.
# This checks whether Super Bowl artists tend to have more popular songs.
plt.figure(figsize=(8, 6))
sns.boxplot(x='is_superbowl_performer', y='track_popularity', data=df, palette='Set2')
plt.xticks([0, 1], ['Non-Performers', 'Performers'])
plt.title("Popularity Edge: Super Bowl vs. Regular Artists")
plt.show()

# 3. AUDIO FEATURE BAR CHART
# Grouped bars compare mean Spotify audio features for performers vs non-performers.
categories = ['danceability', 'energy', 'valence', 'acousticness']
perf_mean = df[df['is_superbowl_performer'] == 1][categories].mean()
non_perf_mean = df[df['is_superbowl_performer'] == 0][categories].mean()

plot_df = pd.DataFrame({
    "feature": list(categories) + list(categories),
    "group": ["Non-Performers"] * len(categories) + ["Performers"] * len(categories),
    "mean": list(non_perf_mean.values) + list(perf_mean.values),
})

plt.figure(figsize=(10, 6))
sns.barplot(
    data=plot_df,
    x="feature",
    y="mean",
    hue="group",
    palette=["#66c2a5", "#fc8d62"],
)
plt.xlabel("Audio feature")
plt.ylabel("Mean value (Spotify 0–1 scale)")
plt.title("Mean Audio Features: Performers vs Non-Performers")
plt.legend(title="")
plt.tight_layout()
plt.show()
