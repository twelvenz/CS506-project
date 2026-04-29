import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

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

# 3. AUDIO FEATURE RADAR CHART
# Summarizes average audio-feature profiles for performers and non-performers.
# The radar chart is useful for visual comparison, not direct model training.
categories = ['danceability', 'energy', 'valence', 'acousticness']
N = len(categories)

# Average each audio feature within the two label groups.
perf_means = df[df['is_superbowl_performer'] == 1][categories].mean().tolist()
non_perf_means = df[df['is_superbowl_performer'] == 0][categories].mean().tolist()

# Repeat the first point at the end so the radar chart closes cleanly.
perf_means += perf_means[:1]
non_perf_means += non_perf_means[:1]
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Draw both group profiles on the same polar chart.
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories)
ax.plot(angles, non_perf_means, linewidth=1, label='Non-Performers')
ax.fill(angles, non_perf_means, alpha=0.1)
ax.plot(angles, perf_means, linewidth=2, label='Performers', color='red')
ax.fill(angles, perf_means, color='red', alpha=0.2)

plt.title('The Mathematical Signature of a Performer')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()
