import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configs.
SPOTIFY_FILE        = 'cleaned_Spotify_dataset.csv'
SUPERBOWL_FILE      = 'data/superbowl_halftime_shows/superbowl_halftime_performers.csv'
ROC_NATION_START    = 2019   # when Roc Nation took over

AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'loudness',
    'tempo', 'acousticness', 'speechiness', 'instrumentalness', 'liveness'
]

ARTIST_FEATURES = [
    'track_popularity', 'artist_popularity', 'artist_followers', 'years_on_billboard'
]


# Helper Functions
def section(title):
    """Prints a clear section divider for readability in the console."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def data_quality_report(df, stage=""):
    """Prints a concise data quality snapshot at any pipeline stage."""
    section(f"Data Quality Report -- {stage}")
    print(f"  Shape            : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Duplicate rows   : {df.duplicated().sum():,}")
    print(f"  Missing values   : {df.isnull().sum().sum():,} total cells")
    print()

    # Per-column missingness (only show columns with any nulls)
    missing = df.isnull().mean().mul(100).round(1)
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print("  Missing % by column:")
        for col, pct in missing.items():
            print(f"    {col:<30} {pct:.1f}%")
    else:
        print("  No missing values.")

    print()
    print("  Dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<30} {dtype}")


def load_sb_performers(sb_file):
    """
    Loads the Super Bowl performers CSV and returns two sets:
      - all_performers   : every name who ever performed (headliner or guest performer)
      - roc_nation_pool  : headliners + guests from Roc Nation era (>= ROC_NATION_START)
    Names are lowercased to match artist_name formatting in the Spotify dataset.
    """
    sb_df = pd.read_csv(sb_file)

    def parse_names(series):
        names = set()
        for val in series.dropna():
            for n in val.replace('"', '').replace('&', ',').split(','):
                cleaned = n.strip().lower()
                if cleaned:
                    names.add(cleaned)
        return names

    all_performers = set()
    roc_nation_pool = set()

    for _, row in sb_df.iterrows():
        year = row.get('year', 0)
        headliners_str  = row.get('headliners', '')
        guests_str      = row.get('guest performers', '')

        row_names = parse_names(pd.Series([headliners_str, guests_str]))
        all_performers.update(row_names)

        if year >= ROC_NATION_START:
            roc_nation_pool.update(row_names)

    return all_performers, roc_nation_pool


def flag_sb_membership(artist_name_str, performer_set):
    """Returns 1 if any artist in a (possibly multi-artist) string is in performer_set."""
    if pd.isna(artist_name_str):
        return 0
    parts = [p.strip().lower() for p in str(artist_name_str).replace(';', ',').split(',')]
    return 1 if any(p in performer_set for p in parts) else 0


# Section 1: Load the Spotify dataset
section("Section 1: Load Dataset")
df = pd.read_csv(SPOTIFY_FILE)

# Ensure artist_name is lowercase (consistent with Billboard pipeline)
df['artist_name'] = df['artist_name'].astype(str).str.strip().str.lower()
df['track_name']  = df['track_name'].astype(str).str.strip().str.lower()

# Parse release_date -> release_year (integer, nullable)
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

data_quality_report(df, "Initial Load")


# Section 2: Checking for missing values
section("Section 2: Audio Feature Missingness Analysis")

audio_missing = df[AUDIO_FEATURES].isnull().mean().mul(100).sort_values(ascending=False)
print("  Audio feature missingness (%):")
print(audio_missing.to_string())

# How many Spotify songs/tracks have a full audio profile vs partial vs none?
df['audio_completeness'] = df[AUDIO_FEATURES].notnull().sum(axis=1)
complete    = (df['audio_completeness'] == len(AUDIO_FEATURES)).sum()
partial     = ((df['audio_completeness'] > 0) & (df['audio_completeness'] < len(AUDIO_FEATURES))).sum()
none_       = (df['audio_completeness'] == 0).sum()
print(f"\n  Tracks with ALL audio features   : {complete:,} ({complete/len(df)*100:.1f}%)")
print(f"  Tracks with SOME audio features  : {partial:,} ({partial/len(df)*100:.1f}%)")
print(f"  Tracks with NO audio features    : {none_:,} ({none_/len(df)*100:.1f}%)")

# Heatmap: missingness pattern across audio features
fig, ax = plt.subplots(figsize=(10, 4))
missing_matrix = df[AUDIO_FEATURES].isnull().astype(int)
sns.heatmap(
    missing_matrix.T,
    cmap='Reds', cbar=False, yticklabels=AUDIO_FEATURES,
    xticklabels=False, ax=ax
)
ax.set_title('Audio Feature Missingness Heatmap (Red = Missing)', fontsize=13)
ax.set_xlabel('Tracks')
plt.tight_layout()
'''
plt.savefig('spotify_missing_heatmap.png', dpi=150)
plt.close()
print("\n  Saved: spotify_missing_heatmap.png")
'''


# Section 3: Looking at data distribution
section("Section 3: Dataset Overview — Key Distributions")

# 3a. Track popularity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['track_popularity'].dropna(), bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Track Popularity Distribution', fontsize=12)
axes[0].set_xlabel('Popularity Score (0–100)')
axes[0].set_ylabel('# Tracks')

axes[1].hist(df['artist_popularity'].dropna(), bins=40, color='mediumpurple', edgecolor='white')
axes[1].set_title('Artist Popularity Distribution', fontsize=12)
axes[1].set_xlabel('Popularity Score (0–100)')
axes[1].set_ylabel('# Artists')

plt.suptitle('Popularity Distributions in Spotify Dataset', fontsize=14)
plt.tight_layout()
'''
plt.savefig('spotify_popularity_distributions.png', dpi=150)
plt.close()
print("  Saved: spotify_popularity_distributions.png")
'''

# 3b. Release year distribution (volume over time)
fig, ax = plt.subplots(figsize=(14, 5))
year_counts = df['release_year'].dropna().astype(int)
year_counts = year_counts[(year_counts >= 1960) & (year_counts <= 2025)]
year_counts.value_counts().sort_index().plot(kind='bar', ax=ax, color='teal', edgecolor='white', width=0.8)
ax.axvline(
    x=list(year_counts.value_counts().sort_index().index).index(ROC_NATION_START)
      if ROC_NATION_START in year_counts.value_counts().index else 0,
    color='red', linestyle='--', linewidth=1.5, label=f'Roc Nation Era ({ROC_NATION_START}+)'
)
ax.set_title('Track Releases by Year in Dataset', fontsize=13)
ax.set_xlabel('Release Year')
ax.set_ylabel('# Tracks')
ax.legend()
plt.tight_layout()
'''
plt.savefig('spotify_releases_by_year.png', dpi=150)
plt.close()
print("  Saved: spotify_releases_by_year.png")
'''

# 3c. Audio feature distributions (for tracks WITH data)
df_audio = df.dropna(subset=AUDIO_FEATURES)
print(f"\n  Using {len(df_audio):,} tracks with complete audio features for distribution plots.")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
colors = sns.color_palette("husl", len(AUDIO_FEATURES))

for i, feat in enumerate(AUDIO_FEATURES):
    axes[i].hist(df_audio[feat], bins=40, color=colors[i], edgecolor='white', alpha=0.85)
    axes[i].set_title(feat.capitalize(), fontsize=11)
    axes[i].set_ylabel('# Tracks')

plt.suptitle('Audio Feature Distributions (Complete Records Only)', fontsize=14)
plt.tight_layout()
'''
plt.savefig('spotify_audio_feature_distributions.png', dpi=150)
plt.close()
print("  Saved: spotify_audio_feature_distributions.png")
'''

# 3d. Correlation matrix among audio + popularity features
corr_cols = AUDIO_FEATURES + ['track_popularity', 'artist_popularity']
df_corr = df[corr_cols].dropna()
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, linewidths=0.5, ax=ax
)
ax.set_title('Feature Correlation Matrix (Audio + Popularity)', fontsize=13)
plt.tight_layout()
'''
plt.savefig('spotify_correlation_matrix.png', dpi=150)
plt.close()
print("  Saved: spotify_correlation_matrix.png")
'''

# 3e. Top artists by track count in the dataset
top_artists = df['artist_name'].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 6))
top_artists.sort_values().plot(kind='barh', ax=ax, color='slateblue', edgecolor='white')
ax.set_title('Top 20 Artists by Track Count in Dataset', fontsize=13)
ax.set_xlabel('# Tracks')
plt.tight_layout()
'''
plt.savefig('spotify_top_artists_track_count.png', dpi=150)
plt.close()
print("  Saved: spotify_top_artists_track_count.png")
'''


# Section 4: Label Superbowl Membership 
section("Section 4: Label Superbowl Membership")

all_sb, roc_sb = load_sb_performers(SUPERBOWL_FILE)

# Sanity Check: Validate against existing is_superbowl_performer column
df['is_sb_roc_nation'] = df['artist_name'].apply(lambda x: flag_sb_membership(x, roc_sb))

# Cross-check: how many tracks in dataset belong to Roc Nation-era SB performers?
sb_track_count     = df['is_superbowl_performer'].sum()
roc_track_count    = df['is_sb_roc_nation'].sum()
total_tracks       = len(df)

print(f"  Total tracks                          : {total_tracks:,}")
print(f"  Tracks by any SB performer            : {int(sb_track_count):,} ({sb_track_count/total_tracks*100:.1f}%)")
print(f"  Tracks by Roc Nation SB performer     : {int(roc_track_count):,} ({roc_track_count/total_tracks*100:.1f}%)")


# Section 5: ROC NATION-ERA: SB vs NON-SB
#              Artist-level comparison
section("Section 5: Roc Nation Era -- SB vs Non-SB Artist Comparison")

# Build an artist-level summary table
def build_artist_summary(df_input, sb_label_col):
    """
    Aggregates track-level data into artist-level metrics.
    Returns one row per artist with avg/peak/count metrics.
    """
    grp = df_input.groupby('artist_name')
    summary = pd.DataFrame({
        'track_count'            : grp['track_name'].count(),
        'avg_track_popularity'   : grp['track_popularity'].mean(),
        'peak_track_popularity'  : grp['track_popularity'].max(),
        'avg_artist_popularity'  : grp['artist_popularity'].mean(),
        'avg_artist_followers'   : grp['artist_followers'].mean(),
        'years_on_billboard'     : grp['years_on_billboard'].max(),
        'avg_danceability'       : grp['danceability'].mean(),
        'avg_energy'             : grp['energy'].mean(),
        'avg_valence'            : grp['valence'].mean(),
        'avg_acousticness'       : grp['acousticness'].mean(),
        'avg_speechiness'        : grp['speechiness'].mean(),
        'avg_liveness'           : grp['liveness'].mean(),
        sb_label_col             : grp[sb_label_col].max()
    }).reset_index()
    return summary

# Use Roc Nation-era label for the SB vs non-SB split
df_artist = build_artist_summary(df, 'is_sb_roc_nation')

sb_artists     = df_artist[df_artist['is_sb_roc_nation'] == 1]
non_sb_artists = df_artist[df_artist['is_sb_roc_nation'] == 0]

print(f"\n  Roc Nation SB artist profiles    : {len(sb_artists)}")
print(f"  Non-SB artist profiles           : {len(non_sb_artists)}")

# ── 5a. Side-by-side median comparison table ──
compare_cols = [
    'avg_track_popularity', 'peak_track_popularity',
    'avg_artist_popularity', 'avg_artist_followers',
    'years_on_billboard', 'track_count',
    'avg_danceability', 'avg_energy', 'avg_valence'
]

comparison = pd.DataFrame({
    'SB Performer (Roc Nation)' : sb_artists[compare_cols].median(),
    'Non-SB Artist'             : non_sb_artists[compare_cols].median()
})
comparison['SB Advantage'] = (
    (comparison['SB Performer (Roc Nation)'] - comparison['Non-SB Artist'])
    / comparison['Non-SB Artist'].replace(0, np.nan) * 100
).round(1).astype(str) + '%'

print("\n  Median Comparison Table (SB Roc Nation vs Non-SB):")
print(comparison.to_string())

# ── 5b. Violin plots: audio features SB vs non-SB ──
audio_compare = ['avg_danceability', 'avg_energy', 'avg_valence',
                 'avg_acousticness', 'avg_speechiness']

df_plot = df_artist[['is_sb_roc_nation'] + audio_compare].copy()
df_plot['Group'] = df_plot['is_sb_roc_nation'].map({1: 'SB Performer\n(Roc Nation)', 0: 'Non-SB Artist'})
df_melt = df_plot.melt(id_vars='Group', value_vars=audio_compare, var_name='Feature', value_name='Value')
df_melt['Feature'] = df_melt['Feature'].str.replace('avg_', '').str.capitalize()

fig, axes = plt.subplots(1, len(audio_compare), figsize=(18, 6), sharey=False)
palette = {'SB Performer\n(Roc Nation)': '#E63946', 'Non-SB Artist': '#457B9D'}

for i, feat in enumerate(audio_compare):
    feat_label = feat.replace('avg_', '').capitalize()
    feat_data  = df_melt[df_melt['Feature'] == feat_label]
    sns.violinplot(
        data=feat_data, x='Group', y='Value',
        palette=palette, inner='box', cut=0, ax=axes[i]
    )
    axes[i].set_title(feat_label, fontsize=11)
    axes[i].set_xlabel('')
    if i > 0:
        axes[i].set_ylabel('')

plt.suptitle('Audio Feature Profiles: SB Performers (Roc Nation Era) vs Non-SB Artists', fontsize=13)
plt.tight_layout()
'''
plt.savefig('spotify_sb_vs_nonsb_audio_violins.png', dpi=150)
plt.close()
print("\n  Saved: spotify_sb_vs_nonsb_audio_violins.png")
'''

# ── 5c. Popularity & scale comparison ──
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, col, title, color in zip(
    axes,
    ['avg_track_popularity', 'avg_artist_popularity', 'avg_artist_followers'],
    ['Avg Track Popularity', 'Avg Artist Popularity', 'Avg Artist Followers'],
    ['steelblue', 'mediumpurple', 'seagreen']
):
    data_sb     = sb_artists[col].dropna()
    data_non_sb = non_sb_artists[col].dropna()

    ax.boxplot(
        [data_non_sb, data_sb],
        labels=['Non-SB', 'SB\n(Roc Nation)'],
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.6),
        medianprops=dict(color='black', linewidth=2)
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(col)

plt.suptitle('Artist Scale: Roc Nation SB Performers vs Non-SB Artists', fontsize=13)
plt.tight_layout()
'''
plt.savefig('spotify_sb_vs_nonsb_scale.png', dpi=150)
plt.close()
print("  Saved: spotify_sb_vs_nonsb_scale.png")
'''

# ── 5d. Years on Billboard vs Track Popularity scatter ──
fig, ax = plt.subplots(figsize=(10, 7))

non_sb_plot = non_sb_artists[non_sb_artists['years_on_billboard'] > 0]
sb_plot     = sb_artists[sb_artists['years_on_billboard'] > 0]

ax.scatter(
    non_sb_plot['years_on_billboard'], non_sb_plot['avg_track_popularity'],
    alpha=0.3, s=20, color='#457B9D', label='Non-SB Artist'
)
ax.scatter(
    sb_plot['years_on_billboard'], sb_plot['avg_track_popularity'],
    alpha=0.9, s=80, color='#E63946', zorder=5, label='SB Performer (Roc Nation)'
)

# Annotate SB performers
for _, row in sb_plot.iterrows():
    ax.annotate(
        row['artist_name'].title(),
        xy=(row['years_on_billboard'], row['avg_track_popularity']),
        fontsize=7, alpha=0.8,
        xytext=(4, 2), textcoords='offset points'
    )

ax.set_xlabel('Years on Billboard (Top 100 Artist Chart)')
ax.set_ylabel('Avg Track Popularity (Spotify)')
ax.set_title('Billboard Longevity vs Spotify Popularity\n(Roc Nation SB Performers Highlighted)', fontsize=12)
ax.legend()
plt.tight_layout()
'''
plt.savefig('spotify_billboard_longevity_vs_spotify_popularity.png', dpi=150)
plt.close()
print("  Saved: spotify_billboard_longevity_vs_spotify_popularity.png")
'''

# Section 6: Roc Nation Artists Ranking
#  (mirror of Billboard scoring logic)
section("Section 6: Roc Nation Candidate Scoring")

'''
Scoring mirrors the Billboard EDA philosophy:
  - Peak Strategy    : best single track's popularity score
  - Average Strategy : mean popularity across all tracks
  - Cumulative       : sum of all popularity scores (dominance proxy)

Only consider tracks released BEFORE 2025 (no future leakage).
Only consider non-SB artists (the candidates we'd want to predict).
'''

df_candidates = df[
    (df['is_superbowl_performer'] == 0) &
    (df['release_year'].notna()) &
    (df['release_year'] < 2025)
].copy()

candidate_scores = df_candidates.groupby('artist_name').agg(
    track_count              = ('track_name',         'count'),
    peak_track_popularity    = ('track_popularity',   'max'),
    avg_track_popularity     = ('track_popularity',   'mean'),
    cumulative_popularity    = ('track_popularity',   'sum'),
    avg_artist_popularity    = ('artist_popularity',  'mean'),
    avg_artist_followers     = ('artist_followers',   'mean'),
    years_on_billboard       = ('years_on_billboard', 'max'),
    avg_danceability         = ('danceability',       'mean'),
    avg_energy               = ('energy',             'mean'),
    avg_valence              = ('valence',             'mean'),
).reset_index()

# Composite score: weighted blend matching Billboard high-peak model
# Weights reflect Roc Nation's preference for elite, culturally dominant artists
candidate_scores['composite_score'] = (
    0.40 * candidate_scores['peak_track_popularity'] +
    0.25 * candidate_scores['avg_artist_popularity'].fillna(0) +
    0.20 * candidate_scores['avg_track_popularity'] +
    0.15 * candidate_scores['years_on_billboard'].clip(0, 20) * 5  # normalize ~0-100
)

top_candidates = (
    candidate_scores
    .sort_values('composite_score', ascending=False)
    .head(30)
    .reset_index(drop=True)
)
top_candidates.index += 1  # rank starts at 1

print("\n  Top 30 Non-SB Candidates by Composite Score:")
print(top_candidates[['artist_name', 'composite_score', 'peak_track_popularity',
                       'avg_artist_popularity', 'years_on_billboard', 'track_count']].to_string())

# Save the ranked candidates
top_candidates.to_csv('spotify_top_candidates.csv', index_label='rank')
print("\n  Saved: spotify_top_candidates.csv")

# ── Visualize top 20 candidates ──
fig, ax = plt.subplots(figsize=(12, 8))
plot_data = top_candidates.head(20).sort_values('composite_score')
colors_bar = ['#E63946' if v >= plot_data['composite_score'].quantile(0.75) else '#457B9D'
              for v in plot_data['composite_score']]
ax.barh(plot_data['artist_name'].str.title(), plot_data['composite_score'],
        color=colors_bar, edgecolor='white')
ax.set_title('Top 20 Non-SB Artist Candidates — Spotify Composite Score\n(Roc Nation Era Model)',
             fontsize=13)
ax.set_xlabel('Composite Score')
plt.tight_layout()
'''
plt.savefig('spotify_top20_candidates.png', dpi=150)
plt.close()
print("  Saved: spotify_top20_candidates.png")
'''


# Spotify EDA Summary
section("EDA Summary")

"""
---- Dataset Overview ----
The cleaned Spotify dataset contains 5,877 tracks from Billboard Top 100 artists or artists with a Top 100 song.
A significant share of tracks are missing audio features (danceability, energy, etc.),
likely due to variations/incompleteness across the source Kaggle datasets. Artist-level metadata 
(artist_popularity, artist_followers) is more complete and more reliable for modeling.

---- Key Findings ----
1. Roc Nation SB performers cluster at the TOP of both Spotify and Billboard metrics:
   - Higher peak track popularity and artist popularity than non-SB artists
   - More years on the Billboard Top 100 Artist chart
   - Larger follower bases — a proxy for cultural scale

2. Audio profiles show moderate separation:
   - SB performers trend slightly higher in danceability and speechiness
   - Valence and acousticness shows less separation - SB selection is not purely about 'feel-good' music
   - Acousticness is notably lower for SB performers (high-production, concert-ready sound)

3. The Billboard longevity x Spotify popularity scatter is a strong signal:
   - Artists with 3+ Billboard years AND high Spotify popularity form a distinct cluster
   - This cluster almost entirely contains known or plausible SB performers

---- Modeling Implications ----
Strongest candidate features for prediction:
  - peak_track_popularity      (Spotify)
  - avg_artist_popularity      (Spotify)
  - avg_artist_followers       (Spotify)
  - years_on_billboard         (Billboard)
  - avg_energy / avg_danceability (audio — where available)
  - composite_score            (engineered; best single feature)

Audio features (danceability, energy, etc.) should be treated as optional enrichment
given their ~50% missingness rate. Use them when present; do not require them.

Output files:
  - spotify_missing_heatmap.png
  - spotify_popularity_distributions.png
  - spotify_releases_by_year.png
  - spotify_audio_feature_distributions.png
  - spotify_correlation_matrix.png
  - spotify_top_artists_track_count.png
  - spotify_sb_vs_nonsb_audio_violins.png
  - spotify_sb_vs_nonsb_scale.png
  - spotify_billboard_longevity_vs_spotify_popularity.png
  - spotify_top20_candidates.png
  - spotify_top_candidates.csv       ← top 30 ranked non-SB candidates
  - cleaned_Spotify_dataset_enriched.csv  ← input for modeling
"""


"""
More personal notes:
    - Gloria Estefan seems to be an outlier? --> does not appear as a top 100 artist or an artist with a top 100 song
        - This could be because our billboard_artists_year_end_charts does not extend far back enough or that she might 
        be a special case as a negative sample where Roc Nation wanted her for the Latina trio act in 2020
            - Solution: Either ignore the outlier or add a cultural feature to account for Roc Nation's official offer to
            Gloria Estefan; Consider much, much older rankings (i.e. 80s/90s; Billboard Hot 100 charts started on 1958)
        - Furthermore, it seems that the Spotify datasets (accumulated in the uncleaned_Spotify_dataset.csv) does not
        have many records/songs by Gloria Estefan --> this indicates the limitations of using Kaggle Spotify datasets
        in building this model
            - Solution: Try working with Spotify Web API to build a more accurate dataset
        - Might have to consider extending the scope to top artists/top songs in Latin America (i.e. other regions
        besides the U.S.)
"""