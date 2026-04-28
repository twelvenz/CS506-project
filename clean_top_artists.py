import pandas as pd
import numpy as np

# 1. Load the dataset
print("Loading top_artists_dataset.csv...")
df = pd.read_csv('top_artists_dataset.csv', low_memory=False)
print(f"Original shape: {df.shape} (Rows, Columns)")

# 2. CLEAN TRACK NAMES and ARTIST NAMES
# Lowercase and strip whitespace so variations group together perfectly
df['track_name_clean'] = df['track_name'].astype(str).str.lower()
df['track_name_clean'] = df['track_name_clean'].str.replace(r' - remaster.*| \(feat\..*| \[.*', '', regex=True).str.strip()
df['artist_name_clean'] = df['artist_name'].astype(str).str.lower().str.strip()

# Check release_date to handle "Unknown" cleanly before using min()
# Replace 'Unknown' with NaN temporarily so min() ignores it
if 'release_date' in df.columns:
    df['release_date_temp'] = df['release_date'].replace({'Unknown': np.nan, 'unknown': np.nan})

# 3. DEFINE AGGREGATION RULES
# We dynamically check if the column exists in your CSV before adding a rule for it
agg_rules = {}

if 'track_popularity' in df.columns: agg_rules['track_popularity'] = 'max'
if 'stream_count' in df.columns: agg_rules['stream_count'] = 'max'
if 'release_date_temp' in df.columns: agg_rules['release_date_temp'] = lambda x: x.dropna().min() if not x.dropna().empty else np.nan

# Artist stats (Keep the highest recorded values)
if 'artist_popularity' in df.columns: agg_rules['artist_popularity'] = 'max'
if 'artist_followers' in df.columns: agg_rules['artist_followers'] = 'max'
if 'years_on_billboard' in df.columns: agg_rules['years_on_billboard'] = 'max'
if 'is_superbowl_performer' in df.columns: agg_rules['is_superbowl_performer'] = 'max'

# Audio features (Average them across the duplicates)
audio_features = ['danceability', 'energy', 'valence', 'loudness', 'tempo', 
                  'duration_ms', 'acousticness', 'speechiness', 'instrumentalness', 'liveness']
for feat in audio_features:
    if feat in df.columns:
        agg_rules[feat] = 'mean'

# Keep one valid track ID
if 'track_id' in df.columns: agg_rules['track_id'] = 'first'

# 4. PERFORM THE GROUPING AND AGGREGATION
print("\nCollapsing duplicates using the Hybrid Approach...")
grouped_df = df.groupby(['track_name_clean', 'artist_name_clean']).agg(agg_rules).reset_index()

# Rename our temporary cleaned columns back to the standard names
grouped_df = grouped_df.rename(columns={
    'track_name_clean': 'track_name', 
    'artist_name_clean': 'artist_name',
    'release_date_temp': 'release_date'
})

# Fill any remaining NaNs in release_date back to the string 'Unknown'
if 'release_date' in grouped_df.columns:
    grouped_df['release_date'] = grouped_df['release_date'].fillna('Unknown')

print(f"Deduplicated shape: {grouped_df.shape} (Rows, Columns)")

# 5. SAVE TO A NEW CSV
output_filename = 'final_training_dataset.csv'
grouped_df.to_csv(output_filename, index=False)
print(f"\nSUCCESS: Saved fully optimized dataset to: {output_filename}")