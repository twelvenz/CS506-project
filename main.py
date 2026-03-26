import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
SPOTIFY_FILES = [
    'data/spotify_music_dataset/spotify_data_clean.csv', 
    'data/spotify_music_dataset/track_data_final.csv', 
    'data/spotify_music_dataset/dataset.csv', 
    'data/spotify_music_dataset/spotify_2015_2025_85k.csv', 
    'data/spotify_music_dataset/spotify-tracks-dataset-detailed.csv'
]
SUPERBOWL_FILE = 'data/superbowl_halftime_shows/superbowl_halftime_performers.csv'

# Step 1: Identify data quality issues
def data_quality_check(df, stage="Initial"):
    report = {
        'total_records': len(df),
        'duplicate_track_ids': df.duplicated(subset=['track_id']).sum() if 'track_id' in df.columns else "N/A",
        'missing_values': df.isnull().sum().sum(),
    }
    print(f"\n--- Data Quality Report ({stage}) ---")
    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return report

# Step 2: Load and Align Datasets
def load_and_merge():
    print("Loading and aligning datasets...")
    dfs = []
    
    # Load each file and normalize key columns to match
    for file in SPOTIFY_FILES:
        try:
            temp_df = pd.read_csv(file)
            # Alignment mapping
            rename_map = {
                'artists': 'artist_name',
                'popularity': 'track_popularity',
                'album_release_date': 'release_date'
            }
            temp_df = temp_df.rename(columns=rename_map)
            dfs.append(temp_df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Combine all
    df_merged = pd.concat(dfs, ignore_index=True)
    
    # Sort by track_popularity and release_date to keep the best quality metadata when deduplicating
    # i.e. if there are duplicates of a track, you would want TODO: what is the code doing here?
    sort_cols = [c for c in ['track_id', 'track_popularity', 'release_date'] if c in df_merged.columns]
    df_merged = df_merged.sort_values(sort_cols, ascending=False)
    
    df_clean = df_merged.drop_duplicates(subset=['track_id'], keep='first').copy()
    return df_clean

# Step 3: Standardize text formatting
def standardize_text(df):
    print("Standardizing text...")
    text_cols = ['track_name', 'artist_name', 'album_name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

# Step 4: Handle missing values
def handle_missing_values(df):
    # Essential metadata
    df['artist_name'] = df['artist_name'].fillna("Unknown Artist")
    df['track_name'] = df['track_name'].fillna("Unknown Track")
    
    # Release date - if missing, flag it as "Unknown"
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].fillna("Unknown")
    return df

# Step 5: Filter invalid records
def filter_records(df):
    initial_count = len(df)
    # Filter out Soundtracks and Broadway casts which skew audio averages
    keywords = ['Soundtrack', 'Broadway Cast', 'Motion Picture', 'Musical', 'Theme']
    pattern = '|'.join(keywords)
    
    mask = (df['album_name'].str.contains(pattern, case=False, na=False) | 
            df['track_name'].str.contains(pattern, case=False, na=False) |
            df['artist_name'].str.contains('Cast', case=False, na=False))
    
    df_filtered = df[~mask].copy()
    print(f"Removed {initial_count - len(df_filtered)} soundtrack/non-studio records.")
    return df_filtered

# Step 6: Enrich with Target and Calculated Fields
def enrich_data(df):
    print("Enriching with Super Bowl labels and Billboard longevity...")
    
    # Load Super Bowl performers
    sb_df = pd.read_csv(SUPERBOWL_FILE)
    sb_performers = set()
    for col in ['headliners', 'guest performers']:
        for val in sb_df[col].dropna():
            names = [n.strip() for n in val.replace('"', '').replace('&', ',').split(',')]
            sb_performers.update(names)

    # Label Target Variable
    def check_sb(name_str):
        if pd.isna(name_str): return 0
        names = [n.strip() for n in name_str.replace(';', ',').split(',')]
        return 1 if any(n in sb_performers for n in names) else 0

    df['is_superbowl_performer'] = df['artist_name'].apply(check_sb)

    # Billboard longevity (Count how many years an artist appeared in Top 100)
    billboard_files = [f for f in os.listdir() if 'billboard_artists_year_end' in f]
    counts = {}
    for f in billboard_files:
        try:
            b_df = pd.read_csv(f, on_bad_lines='skip')
            for artist in b_df['artist'].unique():
                counts[artist] = counts.get(artist, 0) + 1
        except: continue
    
    df['years_on_billboard'] = df['artist_name'].map(counts).fillna(0)
    return df

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    # 1 & 2: Load and check initial
    main_df = load_and_merge()
    data_quality_check(main_df, "After Merge & Deduplication")

    # 3: Format
    main_df = standardize_text(main_df)

    # 4: Nulls
    main_df = handle_missing_values(main_df)

    # 5: Filter
    main_df = filter_records(main_df)

    # 6: Enrich
    main_df = enrich_data(main_df)

    # Final Summary
    data_quality_check(main_df, "Final Dataset")
    
    '''
    # Save Output
    main_df.to_csv('complete_training_table.csv', index=False)
    print("\nPipeline Complete. Saved to 'complete_training_table.csv'")
    '''


