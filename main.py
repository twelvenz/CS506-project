import pandas as pd
import numpy as np
import os
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
'''
# Code for combining the Billboard data into master datasets; Created billboard_albums_master.csv, billboard_artists_master.csv, billboard_songs_master.csv

ALBUMS_PATTERN = 'data/billboard_albums_year_end_charts/billboard_albums_year_end_*.csv'
SONGS_PATTERN = 'data/billboard_songs_year_end_charts/billboard_songs_year_end_*.csv'
ARTISTS_PATTERN = 'data/billboard_artists_year_end_charts/billboard_artists_year_end_*.csv'

def combine_billboard_data(search_pattern):
    files = glob.glob(search_pattern)
    if not files:
        print(f"Warning: No files found for pattern: {search_pattern}")
        return pd.DataFrame()
    
    print(f"Combining {len(files)} files for pattern...")
    data_list = []
    
    # Determine expected columns based on the pattern
    if 'albums' in search_pattern:
        expected_cols = ['rank', 'album', 'artist']
    elif 'songs' in search_pattern:
        expected_cols = ['rank', 'song', 'artist']
    else: # artists
        expected_cols = ['rank', 'artist']

    # Check for input errors in reading the csv files
    def handle_bad_line(bad_line, filename):
        print(f"Parser Error in {filename}: Skipping malformed line -> {bad_line}")
        return None  # Returning None tells pandas to skip the line and continue

    for f in files:
        year_match = re.search(r'(\d{4})', os.path.basename(f))
        year = int(year_match.group(1)) if year_match else None
        
        try:
            # Enforce the schema using usecols=expected_cols
            df_year = pd.read_csv(
                f, 
                on_bad_lines=lambda line: handle_bad_line(line, f), 
                engine='python',
                usecols=lambda x: x.lower().strip() in expected_cols
            )
            
            # Standardize column names immediately to ensure alignment during concat
            df_year.columns = [c.lower().strip() for c in df_year.columns]
            
            df_year['year'] = year
            data_list.append(df_year)
        except Exception as e:
            print(f"Skipping file {f} due to error: {e}")
    
    if not data_list:
        return pd.DataFrame()

    combined = pd.concat(data_list, ignore_index=True)
    
    # Standardize column names
    combined.columns = [c.lower().strip() for c in combined.columns]
    
    # Clean up the artist column if it exists
    if 'artist' in combined.columns:
        combined['artist'] = combined['artist'].astype(str).str.lower().str.strip()
        # Optional: Remove the trailing quotes often found in these messy CSVs
        combined['artist'] = combined['artist'].str.replace('"', '', regex=False)
        
    return combined

# Step 1: Combine Billboard Master Datasets
print("--- Step 1: Billboard Master Combination ---")
df_billboard_albums = combine_billboard_data(ALBUMS_PATTERN)
df_billboard_songs = combine_billboard_data(SONGS_PATTERN)
df_billboard_artists = combine_billboard_data(ARTISTS_PATTERN)

# Save them to the project root
df_billboard_albums.to_csv('billboard_albums_master.csv', index=False)
df_billboard_songs.to_csv('billboard_songs_master.csv', index=False)
df_billboard_artists.to_csv('billboard_artists_master.csv', index=False)
'''

# Data Exploration: Billboard master datasets and Superbowl Halftime Performers

# Load the master datasets (assuming they are in the project root)
df_albums = pd.read_csv('billboard_albums_master.csv')
df_songs = pd.read_csv('billboard_songs_master.csv')
df_artists = pd.read_csv('billboard_artists_master.csv')
df_sb = pd.read_csv('data/superbowl_halftime_Headliners/superbowl_halftime_performers.csv')

def get_headliners(performer_str):
    """Splits comma-separated headliners into a clean list."""
    if pd.isna(performer_str): return []
    return [h.strip().lower() for h in str(performer_str).split(',')]

def calculate_score(rank):
    """Assigns 100 points for rank 1, 1 point for rank 100."""
    return 101 - rank

def analyze_claim(sb_start_year, match_type='exact'):
    """
    Analyzes headliner success for a specific SB cohort.
    match_type: 'exact' (for exact matching) or 'partial' (for partial partching; ex. "The Weeknd" --> "The Weeknd & Ariana Grande")
    """
    # Filter Super Bowls for the requested cohort
    cohort = df_sb[(df_sb['year'] >= sb_start_year) & (df_sb['year'] <= 2025)].copy()
    
    results = []
    
    for _, row in cohort.iterrows():
        sb_year = row['year']
        headliners = get_headliners(row['headliners'])
        
        # We only consider Billboard data BEFORE the Super Bowl year
        hist_songs = df_songs[df_songs['year'] < sb_year]
        hist_albums = df_albums[df_albums['year'] < sb_year]
        hist_artists = df_artists[df_artists['year'] < sb_year]
        
        show_stats = {
            'year': sb_year,
            'has_rank1_song': False,
            'has_top10_song': False,
            'has_rank1_album': False,
            'has_top10_album': False,
            'has_rank1_artist': False,
            'has_top10_artist': False,
            'scores_songs': [],
            'scores_albums': [],
            'scores_artists': []
        }
        
        for artist in headliners:
            # Matching logic
            if match_type == 'exact':
                s_mask = hist_songs['artist'] == artist
                a_mask = hist_albums['artist'] == artist
                ar_mask = hist_artists['artist'] == artist
            else: # partial
                s_mask = hist_songs['artist'].str.contains(artist, na=False)
                a_mask = hist_albums['artist'].str.contains(artist, na=False)
                ar_mask = hist_artists['artist'].str.contains(artist, na=False)
            
            # 1. Song Stats
            artist_songs = hist_songs[s_mask]
            if not artist_songs.empty:
                if (artist_songs['rank'] == 1).any(): show_stats['has_rank1_song'] = True
                if (artist_songs['rank'] <= 10).any(): show_stats['has_top10_song'] = True
                show_stats['scores_songs'].extend(artist_songs['rank'].apply(calculate_score).tolist())
            
            # 2. Album Stats
            artist_albums = hist_albums[a_mask]
            if not artist_albums.empty:
                if (artist_albums['rank'] == 1).any(): show_stats['has_rank1_album'] = True
                if (artist_albums['rank'] <= 10).any(): show_stats['has_top10_album'] = True
                show_stats['scores_albums'].extend(artist_albums['rank'].apply(calculate_score).tolist())
                
            # 3. Artist Stats (2017-2025)
            artist_rankings = hist_artists[ar_mask]
            if not artist_rankings.empty:
                if (artist_rankings['rank'] == 1).any(): show_stats['has_rank1_artist'] = True
                if (artist_rankings['rank'] <= 10).any(): show_stats['has_top10_artist'] = True
                show_stats['scores_artists'].extend(artist_rankings['rank'].apply(calculate_score).tolist())
        
        results.append(show_stats)
    
    # Aggregating final percentages for the cohort
    res_df = pd.DataFrame(results)
    summary = {
        "Cohort": f"{sb_start_year}-2025",
        "Match Type": match_type.capitalize(),
        "% Headliners with #1 Song": res_df['has_rank1_song'].mean() * 100,
        "% Headliners with Top 10 Song": res_df['has_top10_song'].mean() * 100,
        "% Headliners with #1 Album": res_df['has_rank1_album'].mean() * 100,
        "% Headliners with Top 10 Album": res_df['has_top10_album'].mean() * 100,
        "% Headliners with #1 Artist": res_df['has_rank1_artist'].mean() * 100,
        "% Headliners with Top 10 Artist": res_df['has_top10_artist'].mean() * 100,
        "Avg Song Score": np.mean([s for sublist in res_df['scores_songs'] for s in sublist]) if any(res_df['scores_songs']) else 0,
        "Avg Album Score": np.mean([s for sublist in res_df['scores_albums'] for s in sublist]) if any(res_df['scores_albums']) else 0,
        "Avg Artist Score": np.mean([s for sublist in res_df['scores_artists'] for s in sublist]) if any(res_df['scores_artists']) else 0,
    }
    return summary

# Run analysis for all permutations requested
final_stats = []
for year in [2019, 2020]:
    for m_type in ['exact', 'partial']:
        final_stats.append(analyze_claim(year, m_type))

# Display the results
df_final = pd.DataFrame(final_stats).round(2)
print("--- Roc Nation Era Super Bowl Data Exploration ---")
print(df_final.to_string(index=False))

# Set the visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 7]

def plot_billboard_exploration(df):
    # 1. Visualize Percentages (Success Rate)
    # We melt the dataframe to make it "long-form" for Seaborn
    pct_cols = [
        '% Headliners with #1 Song', '% Headliners with Top 10 Song',
        '% Headliners with #1 Album', '% Headliners with Top 10 Album',
        '% Headliners with #1 Artist', '% Headliners with Top 10 Artist'
    ]
    
    df_melted = df.melt(
        id_vars=['Cohort', 'Match Type'], 
        value_vars=pct_cols, 
        var_name='Metric', 
        value_name='Percentage'
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_melted, 
        x='Metric', 
        y='Percentage', 
        hue='Cohort', 
        palette='viridis'
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Billboard Success Rates: 2019-2025 vs 2020-2025 Cohorts', fontsize=15)
    plt.ylabel('Percentage of Headliners (%)')
    plt.legend(title='Start Year')
    plt.tight_layout()
    plt.show()

    # 2. Visualize Average Scores (Pedigree/Strength)
    score_cols = ['Avg Song Score', 'Avg Album Score', 'Avg Artist Score']
    df_scores = df.melt(
        id_vars=['Cohort', 'Match Type'], 
        value_vars=score_cols, 
        var_name='Category', 
        value_name='Score'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_scores, 
        x='Category', 
        y='Score', 
        hue='Match Type', 
        palette='magma'
    )
    plt.title('Average "Power Score" of Headliners (100 = Rank 1, 1 = Rank 100)', fontsize=15)
    plt.ylabel('Mean Normalized Score')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Execute the plots
plot_billboard_exploration(df_final)


'''
# --- CONFIGURATION ---
SPOTIFY_FILES = [
    'data/spotify_music_dataset/spotify_data_clean.csv', 
    'data/spotify_music_dataset/track_data_final.csv', 
    'data/spotify_music_dataset/dataset.csv', 
    'data/spotify_music_dataset/spotify_2015_2025_85k.csv', 
    'data/spotify_music_dataset/spotify-tracks-dataset-detailed.csv'
]
SUPERBOWL_FILE = 'data/superbowl_halftime_Headliners/superbowl_halftime_performers.csv'

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
    
    # Save Output
    main_df.to_csv('complete_training_table.csv', index=False)
    print("\nPipeline Complete. Saved to 'complete_training_table.csv'")
'''
    