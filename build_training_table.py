"""
build_training_table.py

Purpose:
  Fixes the broken "years_on_billboard" feature and builds a properly structured
  training table for the Roc Nation SB halftime prediction model.

Time-series Design:
  For each Roc Nation Super Bowl year Y (2019-2026), every candidate artist
  gets a feature snapshot computed STRICTLY from data available before year Y.
  This prevents leakage and enables forward-chaining cross-validation:

  One fold per year:
    Fold 1: Train 2019       --> Validate 2020 (JLo/Shakira)
    Fold 2: Train 2019-2020  --> Validate 2021 (The Weeknd)
    Fold 3: Train 2019-2021  --> Validate 2022 (Dr. Dre ensemble)
    Fold 4: Train 2019-2022  --> Validate 2023 (Rihanna)
    Fold 5: Train 2019-2023  --> Validate 2024 (Usher)
    Fold 6: Train 2019-2024  --> Validate 2025 (Kendrick Lamar)
    Live  : Train 2019-2025  --> Predict  2026

Leakage Disclosure/Limitation (documented per project decision):
  - Spotify features (artist_popularity, artist_followers, track_popularity,
    audio features) are CURRENT snapshots, not year-Y values.
  - These are used as static proxies. The model may slightly overestimate
    scores for artists who grew in popularity after year Y.
  - Billboard features (years_on_billboard, peak_rank, recency signals)
    ARE correctly time-gated to data before year Y.

OUTPUTS:
  - training_table.csv          : one row per (artist ? sb_year) - model input
  - artist_billboard_profiles.csv : artist-level Billboard summary (all years)
  - join_diagnostics.csv        : audit of name matching quality
----------------------------------------------------------------------------?
"""

import pandas as pd
import numpy as np
import re
import os

# Configs.
BILLBOARD_ARTISTS_FILE = 'billboard_artists_master.csv'
BILLBOARD_SONGS_FILE   = 'billboard_songs_master.csv'
SPOTIFY_FILE           = 'cleaned_Spotify_dataset.csv'
SUPERBOWL_FILE         = 'data/superbowl_halftime_shows/superbowl_halftime_performers.csv'

# Roc Nation era: SB LIII (Feb 2019) onward
ROC_NATION_SB_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PREDICT_YEAR        = 2026   # live prediction target

AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'loudness', 'tempo',
    'acousticness', 'speechiness', 'instrumentalness', 'liveness'
]

# Recency decay parameters
# Exponential decay applied to cumulative song score based on years since last chart.
# recency_weighted_score = song_cumulative_score * DECAY_FACTOR^(years_since_last_chart)
# Floor prevents artists from disappearing entirely (e.g. pre-2017 legacy acts).
DECAY_FACTOR       = 0.85   # 15% relevance lost per year of inactivity
RECENCY_FLOOR      = 0.15   # minimum decay multiplier --> no artist fully disappears from the prediction

# Legacy thresholds (no longer used in scoring b/c of emphasis on relevancy decay >> legacy)
# is_legacy_act is now only used as a soft eligibility signal, not a score component
LEGACY_MIN_TOP25_SONGS  = 3
LEGACY_MIN_CHART_YEARS  = 3
LEGACY_MIN_PEAK_SCORE   = 50


# Helper Functions
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def normalize_name(name):
    """
    Aggressive name normalization for fuzzy matching across datasets.
    Handles the, feat., featuring, &, punctuation differences.
    e.g. 'The Weeknd' ? 'weeknd'
         'Jay-Z'      ? 'jayz'
         'will.i.am'  ? 'william'
    """
    if pd.isna(name):
        return ''
    s = str(name).lower().strip()
    s = re.sub(r'\bthe\b', '', s)           # remove leading "the"
    s = re.sub(r'feat\.?.*$', '', s)        # strip features
    s = re.sub(r'featuring.*$', '', s)      # strip featuring
    s = re.sub(r'[^a-z0-9\s]', '', s)      # remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def split_artist_string(artist_str):
    """
    Splits multi-artist strings into individual names.
    Handles: feat., &, featuring, with, x, +, /, ,
    """
    if pd.isna(artist_str):
        return []
    delimiters = r' feat\.? | featuring | & | with | \bx\b | \+ | / | , | ; '
    standardized = re.sub(delimiters, '|', str(artist_str), flags=re.IGNORECASE)
    standardized = re.sub(r'["\(\)]', '', standardized)
    return [a.strip().lower() for a in standardized.split('|') if a.strip()]


def build_name_lookup(series):
    """
    Builds a dict mapping normalized_name ? original_name for a pd.Series.
    Used to cross-reference Billboard and Spotify artist names.
    """
    lookup = {}
    for name in series.dropna().unique():
        norm = normalize_name(name)
        if norm:
            lookup[norm] = name.lower().strip()
    return lookup


# Seciont 1: Loading the data
section("Section 1: Load Data")

df_billboard = pd.read_csv(BILLBOARD_ARTISTS_FILE)
df_songs     = pd.read_csv(BILLBOARD_SONGS_FILE)
df_spotify   = pd.read_csv(SPOTIFY_FILE)
df_sb        = pd.read_csv(SUPERBOWL_FILE)

# Standardize
df_billboard['artist'] = df_billboard['artist'].astype(str).str.strip().str.lower()
df_spotify['artist_name'] = df_spotify['artist_name'].astype(str).str.strip().str.lower()

# Songs master: explode multi-artist strings so each individual artist gets credit
# e.g. "drake feat. rihanna" -> both drake and rihanna get a row
def explode_song_artists(df):
    rows = []
    for _, row in df.iterrows():
        artists = split_artist_string(row['artist'])
        for a in artists:
            rows.append({'artist': a, 'year': row['year'], 'rank': row['rank']})
    return pd.DataFrame(rows)

df_songs_exploded = explode_song_artists(df_songs)
print(f"  Billboard artists master : {len(df_billboard):,} rows | years: "
      f"{df_billboard['year'].min()}-{df_billboard['year'].max()}")
print(f"  Billboard songs master   : {len(df_songs):,} rows -> "
      f"{len(df_songs_exploded):,} artist-song rows after exploding features")
print(f"  Spotify dataset          : {len(df_spotify):,} tracks | "
      f"{df_spotify['artist_name'].nunique()} unique artists")
print(f"  Super Bowl performers    : {len(df_sb)} shows")

# Section 2: Building the artist labels
#  (Roc Nation SB headliners + guest performers, per year)
section("Section 2: Build Per-Year SB Performer Labels")

def parse_sb_performers(df_sb, col='headliners'):
    """Returns dict: {year: set_of_normalized_names}"""
    result = {}
    for _, row in df_sb.iterrows():
        year = int(row['year'])
        names = set()
        for c in [col, 'guest performers']:
            if pd.notna(row.get(c, np.nan)):
                for n in row[c].replace('"', '').replace('&', ',').split(','):
                    norm = normalize_name(n.strip())
                    if norm:
                        names.add(norm)
                        names.add(n.strip().lower())  # also keep raw for exact match
        result[year] = names
    return result

sb_by_year = parse_sb_performers(df_sb)

# All Roc Nation performers (for labeling)
roc_nation_performers = set()
for year in ROC_NATION_SB_YEARS:
    roc_nation_performers.update(sb_by_year.get(year, set()))

print(f"  Roc Nation SB performers identified: {len(roc_nation_performers)}")
print(f"  Years covered: {ROC_NATION_SB_YEARS}")

# Section 3: Billboard Feature Engineering
#  Time-gated: all features computed as of year < Y
section("Section 3: Billboard Feature Engineering (Time-Gated)")

def compute_billboard_features_as_of(df_artists, df_songs_exp, artist_name, cutoff_year):
    """
    Computes all Billboard-derived features for an artist using only
    data from years STRICTLY BEFORE cutoff_year.

    Sources:
      df_artists   : billboard_artists_master (year-end artist chart)
      df_songs_exp : billboard_songs_master exploded to one row per artist
                     captures artists who charted via songs but not artist chart
                     (e.g. usher, jennifer lopez, dr. dre, shakira)

    Returns dict of features.
    """
    # Artist chart history (longevity / recency signal)
    hist_artists = df_artists[
        (df_artists['artist'] == artist_name) &
        (df_artists['year'] < cutoff_year)
    ]

    # Song chart history (peak song rank, hit count signal)
    hist_songs = df_songs_exp[
        (df_songs_exp['artist'] == artist_name) &
        (df_songs_exp['year'] < cutoff_year)
    ]

    # Combine both sources to determine overall chart presence
    # An artist "appears on Billboard" if they show up in EITHER chart in a given year
    artist_years = set(hist_artists['year'].unique())
    song_years   = set(hist_songs['year'].unique())
    all_chart_years = artist_years | song_years

    if not all_chart_years:
        return {
            'years_on_billboard'          : 0,
            'years_on_artist_chart'       : 0,
            'years_on_songs_chart'        : 0,
            'peak_artist_rank'            : np.nan,
            'peak_song_rank'              : np.nan,
            'last_billboard_year'         : np.nan,
            'years_since_last_chart'      : np.nan,
            'billboard_recency_score'     : 0.0,
            'legacy_adjusted_recency'     : 0.0,
            'recency_weighted_score'      : 0.0,
            'is_legacy_act'               : 0,
            'top10_artist_appearances'    : 0,
            'top10_song_appearances'      : 0,
            'top25_song_appearances'      : 0,
            'artist_peak_score'           : 0,
            'song_peak_score'             : 0,
            'song_cumulative_score'       : 0,
            'song_avg_score'              : 0,
        }

    last_year   = max(all_chart_years)
    years_since = cutoff_year - last_year

    # Step-function recency score (kept for reference / backward compat)
    recency_score = max(0.0, 1.0 - (years_since - 1) * 0.15)

    # Artist chart features
    peak_artist_rank  = hist_artists['rank'].min() if not hist_artists.empty else np.nan
    top10_artist      = int((hist_artists['rank'] <= 10).sum()) if not hist_artists.empty else 0
    artist_scores     = (101 - hist_artists['rank']).tolist() if not hist_artists.empty else [0]
    artist_peak_score = max(artist_scores)

    # Song chart features
    peak_song_rank        = hist_songs['rank'].min() if not hist_songs.empty else np.nan
    top10_songs           = int((hist_songs['rank'] <= 10).sum()) if not hist_songs.empty else 0
    top25_songs           = int((hist_songs['rank'] <= 25).sum()) if not hist_songs.empty else 0
    song_scores           = (101 - hist_songs['rank']).tolist() if not hist_songs.empty else [0]
    song_peak_score       = max(song_scores)
    song_cumulative_score = sum(song_scores)
    song_avg_score        = round(np.mean(song_scores), 2)

    # Exponential recency decay
    # decay_multiplier = DECAY_FACTOR^(years_since_last_chart), floored at RECENCY_FLOOR
    # An artist active last year gets multiplier ~0.85; 5 years ago ~0.44; 10 years ago ~0.20; etc.
    # The floor (0.15) ensures pre-2017 legacy acts are discounted but never fully zero
    decay_multiplier = max(RECENCY_FLOOR, DECAY_FACTOR ** years_since)

    # Recency-weighted cumulative score: the primary relevance signal
    # Combines chart dominance AND recency into one continuous measure.
    # A strong artist who charted recently scores high.
    # A strong artist who hasn't charted in years scores meaningfully but lower.
    # An artist with no chart history scores ~0 regardless.
    recency_weighted_score = round(song_cumulative_score * decay_multiplier, 4)

    # Legacy flag (kept for eligibility gating, no longer drives scoring directly)
    is_legacy = int(
        top25_songs >= LEGACY_MIN_TOP25_SONGS or
        len(all_chart_years) >= LEGACY_MIN_CHART_YEARS or
        song_peak_score >= LEGACY_MIN_PEAK_SCORE
    )

    # Legacy-adjusted recency (kept for backward compat)
    legacy_adjusted_recency = max(recency_score, 0.4) if is_legacy else recency_score

    return {
        'years_on_billboard'          : len(all_chart_years),
        'years_on_artist_chart'       : len(artist_years),
        'years_on_songs_chart'        : len(song_years),
        'peak_artist_rank'            : peak_artist_rank,
        'peak_song_rank'              : peak_song_rank,
        'last_billboard_year'         : last_year,
        'years_since_last_chart'      : years_since,
        'billboard_recency_score'     : round(recency_score, 4),
        'legacy_adjusted_recency'     : round(legacy_adjusted_recency, 4),
        'recency_weighted_score'      : recency_weighted_score,
        'is_legacy_act'               : is_legacy,
        'top10_artist_appearances'    : top10_artist,
        'top10_song_appearances'      : top10_songs,
        'top25_song_appearances'      : top25_songs,
        'artist_peak_score'           : int(artist_peak_score),
        'song_peak_score'             : int(song_peak_score),
        'song_cumulative_score'       : int(song_cumulative_score),
        'song_avg_score'              : song_avg_score,
    }


# Sanity Check on known artists
section_artists = ['drake', 'beyonce', 'taylor swift', 'the weeknd', 'usher']
print(f"\n  Sanity check - Billboard features as of 2025:")
print(f"  {'Artist':<25} {'Yrs':>4} {'ArtRk':>6} {'SongRk':>7} {'Recency':>8} {'DecayWt':>8} {'RWScore':>9}")
print(f"  {'-'*80}")
for a in section_artists:
    f = compute_billboard_features_as_of(df_billboard, df_songs_exploded, a, 2025)
    decay_mult = max(RECENCY_FLOOR, DECAY_FACTOR ** f['years_since_last_chart'])         if pd.notna(f['years_since_last_chart']) else RECENCY_FLOOR
    print(f"  {a:<25} {f['years_on_billboard']:>4} "
          f"{str(int(f['peak_artist_rank']) if pd.notna(f['peak_artist_rank']) else 'n/a'):>6} "
          f"{str(int(f['peak_song_rank'])) if pd.notna(f['peak_song_rank']) else 'n/a':>7} "
          f"{f['billboard_recency_score']:>8.2f} "
          f"{decay_mult:>8.2f} "
          f"{f['recency_weighted_score']:>9.1f}")

# Section 4: Spotify Feature Engineering
#  Artist-level aggregates (static snapshot - leakage documented)
section("Section 4: Spotify Feature Engineering (Static Snapshot)")

print("""
  [WARNING] LEAKAGE NOTE (documented):
     Spotify features below are current snapshots (as of dataset creation).
     They are NOT time-gated to pre-SB-year values.
     Impact: Model may slightly overestimate scores for artists who grew
     in popularity after the prediction year.
     Mitigation: Billboard features ARE correctly time-gated and carry
     the primary temporal signal. Spotify features serve as static
     proxies for scale and audio profile.
""")

# Build artist-level Spotify profile (one row per artist)
agg_dict = {
    'track_popularity'   : ['mean', 'max'],
    'artist_popularity'  : 'mean',
    'artist_followers'   : 'mean',
    'years_on_billboard' : 'max',   # keep the broken original (will be fixed/replaced later)
}
for feat in AUDIO_FEATURES:
    agg_dict[feat] = 'mean'

df_spotify_profile = df_spotify.groupby('artist_name').agg(agg_dict)
df_spotify_profile.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c
                               for c in df_spotify_profile.columns]
df_spotify_profile = df_spotify_profile.rename(columns={
    'track_popularity_mean' : 'avg_track_popularity',
    'track_popularity_max'  : 'peak_track_popularity',
    'artist_popularity_mean': 'artist_popularity',
    'artist_followers_mean' : 'artist_followers',
})

# Rename audio feature columns for clarity
for feat in AUDIO_FEATURES:
    old = f'{feat}_mean'
    if old in df_spotify_profile.columns:
        df_spotify_profile = df_spotify_profile.rename(columns={old: f'avg_{feat}'})

df_spotify_profile = df_spotify_profile.reset_index()
df_spotify_profile['artist_norm'] = df_spotify_profile['artist_name'].apply(normalize_name)

print(f"  Built Spotify profiles for {len(df_spotify_profile):,} unique artists")
audio_complete = df_spotify_profile[[f'avg_{f}' for f in AUDIO_FEATURES]].notna().all(axis=1).sum()
print(f"  Artists with full audio profiles : {audio_complete} ({audio_complete/len(df_spotify_profile)*100:.1f}%)")

# Section 5: Matching the names
#  Billboard-Spotify artist name alignment
section("Section 5: Name Matching Diagnostics")

bb_artists_all   = set(df_billboard['artist'].unique())
spot_artists_all = set(df_spotify_profile['artist_name'].unique())

# Exact match
exact_matches = bb_artists_all & spot_artists_all

# Normalized match (catches 'the weeknd' vs 'weeknd', punctuation diffs)
bb_norm_lookup   = build_name_lookup(pd.Series(list(bb_artists_all)))
spot_norm_lookup = build_name_lookup(pd.Series(list(spot_artists_all)))
norm_matches     = set(bb_norm_lookup.keys()) & set(spot_norm_lookup.keys())

print(f"  Billboard unique artists : {len(bb_artists_all):,}")
print(f"  Spotify unique artists   : {len(spot_artists_all):,}")
print(f"  Exact name matches       : {len(exact_matches):,}")
print(f"  Normalized matches       : {len(norm_matches):,}")
print(f"  Match rate (normalized)  : {len(norm_matches)/len(spot_artists_all)*100:.1f}% of Spotify artists found in Billboard")

# Flag Spotify artists NOT found in Billboard (they get years_on_billboard = 0)
unmatched = spot_artists_all - exact_matches
unmatched_sample = sorted(list(unmatched))[:20]
print(f"\n  Sample unmatched Spotify artists (first 20 of {len(unmatched)}):")
for a in unmatched_sample:
    print(f"    {a}")

# Save diagnostics
diag_rows = []
for artist in spot_artists_all:
    exact = artist in bb_artists_all
    norm  = normalize_name(artist) in bb_norm_lookup
    diag_rows.append({'artist_name': artist, 'exact_match': exact, 'norm_match': norm})
df_diag = pd.DataFrame(diag_rows)
'''
df_diag.to_csv('join_diagnostics.csv', index=False)
print(f"\n  Saved: join_diagnostics.csv")
'''

# Section 6: Building the training table
#  One row per (artist, sb_year)
section("Section 6: Build Time-Series Training Table")

all_years = ROC_NATION_SB_YEARS + [PREDICT_YEAR]

# Candidate pool: all Spotify artists (they already passed the Billboard filter
# during cleaned_Spotify_dataset creation)
candidate_artists = df_spotify_profile['artist_name'].tolist()

rows = []
total_iterations = len(candidate_artists) * len(all_years)
print(f"  Building {len(candidate_artists)} artists ? {len(all_years)} years "
      f"= {total_iterations:,} snapshot rows...")

for artist in candidate_artists:
    # Try exact match first, then normalized
    bb_name = artist  # default: use same name
    if artist not in bb_artists_all:
        norm = normalize_name(artist)
        if norm in bb_norm_lookup:
            bb_name = bb_norm_lookup[norm]

    # Spotify static features (same across all years - leakage documented)
    sp_row = df_spotify_profile[df_spotify_profile['artist_name'] == artist]
    if sp_row.empty:
        continue
    sp = sp_row.iloc[0]

    for sb_year in all_years:
        # Billboard features: time-gated to data before sb_year
        bb_feats = compute_billboard_features_as_of(df_billboard, df_songs_exploded, bb_name, sb_year)

        # Label: did this artist perform at the SB in sb_year?
        # For PREDICT_YEAR (2026), label is NaN (unknown)
        if sb_year == PREDICT_YEAR:
            label = np.nan
        else:
            year_performers = sb_by_year.get(sb_year, set())
            artist_norm = normalize_name(artist)
            label = 1 if (
                artist in year_performers or
                artist_norm in year_performers
            ) else 0

        row = {
            'artist_name'        : artist,
            'sb_year'            : sb_year,
            'is_sb_performer'    : label,

            # Billboard features (TIME-GATED - no leakage)
            'years_on_billboard'          : bb_feats['years_on_billboard'],
            'years_on_artist_chart'       : bb_feats['years_on_artist_chart'],
            'years_on_songs_chart'        : bb_feats['years_on_songs_chart'],
            'peak_artist_rank'            : bb_feats['peak_artist_rank'],
            'peak_song_rank'              : bb_feats['peak_song_rank'],
            'last_billboard_year'         : bb_feats['last_billboard_year'],
            'years_since_last_chart'      : bb_feats['years_since_last_chart'],
            'billboard_recency_score'     : bb_feats['billboard_recency_score'],
            'legacy_adjusted_recency'     : bb_feats['legacy_adjusted_recency'],
            'recency_weighted_score'      : bb_feats['recency_weighted_score'],
            'is_legacy_act'               : bb_feats['is_legacy_act'],
            'top10_artist_appearances'    : bb_feats['top10_artist_appearances'],
            'top10_song_appearances'      : bb_feats['top10_song_appearances'],
            'top25_song_appearances'      : bb_feats['top25_song_appearances'],
            'artist_peak_score'           : bb_feats['artist_peak_score'],
            'song_peak_score'             : bb_feats['song_peak_score'],
            'song_cumulative_score'       : bb_feats['song_cumulative_score'],
            'song_avg_score'              : bb_feats['song_avg_score'],

            # Spotify features (STATIC SNAPSHOT [WARNING] leakage documented) 
            'avg_track_popularity'    : sp.get('avg_track_popularity'),
            'peak_track_popularity'   : sp.get('peak_track_popularity'),
            'artist_popularity'       : sp.get('artist_popularity'),
            'artist_followers'        : sp.get('artist_followers'),
            'track_count'             : sp.get('track_count', np.nan),

            # Audio features (STATIC SNAPSHOT [WARNING] leakage documented) 
            **{f'avg_{f}': sp.get(f'avg_{f}') for f in AUDIO_FEATURES},

            # ?? Metadata ??
            'bb_name_used'       : bb_name,   # for audit - which Billboard name was matched
        }
        rows.append(row)

df_train = pd.DataFrame(rows)

# Section 7: Validation/Sanity Checks
section("Section 7: Validation")

total_rows      = len(df_train)
labeled_rows    = df_train[df_train['sb_year'] != PREDICT_YEAR]
predict_rows    = df_train[df_train['sb_year'] == PREDICT_YEAR]
positive_labels = labeled_rows['is_sb_performer'].sum()

print(f"  Total rows in training table      : {total_rows:,}")
print(f"  Labeled rows (2019-2025)          : {len(labeled_rows):,}")
print(f"  Prediction rows (2026, label=NaN) : {len(predict_rows):,}")
print(f"  Positive labels (SB performers)  : {int(positive_labels)}")
print(f"  Class balance                     : {positive_labels/len(labeled_rows)*100:.2f}% positive")

# Check known SB performers are correctly labeled
print(f"\n  Known Roc Nation SB performers - label check:")
print(f"  {'Artist':<25} {'SB Year':>8} {'Label':>6} {'Yrs BB':>7} {'PkSong':>9} {'Recency':>9} {'Legacy':>7}")
print(f"  {'-'*77}")

known_performers = {
    2019: ['jennifer lopez', 'shakira'],
    2020: ['jennifer lopez', 'shakira'],
    2021: ['the weeknd'],
    2022: ['dr. dre', 'snoop dogg', 'eminem', 'mary j. blige', 'kendrick lamar'],
    2023: ['rihanna'],
    2024: ['usher'],
    2025: ['kendrick lamar'],
}

for year, artists in known_performers.items():
    for artist in artists:
        # Use full name match to avoid partial matches (e.g. 'jennifer' matching 'jennifer lawrence')
        match = df_train[
            (df_train['artist_name'] == artist) &
            (df_train['sb_year'] == year)
        ]
        # Fall back to normalized match if exact fails
        if match.empty:
            artist_norm = normalize_name(artist)
            match = df_train[
                (df_train['artist_name'].apply(normalize_name) == artist_norm) &
                (df_train['sb_year'] == year)
            ]
        if not match.empty:
            r = match.iloc[0]
            print(f"  {r['artist_name']:<25} {year:>8} {str(int(r['is_sb_performer'])):>6} "
                  f"{r['years_on_billboard']:>7.0f} "
                  f"{str(int(r['peak_song_rank'])) if pd.notna(r['peak_song_rank']) else 'n/a':>9} "
                  f"{r['billboard_recency_score']:>9.2f} "
                  f"{str(int(r['is_legacy_act'])):>7}")
        else:
            print(f"  {artist:<25} {year:>8} {'NOT FOUND':>6}")

# Section8: Cross-validation fold previews
section("Section 8: Forward-Chaining CV Fold Structure")

print("""
  Forward-chaining cross-validation folds (time-series safe):

  Fold  Train Years          Validate Year   Positive Labels
  ----  ----------------     -------------   ---------------""")

fold_structure = [
    (1, [2019],                   2020),
    (2, [2019, 2020],             2021),
    (3, [2019, 2020, 2021],       2022),
    (4, [2019, 2020, 2021, 2022], 2023),
    (5, list(range(2019, 2024)),  2024),
    (6, list(range(2019, 2025)),  2025),
]

for fold, train_yrs, val_yr in fold_structure:
    train_pos = labeled_rows[
        labeled_rows['sb_year'].isin(train_yrs) &
        (labeled_rows['is_sb_performer'] == 1)
    ]['is_sb_performer'].sum()
    val_pos = labeled_rows[
        (labeled_rows['sb_year'] == val_yr) &
        (labeled_rows['is_sb_performer'] == 1)
    ]['is_sb_performer'].sum()
    train_str = f"{train_yrs[0]}-{train_yrs[-1]}" if len(train_yrs) > 1 else str(train_yrs[0])
    print(f"  {fold:<5} {train_str:<20} {val_yr:<15} "
          f"Train: {int(train_pos)} pos | Val: {int(val_pos)} pos")

print(f"\n  Live: Train 2019-2025 -> Predict 2026 (no label)")

'''
# Section 9: Saving the outputs
section("Section 9: Save Outputs")

# Full training table
df_train.to_csv('training_table.csv', index=False)
print(f"  Saved: training_table.csv  ({len(df_train):,} rows)")

# Artist-level Billboard profiles (for reference / feature inspection)
bb_profile_rows = []
for artist in candidate_artists:
    bb_name = artist
    if artist not in bb_artists_all:
        norm = normalize_name(artist)
        if norm in bb_norm_lookup:
            bb_name = bb_norm_lookup[norm]
    feats = compute_billboard_features_as_of(df_billboard, df_songs_exploded, bb_name, PREDICT_YEAR)
    bb_profile_rows.append({'artist_name': artist, 'bb_name_matched': bb_name, **feats})

df_bb_profiles = pd.DataFrame(bb_profile_rows).sort_values('years_on_billboard', ascending=False)
df_bb_profiles.to_csv('artist_billboard_profiles.csv', index=False)
print(f"  Saved: artist_billboard_profiles.csv  ({len(df_bb_profiles):,} artists)")

# 2026 prediction slice (for model scoring)
df_2026 = df_train[df_train['sb_year'] == PREDICT_YEAR].drop(columns=['is_sb_performer'])
df_2026.to_csv('prediction_candidates_2026.csv', index=False)
print(f"  Saved: prediction_candidates_2026.csv  ({len(df_2026):,} candidates)")
'''
