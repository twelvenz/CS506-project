"""
sb_scoring_model.py
------------------------------------------------------------------------------
Roc Nation Super Bowl Halftime Performer Scoring System
------------------------------------------------------------------------------

APPROACH:
  Transparent weighted composite scoring system inspired by Schmitz (2020).
  No black-box ML -- every score is explainable and auditable.

SCORING COMPONENTS (total = 100 points):
  1. Billboard Dominance    (25pts) -- peak song rank + cumulative hit score
  2. Artist Scale           (20pts) -- followers (log) + Spotify popularity
  3. Career Longevity       (15pts) -- combined years on both Billboard charts
  4. Legacy & Relevance     (15pts) -- legacy flag x recency-adjusted score
  5. Audio Profile Match    (10pts) -- danceability + speechiness vs SB avg
  6. Catalog Strength       (10pts) -- avg + peak track popularity
  7. Home Ground Bonus      ( 5pts) -- city/state tie to SB host location

BONUS (additive, outside the 100pt base):
  Guest Alumni Bonus: artists who previously guested at a Roc Nation SB
  receive +5 pts. Pattern supported by Bad Bunny (guest 2020 -> headliner 2026)
  and Kendrick Lamar (guest 2022 -> headliner 2025).

FILTERS:
  - Past HEADLINERS are permanently excluded (no artist has ever headlined twice)
  - Deceased artists are excluded
  - Artists inactive for >RECENCY_WINDOW years AND not a legacy act: -40% penalty

HOME GROUND:
  Computed dynamically from artists_hometown.csv vs superbowl_halftime_locations.csv
  Tiers: hometown (+5), region/same state (+3), west coast (+1.5), national (0)

INPUTS:
  training_table.csv
  superbowl_halftime_performers.csv
  artists_hometown.csv
  superbowl_halftime_locations.csv

OUTPUTS:
  sb_lxi_scores.csv
  sb_lxi_top20.png
  sb_lxi_score_breakdown.png
  sb_lxi_audio_scatter.png
  sb_lxi_summary.txt
------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import re
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------
TRAINING_FILE    = 'training_table.csv'
PERFORMERS_FILE  = 'data/superbowl_halftime_shows/superbowl_halftime_performers.csv'
HOMETOWN_FILE    = 'data/artists_hometown/artists_hometown.csv'
SB_LOCS_FILE     = 'data/superbowl_halftime_shows/superbowl_halftime_locations.csv'

PREDICT_YEAR     = 2027   # SB LXI -- Feb 14 2027, SoFi Stadium, Inglewood CA
RECENCY_WINDOW   = 5      # years without charting = recency penalty (unless legacy)
RECENCY_PENALTY  = 0.40   # fraction score reduction for inactive non-legacy artists
GUEST_BONUS      = 5.0    # additive pts for past guest alumni

# Score component weights (must sum to 100)
WEIGHTS = {
    'billboard_dominance' : 25,
    'artist_scale'        : 20,
    'career_longevity'    : 15,
    'legacy_relevance'    : 15,
    'audio_profile'       : 10,
    'catalog_strength'    : 10,
    'home_ground_bonus'   :  5,
}
assert sum(WEIGHTS.values()) == 100

HOME_GROUND_TIERS = {
    'hometown'  : 5.0,
    'region'    : 3.0,
    'westcoast' : 1.5,
    'national'  : 0.0,
}
WEST_COAST_STATES = {'CA', 'OR', 'WA', 'NV', 'AZ'}

DECEASED_ARTISTS = {
    '2pac', 'xxxtentacion', 'juice wrld', 'mac miller',
    'pop smoke', 'prince', 'amy winehouse', 'michael jackson',
    'nipsey hussle', 'king von', 'nate dogg',
}

# SB audio benchmark -- computed from actual Roc Nation headliner catalog averages
# Source: training_table.csv headliner rows with audio data
# Included: Bad Bunny, Dr. Dre, Eminem, Jennifer Lopez, Kendrick Lamar,
#           Mary J. Blige, Shakira, Snoop Dogg, The Weeknd
# Missing (no audio data in dataset): Rihanna, Usher, Maroon 5
# Method: per-artist mean first, then mean across artists to avoid
#         track-count bias
SB_AUDIO_BENCHMARK = {
    'avg_danceability' : 0.7247,
    'avg_speechiness'  : 0.1273,
}
BENCHMARK_ARTISTS = [
    'bad bunny', 'dr. dre', 'eminem', 'jennifer lopez', 'kendrick lamar',
    'mary j. blige', 'shakira', 'snoop dogg', 'the weeknd',
]
BENCHMARK_MISSING = ['rihanna', 'usher', 'maroon 5']  # no audio data in dataset

STATE_ABBREV = {
    'CALIFORNIA':'CA','TEXAS':'TX','NEW YORK':'NY','FLORIDA':'FL',
    'GEORGIA':'GA','TENNESSEE':'TN','NEVADA':'NV','WASHINGTON':'WA',
    'OREGON':'OR','ARIZONA':'AZ','LOUISIANA':'LA','ILLINOIS':'IL',
    'MICHIGAN':'MI','OHIO':'OH','PENNSYLVANIA':'PA',
    'NORTH CAROLINA':'NC','VIRGINIA':'VA',
}


# ------------------------------------------------------------------------------
#  HELPERS
# ------------------------------------------------------------------------------
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def parse_names(cell):
    """Split a comma-separated names cell into a list of lowercase strings."""
    if not cell or str(cell).strip() in ('', 'nan'):
        return []
    return [n.strip().strip('"').lower()
            for n in re.split(r',', str(cell)) if n.strip()]


def load_sb_performer_sets(performers_file, predict_year):
    """
    Reads superbowl_halftime_performers.csv and returns:
      past_headliners : set of all headliners from years < predict_year
                        (permanently excluded -- no artist headlines twice)
      past_guests     : set of all guests from years < predict_year
                        (eligible but get GUEST_BONUS added to score)
    """
    df = pd.read_csv(performers_file)
    df_past = df[df['year'] < predict_year]

    past_headliners = set()
    past_guests     = set()

    for _, row in df_past.iterrows():
        for name in parse_names(row.get('headliners', '')):
            past_headliners.add(name)
        for name in parse_names(row.get('guest performers', '')):
            past_guests.add(name)

    # Guests who later became headliners should be in headliners set only
    past_guests -= past_headliners

    return past_headliners, past_guests


def minmax_scale(series, new_min=0.0, new_max=1.0):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series([new_max] * len(series), index=series.index)
    return new_min + (series - s_min) / (s_max - s_min) * (new_max - new_min)


def compute_home_ground_tier(artist_state, artist_city, sb_state, sb_city):
    if pd.isna(artist_state) or str(artist_state).strip() in ('', 'nan'):
        return 'national'
    a_state = STATE_ABBREV.get(str(artist_state).strip().upper(),
                                str(artist_state).strip().upper())
    a_city  = str(artist_city).strip().lower() if pd.notna(artist_city) else ''
    s_state = STATE_ABBREV.get(str(sb_state).strip().upper(),
                                str(sb_state).strip().upper())
    s_city  = str(sb_city).strip().lower()   if pd.notna(sb_city)  else ''

    if a_city and s_city and a_city == s_city:
        return 'hometown'
    if a_state == s_state:
        return 'region'
    if s_state == 'CA' and a_state in WEST_COAST_STATES:
        return 'westcoast'
    if s_state == 'NV' and a_state in {'CA', 'AZ', 'UT', 'OR', 'WA'}:
        return 'westcoast'
    if s_state == 'GA' and a_state in {'FL', 'SC', 'NC', 'TN', 'AL'}:
        return 'westcoast'
    return 'national'


def audio_profile_score(row, benchmark, weight):
    scores = []
    if pd.notna(row.get('avg_danceability')):
        dist = abs(row['avg_danceability'] - benchmark['avg_danceability'])
        scores.append((max(0.0, 1.0 - dist / benchmark['avg_danceability']), 0.60))
    if pd.notna(row.get('avg_speechiness')):
        dist = abs(row['avg_speechiness'] - benchmark['avg_speechiness'])
        scores.append((max(0.0, 1.0 - dist / benchmark['avg_speechiness']), 0.40))
    if not scores:
        return weight * 0.5  # neutral score for missing audio data
    total_w = sum(w for _, w in scores)
    return (sum(s * w for s, w in scores) / total_w) * weight


# ------------------------------------------------------------------------------
#  SECTION 1 -- LOAD DATA
# ------------------------------------------------------------------------------
section("Section 1: Load Data")

df_train     = pd.read_csv(TRAINING_FILE)
df_hometown  = pd.read_csv(HOMETOWN_FILE)
df_sb_locs   = pd.read_csv(SB_LOCS_FILE)

df_train['artist_name']   = df_train['artist_name'].astype(str).str.strip().str.lower()
df_hometown['artist']     = df_hometown['artist'].astype(str).str.strip().str.lower()

# Load SB host city for prediction year
sb_loc = df_sb_locs[df_sb_locs['year'] == PREDICT_YEAR]
SB_STATE = sb_loc.iloc[0]['state'] if not sb_loc.empty else None
SB_CITY  = sb_loc.iloc[0]['city']  if not sb_loc.empty else None

# Dynamically build exclusion + guest sets from performers CSV
past_headliners, past_guests = load_sb_performer_sets(PERFORMERS_FILE, PREDICT_YEAR)

print(f"  Training rows          : {len(df_train):,}")
print(f"  SB {PREDICT_YEAR} location    : {SB_CITY}, {SB_STATE}")
print(f"  Past headliners (excluded) : {sorted(h.title() for h in past_headliners)}")
print(f"  Past guests (bonus +{GUEST_BONUS}pts)  : {sorted(g.title() for g in past_guests)}")

# Build prediction slice
df_pred = df_train[df_train['sb_year'] == PREDICT_YEAR].copy()
if df_pred.empty:
    print(f"  [INFO] No {PREDICT_YEAR} rows -- using 2026 features as proxy")
    df_pred = df_train[df_train['sb_year'] == 2026].copy()
    df_pred['sb_year'] = PREDICT_YEAR

print(f"  Prediction candidates  : {len(df_pred):,}")

# ------------------------------------------------------------------------------
#  SECTION 2 -- FILTER CANDIDATES
# ------------------------------------------------------------------------------
section("Section 2: Filter Candidates")

df_pred['is_past_headliner'] = df_pred['artist_name'].isin(past_headliners)
df_pred['is_deceased']       = df_pred['artist_name'].isin(DECEASED_ARTISTS)
df_pred['is_past_guest']     = df_pred['artist_name'].isin(past_guests)

excluded_headliners = df_pred[df_pred['is_past_headliner']]['artist_name'].tolist()
excluded_deceased   = df_pred[df_pred['is_deceased']]['artist_name'].tolist()
guest_alumni        = df_pred[df_pred['is_past_guest']]['artist_name'].tolist()

df_eligible = df_pred[
    ~df_pred['is_past_headliner'] & ~df_pred['is_deceased']
].copy().reset_index(drop=True)

print(f"  Excluded -- past headliners : {len(excluded_headliners)}")
print(f"    {sorted([a.title() for a in excluded_headliners])}")
print(f"  Excluded -- deceased        : {len(excluded_deceased)}")
print(f"  Guest alumni in pool        : {len(guest_alumni)}")
print(f"    {sorted([a.title() for a in guest_alumni])}")
print(f"  Eligible candidates         : {len(df_eligible)}")

# Recency flag
df_eligible['last_chart_year']    = df_eligible['last_billboard_year'].fillna(0)
df_eligible['is_recently_active'] = (
    (PREDICT_YEAR - df_eligible['last_chart_year'] <= RECENCY_WINDOW) |
    (df_eligible['is_legacy_act'] == 1)
).astype(int)
inactive = (~df_eligible['is_recently_active'].astype(bool)).sum()
print(f"  Inactive artists (will be penalized) : {inactive}")

# ------------------------------------------------------------------------------
#  SECTION 3 -- JOIN HOMETOWN DATA
# ------------------------------------------------------------------------------
section("Section 3: Join Hometown Data")

df_eligible = df_eligible.merge(
    df_hometown[['artist', 'hometown_state', 'hometown_city']],
    left_on='artist_name', right_on='artist', how='left'
).drop(columns=['artist'])

matched   = df_eligible['hometown_state'].notna().sum()
unmatched = df_eligible['hometown_state'].isna().sum()
print(f"  Hometown matched   : {matched}")
print(f"  Hometown unmatched : {unmatched} (assigned national tier)")

if SB_STATE:
    df_eligible['home_ground_tier'] = df_eligible.apply(
        lambda r: compute_home_ground_tier(
            r['hometown_state'], r['hometown_city'], SB_STATE, SB_CITY
        ), axis=1
    )
else:
    df_eligible['home_ground_tier'] = 'national'

df_eligible['comp_home_ground'] = (
    df_eligible['home_ground_tier'].map(HOME_GROUND_TIERS).fillna(0.0)
)

print(f"\n  Home ground tier distribution:")
for tier, count in df_eligible['home_ground_tier'].value_counts().items():
    print(f"    {tier:<12} : {count}")

# ------------------------------------------------------------------------------
#  SECTION 4 -- COMPUTE SCORE COMPONENTS
# ------------------------------------------------------------------------------
section("Section 4: Compute Score Components")

df = df_eligible.copy()

# Component 1: Billboard Dominance (25 pts)
df['song_peak_inv'] = df['peak_song_rank'].apply(
    lambda x: (101 - x) if pd.notna(x) and x > 0 else 0
)
df['comp_billboard_dominance'] = (
    0.50 * minmax_scale(df['song_peak_inv'].fillna(0)) +
    0.30 * minmax_scale(df['song_cumulative_score'].fillna(0)) +
    0.20 * minmax_scale(df['top10_song_appearances'].fillna(0))
) * WEIGHTS['billboard_dominance']

# Component 2: Artist Scale (20 pts)
df['followers_log'] = np.log1p(df['artist_followers'].fillna(0))
df['comp_artist_scale'] = (
    0.60 * minmax_scale(df['followers_log']) +
    0.40 * minmax_scale(df['artist_popularity'].fillna(0))
) * WEIGHTS['artist_scale']

# Component 3: Career Longevity (15 pts)
df['total_chart_years'] = (
    df['years_on_artist_chart'].fillna(0) +
    df['years_on_songs_chart'].fillna(0)
)
df['comp_career_longevity'] = (
    minmax_scale(df['total_chart_years']) * WEIGHTS['career_longevity']
)

# Component 4: Legacy & Relevance (15 pts)
df['comp_legacy_relevance'] = (
    0.60 * minmax_scale(df['legacy_adjusted_recency'].fillna(0)) +
    0.40 * df['is_legacy_act'].fillna(0)
) * WEIGHTS['legacy_relevance']

# Component 5: Audio Profile Match (10 pts)
# Danceability (60%) + Speechiness (40%) proximity to SB benchmark
df['comp_audio_profile'] = df.apply(
    lambda r: audio_profile_score(r, SB_AUDIO_BENCHMARK, WEIGHTS['audio_profile']),
    axis=1
)

# Component 6: Catalog Strength (10 pts)
df['comp_catalog_strength'] = (
    0.50 * minmax_scale(df['avg_track_popularity'].fillna(0)) +
    0.50 * minmax_scale(df['peak_track_popularity'].fillna(0))
) * WEIGHTS['catalog_strength']

# Component 7: Home Ground (5 pts) -- already computed in Section 3

# Raw base score
score_cols = [
    'comp_billboard_dominance', 'comp_artist_scale', 'comp_career_longevity',
    'comp_legacy_relevance', 'comp_audio_profile', 'comp_catalog_strength',
    'comp_home_ground',
]
df['base_score'] = df[score_cols].sum(axis=1)

# Recency penalty
df['recency_penalty_applied'] = (~df['is_recently_active'].astype(bool)).astype(int)
df['penalized_score'] = df.apply(
    lambda r: r['base_score'] * (1 - RECENCY_PENALTY)
    if r['recency_penalty_applied'] == 1 else r['base_score'],
    axis=1
)

# Guest alumni bonus (+5 pts additive -- outside the 100pt base)
df['comp_guest_bonus'] = df['is_past_guest'].apply(
    lambda x: GUEST_BONUS if x else 0.0
)
df['total_score'] = df['penalized_score'] + df['comp_guest_bonus']

df['rank'] = df['total_score'].rank(ascending=False, method='min').astype(int)
df_ranked  = df.sort_values('total_score', ascending=False).reset_index(drop=True)
df_ranked.index += 1

penalized_count = df_ranked['recency_penalty_applied'].sum()
guest_count     = df_ranked['comp_guest_bonus'].gt(0).sum()
print(f"  Recency penalty applied : {penalized_count} artists")
print(f"  Guest alumni bonus      : {guest_count} artists")
print(f"  Score range             : {df_ranked['total_score'].min():.2f} - "
      f"{df_ranked['total_score'].max():.2f}")

# ------------------------------------------------------------------------------
#  SECTION 5 -- AUDIO PROFILE ANALYSIS
# ------------------------------------------------------------------------------
section("Section 5: Audio Profile Analysis")

print(f"  SB Audio Benchmark (recalibrated from actual headliner data):")
print(f"    Danceability : {SB_AUDIO_BENCHMARK['avg_danceability']:.4f}")
print(f"    Speechiness  : {SB_AUDIO_BENCHMARK['avg_speechiness']:.4f}")
print(f"    Derived from : {', '.join(a.title() for a in BENCHMARK_ARTISTS)}")
print(f"    Missing data : {', '.join(a.title() for a in BENCHMARK_MISSING)} (no audio in dataset)")

print(f"\n  {'Metric':<22} {'Top20 avg':>10} {'Pool avg':>10} {'Benchmark':>10} {'Gap':>8}")
print(f"  {'-'*63}")
for feat, bench in SB_AUDIO_BENCHMARK.items():
    t20  = df_ranked.head(20)[feat].mean() if feat in df_ranked.columns else np.nan
    all_ = df_ranked[feat].mean()          if feat in df_ranked.columns else np.nan
    gap  = t20 - bench if not (t20 != t20) else float('nan')
    print(f"  {feat:<22} {t20:>10.3f} {all_:>10.3f} {bench:>10.4f} {gap:>+8.3f}")

# ------------------------------------------------------------------------------
#  SECTION 6 -- DISPLAY TOP 30
# ------------------------------------------------------------------------------
section(f"Section 6: Top 30 Candidates -- SB LXI {SB_CITY}, {SB_STATE} {PREDICT_YEAR}")

print(f"\n  * = recency penalty (-{int(RECENCY_PENALTY*100)}%)  "
      f"G = guest alumni bonus (+{GUEST_BONUS}pts)\n")
hdr = (f"  {'Rk':<4} {'Artist':<22} {'Score':>7} "
       f"{'BbDom':>6} {'Scale':>6} {'Long':>5} {'Lgcy':>5} "
       f"{'Audio':>5} {'Cat':>5} {'Home':>5} {'Gst':>4} "
       f"{'BBYrs':>6} {'Lgcy?':>6} {'Rcnt?':>6} {'Tier'}")
print(hdr)
print(f"  {'-'*120}")

for i, row in df_ranked.head(30).iterrows():
    penalty_flag = '*' if row['recency_penalty_applied'] == 1 else ' '
    guest_flag   = 'G' if row['comp_guest_bonus'] > 0      else ' '
    print(f"  {i:<4} "
          f"{str(row['artist_name']).title():<22} "
          f"{row['total_score']:>6.2f}{penalty_flag} "
          f"{row['comp_billboard_dominance']:>6.2f} "
          f"{row['comp_artist_scale']:>6.2f} "
          f"{row['comp_career_longevity']:>5.2f} "
          f"{row['comp_legacy_relevance']:>5.2f} "
          f"{row['comp_audio_profile']:>5.2f} "
          f"{row['comp_catalog_strength']:>5.2f} "
          f"{row['comp_home_ground']:>5.2f} "
          f"{row['comp_guest_bonus']:>4.1f} "
          f"{row.get('years_on_billboard',0):>6.0f} "
          f"{int(row.get('is_legacy_act',0)):>6} "
          f"{int(row.get('is_recently_active',0)):>6} "
          f"{row['home_ground_tier']}")

# ------------------------------------------------------------------------------
#  SECTION 7 -- HOME GROUND ANALYSIS
# ------------------------------------------------------------------------------
section(f"Section 7: Home Ground Analysis -- {SB_CITY}, {SB_STATE}")

local = df_ranked[
    df_ranked['home_ground_tier'].isin(['hometown','region','westcoast'])
].head(60)
print(f"\n  {'Artist':<25} {'Tier':<12} {'Rank':>5} {'Score':>7} "
      f"{'City':<20} {'State'}")
print(f"  {'-'*80}")
for _, row in local.iterrows():
    print(f"  {str(row['artist_name']).title():<25} "
          f"{row['home_ground_tier']:<12} "
          f"{int(row['rank']):>5} "
          f"{row['total_score']:>7.2f} "
          f"{str(row.get('hometown_city','')).title():<20} "
          f"{str(row.get('hometown_state',''))}")

if SB_STATE == 'CA':
    print(f"\n  NOTE: SB LVI (2022) was also at SoFi Stadium with west coast hip-hop theme.")
    print(f"  Roc Nation will likely pursue a DIFFERENT cultural angle for LXI.")

# ------------------------------------------------------------------------------
#  SECTION 8 -- SAVE OUTPUTS
# ------------------------------------------------------------------------------
section("Section 8: Save Outputs")

out_cols = [
    'rank','artist_name','total_score','base_score','penalized_score',
    'recency_penalty_applied','comp_guest_bonus',
    'comp_billboard_dominance','comp_artist_scale','comp_career_longevity',
    'comp_legacy_relevance','comp_audio_profile','comp_catalog_strength',
    'comp_home_ground','home_ground_tier','hometown_state','hometown_city',
    'years_on_billboard','peak_song_rank','song_cumulative_score',
    'is_legacy_act','is_recently_active','is_past_guest','legacy_adjusted_recency',
    'artist_popularity','artist_followers',
    'avg_track_popularity','peak_track_popularity',
    'avg_danceability','avg_speechiness',
]
out_cols = [c for c in out_cols if c in df_ranked.columns]
df_ranked[out_cols].to_csv('sb_lxi_scores.csv', index=False)
print(f"  Saved: sb_lxi_scores.csv")

color_map = {
    'hometown':'#E63946','region':'#F4A261',
    'westcoast':'#2A9D8F','national':'#457B9D'
}

# Top 20 bar chart
top20  = df_ranked.head(20).copy()
colors = [color_map.get(t,'#457B9D') for t in top20['home_ground_tier']]
fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(
    top20['artist_name'].str.title()[::-1],
    top20['total_score'][::-1],
    color=colors[::-1], edgecolor='white', height=0.7
)
# Mark guest alumni with a star annotation
for bar, (_, row) in zip(bars[::-1], top20[::-1].iterrows()):
    if row['comp_guest_bonus'] > 0:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                'Guest Alumni', va='center', fontsize=7,
                color='#E63946', fontweight='bold')
ax.set_xlabel('Total Score (out of 100 + guest bonus)', fontsize=11)
ax.set_title(f'Top 20 Predicted Candidates -- SB LXI\n'
             f'{SB_CITY}, {SB_STATE} | Feb 14, {PREDICT_YEAR}', fontsize=13)
patches = [
    mpatches.Patch(color='#E63946', label=f'Hometown ({SB_CITY})'),
    mpatches.Patch(color='#F4A261', label=f'Same State ({SB_STATE})'),
    mpatches.Patch(color='#2A9D8F', label='West Coast'),
    mpatches.Patch(color='#457B9D', label='National/Intl'),
]
ax.legend(handles=patches, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('sb_lxi_top20.png', dpi=150)
plt.close()
print(f"  Saved: sb_lxi_top20.png")

# Stacked score breakdown top 15
top15 = df_ranked.head(15).copy()
comp_map = {
    'comp_billboard_dominance':'BB Dominance',
    'comp_artist_scale':'Artist Scale',
    'comp_career_longevity':'Career Longevity',
    'comp_legacy_relevance':'Legacy & Relevance',
    'comp_audio_profile':'Audio Profile',
    'comp_catalog_strength':'Catalog Strength',
    'comp_home_ground':'Home Ground',
    'comp_guest_bonus':'Guest Alumni Bonus',
}
comp_colors = ['#E63946','#457B9D','#2A9D8F','#F4A261',
               '#6A0572','#888888','#F4D03F','#C0392B']
fig, ax = plt.subplots(figsize=(13, 8))
bottoms = np.zeros(15)
for (col, label), color in zip(comp_map.items(), comp_colors):
    if col not in top15.columns:
        continue
    vals = top15[col].values
    ax.barh(top15['artist_name'].str.title().tolist()[::-1], vals[::-1],
            left=bottoms[::-1], label=label, color=color,
            edgecolor='white', height=0.7)
    bottoms += vals
ax.set_xlabel('Score Breakdown', fontsize=11)
ax.set_title('Score Component Breakdown -- Top 15 SB LXI Candidates\n'
             'Audio = Danceability + Speechiness vs SB Benchmark', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('sb_lxi_score_breakdown.png', dpi=150)
plt.close()
print(f"  Saved: sb_lxi_score_breakdown.png")

# Danceability vs Speechiness scatter
top50 = df_ranked.head(50).copy()
fig, ax = plt.subplots(figsize=(10, 8))
sc = [color_map.get(t,'#457B9D') for t in top50['home_ground_tier']]
ax.scatter(
    top50['avg_danceability'].fillna(SB_AUDIO_BENCHMARK['avg_danceability']),
    top50['avg_speechiness'].fillna(SB_AUDIO_BENCHMARK['avg_speechiness']),
    c=sc, s=70, alpha=0.85, edgecolors='white', linewidths=0.5
)
ax.axvline(SB_AUDIO_BENCHMARK['avg_danceability'], color='red',
           linestyle='--', linewidth=1.2, alpha=0.7, label='SB Benchmark')
ax.axhline(SB_AUDIO_BENCHMARK['avg_speechiness'], color='red',
           linestyle='--', linewidth=1.2, alpha=0.7)
for _, row in top50.head(15).iterrows():
    ax.annotate(
        row['artist_name'].title().split()[0],
        xy=(row['avg_danceability'] if pd.notna(row['avg_danceability'])
            else SB_AUDIO_BENCHMARK['avg_danceability'],
            row['avg_speechiness']  if pd.notna(row['avg_speechiness'])
            else SB_AUDIO_BENCHMARK['avg_speechiness']),
        fontsize=7, alpha=0.9, xytext=(4,2), textcoords='offset points'
    )
ax.set_xlabel('Avg Danceability', fontsize=11)
ax.set_ylabel('Avg Speechiness', fontsize=11)
ax.set_title('Top 50 Candidates: Danceability vs Speechiness\n'
             'Red dashed = SB Halftime Benchmark', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('sb_lxi_audio_scatter.png', dpi=150)
plt.close()
print(f"  Saved: sb_lxi_audio_scatter.png")

# Summary text
lines = [
    "SB LXI Halftime Prediction -- Scoring System Summary",
    "=" * 60, "",
    f"Host: {SB_CITY}, {SB_STATE} | SoFi Stadium | Feb 14, {PREDICT_YEAR}",
    f"Candidates evaluated: {len(df_ranked)}", "",
    "SCORING WEIGHTS (base 100pts):",
] + [f"  {k:<28} {v} pts" for k, v in WEIGHTS.items()] + [
    f"  {'guest_alumni_bonus':<28} +{GUEST_BONUS} pts (additive)",
    "", "AUDIO BENCHMARK (recalibrated from headliner data):",
    f"  Danceability : {SB_AUDIO_BENCHMARK['avg_danceability']:.4f}",
    f"  Speechiness  : {SB_AUDIO_BENCHMARK['avg_speechiness']:.4f}",
    f"  Derived from : {', '.join(a.title() for a in BENCHMARK_ARTISTS)}",
    f"  Missing data : {', '.join(a.title() for a in BENCHMARK_MISSING)} (no audio in dataset)",
    "", "EXCLUDED (past headliners):",
    "  " + ', '.join(sorted(h.title() for h in past_headliners)),
    "", "GUEST ALUMNI (bonus applied):",
    "  " + ', '.join(sorted(g.title() for g in past_guests)),
    f"", f"RECENCY PENALTY: -{int(RECENCY_PENALTY*100)}% for artists "
    f"inactive >{RECENCY_WINDOW} yrs ({penalized_count} artists)",
    "", "TOP 15 CANDIDATES:",
] + [
    f"  {i:>2}. {str(r['artist_name']).title():<22} "
    f"Score:{r['total_score']:>6.2f} | "
    f"{'Legacy' if r['is_legacy_act']==1 else 'Active'} | "
    f"{'Recent' if r['is_recently_active']==1 else 'Inactive*'} | "
    f"{r['home_ground_tier'].title()}"
    f"{' | Guest Alumni' if r['comp_guest_bonus']>0 else ''}"
    for i, r in df_ranked.head(15).iterrows()
] + [
    "", "LIMITATIONS:",
    "  - Spotify scores are current snapshots (documented leakage)",
    "  - Audio features missing ~38% of artists (assigned median score)",
    "  - SB LVI 2022 also in LA -- Roc Nation may avoid west coast hip-hop repeat",
    "  - Model cannot capture Roc Nation internal criteria or business relationships",
    "  - Guest bonus is fixed at +5pts; actual predictive weight is uncertain",
]

summary_text = '\n'.join(lines)
print('\n' + summary_text)
with open('sb_lxi_summary.txt', 'w', encoding='ascii') as f:
    f.write(summary_text)
print(f"\n  Saved: sb_lxi_summary.txt")