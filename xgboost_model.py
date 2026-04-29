"""Artist-level Super Bowl halftime headliner ranking.

  1. Load track-level data and aggregate to one row per artist (Spotify + Billboard).
  2. Train XGBoost on past headliners as the positive class (few positives, heavy imbalance).
  3. Combine the model probability with explicit popularity and legacy rank signals in a
     weighted blend, plus an optional penalty for artists with little Billboard history.
  4. Tune blend/penalty/tree depth on a year-by-year backtest (no future data leakage).
  5. Print a final ranked list of artists who are not already listed as headliners.

The blend exists because backtests showed it improves hit@5 and hit@10 versus using only
raw XGBoost probabilities: the model sees the same signals as features, but the blend
directly forces mainstream appeal and long-term chart success into the final score.
"""

import pandas as pd
import numpy as np
import xgboost as xgb

# Input files used by the model pipeline.
DATA_FILE = "final_training_dataset.csv"
SUPERBOWL_FILE = "data/superbowl_halftime_shows/superbowl_halftime_performers.csv"


def normalize_artist(value):
    """Normalize artist names so joins and comparisons are consistent."""
    return str(value).strip().lower()


def split_people(value):
    """Split comma-separated performer cells into normalized artist names."""
    if pd.isna(value):
        return []
    return [
        normalize_artist(name)
        for name in str(value).replace("&", ",").split(",")
        if name.strip()
    ]


def load_superbowl_headliners():
    """Return confirmed headliners used as the positive training label.

    Only actual headliners are treated as positives. Guest performers and declined
    artists are intentionally ignored so prior Super Bowl involvement does not
    dominate the ranking.
    """
    sb = pd.read_csv(SUPERBOWL_FILE)

    headliners = set()

    for _, row in sb.iterrows():
        row_headliners = set(split_people(row.get("headliners")))
        headliners.update(row_headliners)

    return headliners


def load_superbowl_roles_by_year():
    """Load confirmed headliners grouped by Super Bowl year for backtesting."""
    sb = pd.read_csv(SUPERBOWL_FILE)
    roles = {}
    for _, row in sb.iterrows():
        roles[int(row["year"])] = set(split_people(row.get("headliners")))
    return roles


def load_billboard_features(cutoff_year=None):
    """Build artist-level Billboard history features."""
    artists = pd.read_csv("billboard_artists_master.csv")
    songs = pd.read_csv("billboard_songs_master.csv")
    albums = pd.read_csv("billboard_albums_master.csv")

    # Normalize names and numeric fields before aggregating.
    for frame in (artists, songs, albums):
        frame["artist_key"] = frame["artist"].apply(normalize_artist)
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")

    # For historical prediction, only use chart history that existed at the time.
    if cutoff_year is not None:
        artists = artists[artists["year"] <= cutoff_year]
        songs = songs[songs["year"] <= cutoff_year]
        albums = albums[albums["year"] <= cutoff_year]

    # Artist chart performance: longevity and peak rank.
    artist_stats = artists.groupby("artist_key").agg(
        billboard_artist_years=("year", "nunique"),
        billboard_peak_artist_rank=("rank", "min"),
        billboard_top10_artist_years=("rank", lambda s: (s <= 10).sum()),
    )

    # Song chart performance: hit count and peak song success.
    song_stats = songs.groupby("artist_key").agg(
        billboard_song_count=("song", "count"),
        billboard_peak_song_rank=("rank", "min"),
        billboard_top10_song_count=("rank", lambda s: (s <= 10).sum()),
    )

    # Album chart performance: album count and peak album success.
    album_stats = albums.groupby("artist_key").agg(
        billboard_album_count=("album", "count"),
        billboard_peak_album_rank=("rank", "min"),
        billboard_top10_album_count=("rank", lambda s: (s <= 10).sum()),
    )

    features = pd.concat([artist_stats, song_stats, album_stats], axis=1).reset_index()
    # Convert rank values into scores where higher is better.
    for rank_col in [
        "billboard_peak_artist_rank",
        "billboard_peak_song_rank",
        "billboard_peak_album_rank",
    ]:
        score_col = rank_col.replace("rank", "score")
        features[score_col] = 101 - pd.to_numeric(features[rank_col], errors="coerce")
        features[score_col] = features[score_col].fillna(0).clip(lower=0)

    return features


def build_artist_profiles(cutoff_year=None):
    """Collapse track rows into one model row per artist.

    Each artist profile combines Spotify popularity, release activity, audio
    averages, and Billboard legacy features. cutoff_year keeps the profile
    historically accurate during backtests.
    """
    df = pd.read_csv(DATA_FILE)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["artist_key"] = df["artist_name"].apply(normalize_artist)
    # Historical backtests should not use tracks released after the prediction year.
    if cutoff_year is not None:
        df = df[df["release_year"] <= cutoff_year].copy()

    # Make model inputs numeric so XGBoost receives usable values.
    numeric_cols = [
        "artist_popularity",
        "track_popularity",
        "artist_followers",
        "danceability",
        "energy",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    current_year = int(cutoff_year or df["release_year"].max())
    recent_cutoff = current_year - 2
    # Recent activity is a proxy for whether the artist is currently visible.
    df["is_recent_track"] = (df["release_year"] >= recent_cutoff).astype(int)

    # Aggregate song-level data into one artist-level row.
    artists = df.groupby("artist_key").agg(
        artist_name=("artist_name", "first"),
        artist_popularity=("artist_popularity", "max"),
        artist_followers=("artist_followers", "max"),
        track_popularity_max=("track_popularity", "max"),
        track_popularity_mean=("track_popularity", "mean"),
        track_count=("track_name", "count"),
        recent_track_count=("is_recent_track", "sum"),
        latest_release_year=("release_year", "max"),
        earliest_release_year=("release_year", "min"),
        danceability_mean=("danceability", "mean"),
        energy_mean=("energy", "mean"),
    ).reset_index()

    artists["years_since_latest_release"] = (
        current_year - artists["latest_release_year"]
    ).clip(lower=0)
    artists["active_year_span"] = (
        artists["latest_release_year"] - artists["earliest_release_year"] + 1
    ).clip(lower=1)

    artists = artists.merge(load_billboard_features(cutoff_year), on="artist_key", how="left")

    # Missing Billboard counts mean the artist was not present in that chart table.
    count_cols = [
        "billboard_artist_years",
        "billboard_top10_artist_years",
        "billboard_song_count",
        "billboard_top10_song_count",
        "billboard_album_count",
        "billboard_top10_album_count",
    ]
    score_cols = [
        "billboard_peak_artist_score",
        "billboard_peak_song_score",
        "billboard_peak_album_score",
    ]
    artists[count_cols + score_cols] = artists[count_cols + score_cols].fillna(0)
    # Popularity signal: percentile ranks (0–1) mixed into one scalar. All terms point in
    # the same direction (more fame / activity = higher). Weights are a design choice so
    # artist-level Spotify popularity and followers matter most, with a small nudge for
    # having very recent tracks (current relevance without needing raw dates in the blend).
    artists["popularity_signal"] = (
        artists["artist_popularity"].rank(pct=True).fillna(0) * 0.35
        + artists["track_popularity_max"].rank(pct=True).fillna(0) * 0.25
        + artists["track_popularity_mean"].rank(pct=True).fillna(0) * 0.15
        + np.log1p(artists["artist_followers"]).rank(pct=True).fillna(0) * 0.20
        + artists["recent_track_count"].rank(pct=True).fillna(0) * 0.05
    )
    # Legacy signal: same idea but from Billboard aggregates (years on chart, top-10 counts).
    # It rewards artists who look like established hit-makers, not one-off viral spikes.
    artists["legacy_signal"] = (
        artists["billboard_artist_years"].rank(pct=True).fillna(0) * 0.30
        + artists["billboard_top10_artist_years"].rank(pct=True).fillna(0) * 0.20
        + artists["billboard_song_count"].rank(pct=True).fillna(0) * 0.15
        + artists["billboard_top10_song_count"].rank(pct=True).fillna(0) * 0.15
        + artists["billboard_album_count"].rank(pct=True).fillna(0) * 0.10
        + artists["billboard_top10_album_count"].rank(pct=True).fillna(0) * 0.10
    )

    return artists


# Features passed into XGBoost. Many of these also feed the hand-crafted popularity_signal
# and legacy_signal used in the post-model blend, so the tree sees raw structure and the
# blend re-emphasizes interpretable "star power" for ranking.
FEATURES = [
    "artist_popularity",
    "popularity_signal",
    "artist_followers",
    "track_popularity_max",
    "track_popularity_mean",
    "track_count",
    "recent_track_count",
    "latest_release_year",
    "years_since_latest_release",
    "active_year_span",
    "billboard_artist_years",
    "billboard_peak_artist_score",
    "billboard_top10_artist_years",
    "billboard_song_count",
    "billboard_peak_song_score",
    "billboard_top10_song_count",
    "billboard_album_count",
    "billboard_peak_album_score",
    "billboard_top10_album_count",
    "danceability_mean",
    "energy_mean",
]

# Monotonic constraints tell XGBoost that higher popularity/legacy values should
# not reduce predicted Super Bowl likelihood. Audio averages are unconstrained.
CONSTRAINTS = (
    1,   # artist_popularity
    1,   # popularity_signal
    1,   # artist_followers
    1,   # track_popularity_max
    1,   # track_popularity_mean
    1,   # track_count
    1,   # recent_track_count
    1,   # latest_release_year - newer catalog edge allowed
    -1,  # years_since_latest_release - staler is worse (recent releases help)
    1,   # active_year_span
    1,   # billboard_artist_years
    1,   # billboard_peak_artist_score
    1,   # billboard_top10_artist_years
    1,   # billboard_song_count
    1,   # billboard_peak_song_score
    1,   # billboard_top10_song_count
    1,   # billboard_album_count
    1,   # billboard_peak_album_score
    1,   # billboard_top10_album_count
    0,   # danceability_mean
    0,   # energy_mean
)


def feature_matrix(df):
    """Prepare the feature matrix used by XGBoost."""
    X = df[FEATURES].copy()
    # Missing recency means no known release; treat it as very stale.
    X["years_since_latest_release"] = X["years_since_latest_release"].fillna(99)
    return X.fillna(0)


def train_model(artist_df, positive_artists, xgb_params=None):
    """Train XGBoost to identify artists similar to confirmed headliners."""
    artist_df = artist_df.copy()
    artist_df["target"] = artist_df["artist_key"].isin(positive_artists).astype(int)
    X = feature_matrix(artist_df)
    y = artist_df["target"]
    # Upweight the rare positive class (past headliners) so the model does not collapse to "always 0".
    scale_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    xgb_params = xgb_params or {}
    model = xgb.XGBClassifier(
        monotone_constraints=CONSTRAINTS,
        scale_pos_weight=scale_weight,
        # n_estimators defaults to 350 here; tune_model() uses 250 per config to run the grid faster.
        learning_rate=xgb_params.get("learning_rate", 0.03),
        n_estimators=xgb_params.get("n_estimators", 350),
        max_depth=xgb_params.get("max_depth", 3),
        subsample=xgb_params.get("subsample", 0.9),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.9),
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    return model, int(y.sum())


def add_probabilities(model, candidates, config):
    """Turn XGBoost output and config-driven weights into a single ranking score.

    Why not only use predict_proba? With very few positive labels, the raw probability can
    underweight household names that the public would expect near the top. The blend adds
    direct control: popularity_signal and legacy_signal are percentile-based scalars, so
    each term is on a comparable scale before weighting.

    Formula (weights should sum to 1 for easy reading as "percent from each source"):
        blended = xgb_w * p_xgb + pop_w * popularity_signal + leg_w * legacy_signal
    Then multiply by weak_legacy_penalty when the artist looks "hot but thin" on charts.

    weak_legacy_penalty applies when all are true:
      - fewer than 4 distinct years on Billboard artist charts, and
      - no top-10 song weeks, and no top-10 album weeks.
    Those thresholds are a heuristic for "mostly viral / regional / new" vs sustained US
    mainstream presence typical of past headliners. penalty (0.80 vs 0.90 in tuning) scales
    how hard we ding that bucket: lower penalty = stronger shrink on the final score.
    config.get("weak_legacy_penalty", 0.85) is only used if a config omits the key.
    """
    candidates = candidates.copy()
    xgb_probability = model.predict_proba(feature_matrix(candidates))[:, 1]

    penalty = config.get("weak_legacy_penalty", 0.85)
    weak_legacy_penalty = (
        (candidates["billboard_artist_years"] < 4)
        & (candidates["billboard_top10_song_count"] < 1)
        & (candidates["billboard_top10_album_count"] < 1)
    ).map({True: penalty, False: 1.0})
    candidates["superbowl_probability"] = (
        config["xgb_weight"] * xgb_probability
        + config["popularity_weight"] * candidates["popularity_signal"]
        + config["legacy_weight"] * candidates["legacy_signal"]
    ).mul(weak_legacy_penalty).clip(0, 1)
    return candidates.sort_values("superbowl_probability", ascending=False)


def run_year_by_year_backtest(config, verbose=True):
    """Evaluate the model as if predicting each historical Super Bowl year.

    For each year Y (after the first in the file): train labels = headliners from all years
    before Y; features use data only through Y-1; then we rank everyone and check whether
    that years real headliner(s) appear in the top 5, 10, or 20.

    top5/top10/top20 are counts across headliners for that year (some years have multiple
    names). best_rank is the best list position among them (1 = top of ranking).
    """
    roles_by_year = load_superbowl_roles_by_year()
    years = sorted(roles_by_year)
    results = []

    if verbose:
        print("\n--- YEAR-BY-YEAR BACKTEST ---")
    # Skip the earliest year: there is no "prior headliners only" training set before it.
    for year in years[1:]:
        # Train only on headliners from years before the prediction year.
        prior_headliners = set().union(*(roles_by_year[y] for y in years if y < year))
        current_headliners = roles_by_year[year]
        # Build features using only tracks/charts available before this Super Bowl.
        artist_df = build_artist_profiles(cutoff_year=year - 1)

        if not prior_headliners or not current_headliners:
            continue

        model, positive_count = train_model(artist_df, prior_headliners, config.get("xgb_params"))
        ranking = add_probabilities(model, artist_df, config)

        # Compute where the real headliner(s) landed in the ranked list.
        ranks = []
        for artist in current_headliners:
            matches = ranking.index[ranking["artist_key"] == artist]
            if len(matches):
                ranks.append(ranking.index.get_loc(matches[0]) + 1)

        top5 = sum(rank <= 5 for rank in ranks)
        top10 = sum(rank <= 10 for rank in ranks)
        top20 = sum(rank <= 20 for rank in ranks)
        best_rank = min(ranks) if ranks else None

        results.append({
            "year": year,
            "top5": top5,
            "top10": top10,
            "top20": top20,
            "headliners": len(current_headliners),
            "best_rank": best_rank,
        })

        if verbose:
            print(
                f"{year}: positives_train={positive_count}, "
                f"hit@5={top5}/{len(current_headliners)}, "
                f"hit@10={top10}/{len(current_headliners)}, "
                f"hit@20={top20}/{len(current_headliners)}, "
                f"best_rank={best_rank}, "
                f"headliners={', '.join(sorted(current_headliners))}"
            )

    if results:
        result_df = pd.DataFrame(results)
        if verbose:
            print("\nBacktest averages:")
            print(
                result_df[["top5", "top10", "top20", "best_rank"]]
                .mean(numeric_only=True)
                .round(3)
                .to_string()
            )
        return result_df
    return pd.DataFrame()


def make_tuning_configs():
    """Exhaustive small grid for blend weights, weak-legacy penalty, and tree depth.

    Each blend is (xgb_weight, popularity_weight, legacy_weight). They are chosen to sum to
    1.0 so each tuple is a convex mix of three scores in add_probabilities.

    Rationale for the blend grid:
      - Rows bias heavily toward popularity_weight (0.70–0.80) because backtests showed
        that helps hit@5 and hit@10 vs using XGBoost probability alone, matching the idea
        that the NFL picks widely known acts, not only whoever the tree fits best on few
        past positives.
      - xgb_weight stays at or below 0.20 so the tree still steers the ranking (especially
        for subtle combinations the hand-built signals do not capture) without drowning
        out the explicit popularity and legacy terms.
      - legacy_weight stays between 0.10 and 0.20 so chart veterans are not ignored when
        popularity ties are close.

    Penalties [0.80, 0.90]: these are multipliers applied only to the "weak legacy" bucket.
    0.80 is a stronger correction (more reduction of blended score); 0.90 is milder. Tuning
    picks which level generalizes better across held-out years.

    depths [2, 3]: shallow trees with few positives. Depth 2 is more conservative and fights
    overfitting; depth 3 allows slightly richer splits. Both are tried because the best
    depth can depend on the randomness of which artists are positives in training each year.

    Fixed XGBoost knobs inside each config (not swept here): n_estimators=250 keeps tuning
    runs faster than the default 350 in train_model(); learning_rate=0.03 is a standard
    moderate step; subsample/colsample_bytree=0.9 add light stochastic regularization.
    """
    configs = []
    blend_options = [
        # (xgb, popularity, legacy) - each row sums to 1; see module docstring on why popularity is largest.
        (0.05, 0.80, 0.15),
        (0.10, 0.80, 0.10),
        (0.15, 0.75, 0.10),
        (0.20, 0.70, 0.10),
        (0.10, 0.70, 0.20),
    ]
    penalties = [0.80, 0.90]
    depths = [2, 3]

    for xgb_w, pop_w, legacy_w in blend_options:
        for penalty in penalties:
            for depth in depths:
                configs.append({
                    "xgb_weight": xgb_w,
                    "popularity_weight": pop_w,
                    "legacy_weight": legacy_w,
                    "weak_legacy_penalty": penalty,
                    "xgb_params": {
                        "max_depth": depth,
                        "n_estimators": 250,
                        "learning_rate": 0.03,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    },
                })
    return configs


def tune_model():
    """Pick the config that maximizes a proxy for "good ranking" on the backtest.

    Composite score (higher is better):
      score = 5 * mean(hit@5) + 3 * mean(hit@10) + 1 * mean(hit@20) - mean(best_rank)/100

    The coefficients 5, 3, 1 encode priority: getting the true headliner in the top 5 is
    most valuable, then top 10, then top 20. Subtracting best_rank/100 slightly prefers
    configs that place the headliner closer to rank 1 when hits are similar across trials.
    This is a design choice, not a standard ML metric; it matches the project goal of
    headline ranking rather than pure classification accuracy.
    """
    print("\n--- TUNING MODEL CONFIGURATION ---")
    scored = []
    for i, config in enumerate(make_tuning_configs(), 1):
        results = run_year_by_year_backtest(config, verbose=False)
        if results.empty:
            continue
        avg = results[["top5", "top10", "top20", "best_rank"]].mean(numeric_only=True)
        score = (
            avg["top5"] * 5
            + avg["top10"] * 3
            + avg["top20"]
            - (avg["best_rank"] / 100)
        )
        scored.append((score, avg, config))
        # Print each trial so the final configuration is easy to justify.
        print(
            f"{i:02d}: score={score:.3f} "
            f"hit@5={avg['top5']:.3f} hit@10={avg['top10']:.3f} "
            f"hit@20={avg['top20']:.3f} best_rank={avg['best_rank']:.1f} "
            f"blend=({config['xgb_weight']}, {config['popularity_weight']}, {config['legacy_weight']}) "
            f"penalty={config['weak_legacy_penalty']} depth={config['xgb_params']['max_depth']}"
        )

    best_score, best_avg, best_config = max(scored, key=lambda item: item[0])
    print("\nBest config:")
    print(
        f"score={best_score:.3f}, "
        f"hit@5={best_avg['top5']:.3f}, hit@10={best_avg['top10']:.3f}, "
        f"hit@20={best_avg['top20']:.3f}, best_rank={best_avg['best_rank']:.1f}"
    )
    print(
        f"blend=({best_config['xgb_weight']}, {best_config['popularity_weight']}, {best_config['legacy_weight']}), "
        f"penalty={best_config['weak_legacy_penalty']}, depth={best_config['xgb_params']['max_depth']}"
    )
    return best_config


def main():
    """Run tuning, historical backtest, and final candidate ranking.

    Order matters: tune on the backtest first, print the same backtest with the winner for
    inspection, then retrain on all artists in DATA_FILE with full labels and output the
    forward-looking list for people not already in superbowl_halftime_performers.csv.
    """
    headliners = load_superbowl_headliners()
    artist_df = build_artist_profiles()

    # Pick model/blend settings based on historical ranking performance.
    best_config = tune_model()
    run_year_by_year_backtest(best_config)

    # Train on full timeline: positives are artists who appear as headliners anywhere in SUPERBOWL_FILE.
    # Final ranking excludes those headliners so the printed table reads as "who could be next".
    model, positive_count = train_model(artist_df, headliners, best_config.get("xgb_params"))
    candidates = artist_df[~artist_df["artist_key"].isin(headliners)].copy()
    ranking = add_probabilities(model, candidates, best_config)

    print(
        f"Training artists: {len(artist_df)} | "
        f"positive labels (past headliners in training): {positive_count} | "
        f"ranked candidates: {len(candidates)}"
    )
    print("\n--- SUPER BOWL PERFORMER PROBABILITY RANKING ---")
    print(ranking[["artist_name", "superbowl_probability"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
