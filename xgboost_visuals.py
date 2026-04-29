"""Visualizations for the Super Bowl headliner XGBoost model. (Generated to create visuals)

    1. Load labels and build artist-level features (via `xgboost_model`).
    2. Train XGBoost on past headliners vs everyone else.
    3. Recompute the blended "superbowl_probability" in pieces so we can plot how much
       came from the tree vs popularity vs Billboard legacy (see `score_candidates_with_components`).
    4. Run the year-by-year backtest once and plot hit counts per year.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Reuse all model utilities so plots stay in sync with `xgboost_model.py`.
import xgboost_model as sb_model

# All PNGs go here (created automatically).
OUTPUT_DIR = Path("xgboost_visuals")

# Default blend matches a strong backtest setting without waiting for the full tuning grid.
# Pass `--tune` to replace this with whatever `tune_model()` selects today.
DEFAULT_CONFIG = {
    "xgb_weight": 0.10,
    "popularity_weight": 0.75,
    "legacy_weight": 0.15,
    "weak_legacy_penalty": 0.90,
    "xgb_params": {
        "max_depth": 3,
        "n_estimators": 250,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
}


def score_candidates_with_components(model, candidates, config):
    """Compute the same blended score as `add_probabilities`, but keep each term separate.

    `xgboost_model.add_probabilities` does:
        final = (w_xgb * p_xgb + w_pop * pop + w_leg * leg) * weak_legacy_penalty

    For charts we store the three weighted parts *after* multiplying by the penalty, so
    summing xgb_component + popularity_component + legacy_component equals the final
    probability before clipping. That makes stacked bars read as "where the score came from"
    per artist.
    """
    candidates = candidates.copy()
    xgb_probability = model.predict_proba(sb_model.feature_matrix(candidates))[:, 1]

    # Same weak-legacy rule as xgboost_model: shrink artists with almost no top-10 history.
    penalty = config.get("weak_legacy_penalty", 0.85)
    weak_legacy_penalty = (
        (candidates["billboard_artist_years"] < 4)
        & (candidates["billboard_top10_song_count"] < 1)
        & (candidates["billboard_top10_album_count"] < 1)
    ).map({True: penalty, False: 1.0})

    candidates["xgb_component"] = (
        config["xgb_weight"] * xgb_probability * weak_legacy_penalty
    )
    candidates["popularity_component"] = (
        config["popularity_weight"]
        * candidates["popularity_signal"]
        * weak_legacy_penalty
    )
    candidates["legacy_component"] = (
        config["legacy_weight"] * candidates["legacy_signal"] * weak_legacy_penalty
    )
    candidates["superbowl_probability"] = (
        candidates["xgb_component"]
        + candidates["popularity_component"]
        + candidates["legacy_component"]
    ).clip(0, 1)

    return candidates.sort_values("superbowl_probability", ascending=False)


def build_final_outputs(config):
    """Fit the production-style model and rank non-headliner candidates.

    - Trains on all artists in `final_training_dataset.csv` with positives = known headliners.
    - Drops rows whose `artist_key` is already a headliner in the Super Bowl CSV.
    - Returns the fitted model (for feature importance), the sorted ranking DataFrame, and
      how many positive labels were used (for the console summary).
    """
    headliners = sb_model.load_superbowl_headliners()
    artist_df = sb_model.build_artist_profiles()
    model, positive_count = sb_model.train_model(
        artist_df,
        headliners,
        config.get("xgb_params"),
    )
    candidates = artist_df[~artist_df["artist_key"].isin(headliners)].copy()
    ranking = score_candidates_with_components(model, candidates, config)
    return model, ranking, positive_count


def plot_top_candidates(ranking, output_dir, top_n=20):
    """Horizontal bar chart of the highest blended scores (saved as top_candidates.png)."""
    # Sort ascending so the best artist appears at the bottom of the bar chart (readable).
    top = ranking.head(top_n).sort_values("superbowl_probability")

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top,
        x="superbowl_probability",
        y="artist_name",
        color="#4c72b0",
    )
    plt.title(f"Top {top_n} Super Bowl Halftime Candidate Scores")
    plt.xlabel("Blended Super Bowl Probability")
    plt.ylabel("Artist")
    plt.xlim(0, min(1.0, top["superbowl_probability"].max() * 1.1))
    plt.tight_layout()
    plt.savefig(output_dir / "top_candidates.png", dpi=200)
    plt.close()


def plot_feature_importance(model, output_dir, top_n=15):
    """Which columns in FEATURES drove the most XGBoost splits (saved as feature_importance.png)."""
    importance = pd.DataFrame({
        "feature": sb_model.FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=importance.sort_values("importance"),
        x="importance",
        y="feature",
        color="#55a868",
    )
    plt.title(f"Top {top_n} XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200)
    plt.close()


def plot_score_breakdown(ranking, output_dir, top_n=12):
    """Stacked horizontal bars: XGBoost vs popularity vs legacy slice of each score.

    Uses matplotlib barh (not seaborn) because we need explicit left offsets for stacking.
    Output: score_breakdown.png
    """
    top = ranking.head(top_n).copy()
    top = top.sort_values("superbowl_probability")

    plt.figure(figsize=(10, 7))
    plt.barh(top["artist_name"], top["xgb_component"], label="XGBoost")
    plt.barh(
        top["artist_name"],
        top["popularity_component"],
        left=top["xgb_component"],
        label="Popularity",
    )
    plt.barh(
        top["artist_name"],
        top["legacy_component"],
        left=top["xgb_component"] + top["popularity_component"],
        label="Billboard Legacy",
    )
    plt.title(f"Score Breakdown for Top {top_n} Candidates")
    plt.xlabel("Contribution to Final Score")
    plt.ylabel("Artist")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_breakdown.png", dpi=200)
    plt.close()


def plot_backtest_results(results, output_dir):
    """One line chart: per-year hit counts for top 5, 10, and 20 (backtest_hits.png).

    `results` comes from `run_year_by_year_backtest`. We melt to long form so seaborn can
    draw one line per metric (top5, top10, top20).
    """
    if results.empty:
        return

    plt.figure(figsize=(10, 5))
    hit_cols = ["top5", "top10", "top20"]
    hit_data = results.melt(
        id_vars="year",
        value_vars=hit_cols,
        var_name="metric",
        value_name="hits",
    )
    sns.lineplot(data=hit_data, x="year", y="hits", hue="metric", marker="o")
    plt.title("Year-by-Year Backtest Hits")
    plt.xlabel("Super Bowl Year")
    plt.ylabel("Number of Headliners Found")
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_hits.png", dpi=200)
    plt.close()


def main():
    """Parse CLI, train or tune, write all PNGs, print output directory listing."""
    parser = argparse.ArgumentParser(
        description="Create visualizations for the Super Bowl XGBoost model."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run the full tuning loop before creating plots.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(exist_ok=True)

    config = sb_model.tune_model() if args.tune else DEFAULT_CONFIG
    model, ranking, positive_count = build_final_outputs(config)
    # Quiet backtest: we only need the DataFrame for the plot, not per-year prints.
    backtest_results = sb_model.run_year_by_year_backtest(config, verbose=False)

    plot_top_candidates(ranking, OUTPUT_DIR)
    plot_feature_importance(model, OUTPUT_DIR)
    plot_score_breakdown(ranking, OUTPUT_DIR)
    plot_backtest_results(backtest_results, OUTPUT_DIR)

    print(f"Saved XGBoost visualizations to: {OUTPUT_DIR.resolve()}")
    print(f"Training positives used in final model: {positive_count}")
    print("Files created:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
