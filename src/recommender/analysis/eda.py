"""Generate reproducible exploratory analysis for the processed recommendation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from recommender.config import PROCESSED_DATA_DIR, PROJECT_ROOT

INTERACTIONS_FILE = PROCESSED_DATA_DIR / "interactions.parquet"
ITEMS_FILE = PROCESSED_DATA_DIR / "items.parquet"
DEFAULT_REPORT_FILE = PROJECT_ROOT / "docs" / "eda-report.md"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "docs" / "assets"


def calculate_summary(interactions: pd.DataFrame, items: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the key descriptive statistics for the recommendation dataset."""
    user_count = interactions["user_id"].nunique()
    item_count = interactions["item_id"].nunique()
    interaction_count = len(interactions)
    positive_interactions = int((interactions["rating"] >= 4.0).sum())

    return {
        "interactions": interaction_count,
        "users": user_count,
        "items": item_count,
        "metadata_items": len(items),
        "mean_rating": round(float(interactions["rating"].mean()), 3),
        "positive_interaction_rate": round(positive_interactions / interaction_count, 4),
        "avg_interactions_per_user": round(interaction_count / user_count, 3),
        "avg_interactions_per_item": round(interaction_count / item_count, 3),
        "matrix_density": interaction_count / (user_count * item_count),
        "sparsity": 1 - interaction_count / (user_count * item_count),
    }


def create_figures(interactions: pd.DataFrame, figures_dir: Path) -> None:
    """Save rating and activity-distribution charts for the project documentation."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    ratings = interactions["rating"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(ratings.index.astype(str), ratings.values, color="#2563eb")
    axis.set(title="Rating distribution", xlabel="Rating", ylabel="Interactions")
    figure.tight_layout()
    figure.savefig(figures_dir / "rating-distribution.png", dpi=160)
    plt.close(figure)

    user_activity = interactions.groupby("user_id").size()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(user_activity.clip(upper=10), bins=range(1, 12), color="#7c3aed", align="left")
    axis.set(
        title="Interactions per user (values above 10 grouped at 10)",
        xlabel="Interactions per user",
        ylabel="Users",
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "user-activity-distribution.png", dpi=160)
    plt.close(figure)


def create_report(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    report_path: Path,
) -> dict[str, float | int]:
    """Write a concise Markdown EDA report and return its statistics."""
    summary = calculate_summary(interactions, items)
    rating_counts = interactions["rating"].value_counts().sort_index()
    active_start = interactions["event_time"].min().date()
    active_end = interactions["event_time"].max().date()
    top_items = (
        interactions.groupby("item_id")
        .size()
        .rename("interactions")
        .reset_index()
        .merge(items[["item_id", "title"]], on="item_id", how="left")
        .nlargest(5, "interactions")
    )

    rating_rows = "\n".join(
        f"| {rating:.0f} | {count:,} |" for rating, count in rating_counts.items()
    )
    top_item_rows = "\n".join(
        f"| {row.item_id} | {row.title or 'Untitled product'} | {row.interactions:,} |"
        for row in top_items.itertuples(index=False)
    )
    report = f"""# Day 1 exploratory data analysis

## Dataset profile

| Measure | Value |
| --- | ---: |
| Interactions | {summary['interactions']:,} |
| Users | {summary['users']:,} |
| Items | {summary['items']:,} |
| Mean rating | {summary['mean_rating']:.3f} |
| Positive interactions (rating ≥ 4) | {summary['positive_interaction_rate']:.2%} |
| Average interactions per user | {summary['avg_interactions_per_user']:.3f} |
| Average interactions per item | {summary['avg_interactions_per_item']:.3f} |
| User-item matrix density | {summary['matrix_density']:.8%} |
| Matrix sparsity | {summary['sparsity']:.6%} |
| Interaction period | {active_start} to {active_end} |

## Rating distribution

| Rating | Interactions |
| ---: | ---: |
{rating_rows}

![Rating distribution](assets/rating-distribution.png)

## User activity

![User activity distribution](assets/user-activity-distribution.png)

The interaction matrix is extremely sparse: the average user has only about one interaction.
This is a realistic cold-start and sparsity challenge. Popularity and content-based
approaches will therefore be important baselines, while collaborative models will be
evaluated only on users with sufficient history.

## Most-interacted products

| Item ID | Product title | Interactions |
| --- | --- | ---: |
{top_item_rows}

## Decisions carried into Day 2

1. Preserve chronological event times and use time-aware splits to prevent future leakage.
2. Keep explicit ratings, then define positive implicit feedback as `rating >= 4`
   for top-K evaluation experiments.
3. Use `item_text` (title, description, features, store) for content features
   because the raw category field is empty.
4. Report popularity and cold-start performance separately from personalized-model performance.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    """Create the Day 1 EDA report and figures from processed Parquet files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=INTERACTIONS_FILE)
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    arguments = parser.parse_args()

    interactions = pd.read_parquet(arguments.interactions)
    items = pd.read_parquet(arguments.items)
    create_figures(interactions, arguments.figures_dir)
    summary = create_report(interactions, items, arguments.report)
    print(f"EDA report saved to: {arguments.report}")
    print(f"Interactions: {summary['interactions']:,}; sparsity: {summary['sparsity']:.6%}")


if __name__ == "__main__":
    main()
