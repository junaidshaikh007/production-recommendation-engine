"""Prepare explicit rating events for implicit top-K recommendation tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recommender.config import PROCESSED_DATA_DIR
from recommender.data.preprocess import INTERACTIONS_FILE

LABELED_INTERACTIONS_FILE = PROCESSED_DATA_DIR / "interactions_labeled.parquet"
SUMMARY_FILE = PROCESSED_DATA_DIR / "interaction_label_summary.json"


@dataclass(frozen=True)
class InteractionPolicy:
    """Rules converting explicit ratings into a binary relevance label."""

    positive_rating_threshold: float = 4.0

    def __post_init__(self) -> None:
        """Prevent an invalid rating threshold from silently changing evaluation."""
        if not 1.0 <= self.positive_rating_threshold <= 5.0:
            raise ValueError("positive_rating_threshold must be between 1.0 and 5.0.")


def label_interactions(
    interactions: pd.DataFrame, policy: InteractionPolicy = InteractionPolicy()
) -> pd.DataFrame:
    """Add relevance labels derived only from each observed rating event."""
    required_columns = {"user_id", "item_id", "rating", "timestamp_ms", "event_time"}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Interactions cannot be labeled; missing columns: {missing}.")

    labeled = interactions.copy()
    labeled["is_positive"] = (labeled["rating"] >= policy.positive_rating_threshold).astype("int8")
    labeled["event_date"] = labeled["event_time"].dt.date
    return labeled.sort_values(["user_id", "timestamp_ms"], kind="stable").reset_index(drop=True)


def summarize_labels(labeled_interactions: pd.DataFrame) -> dict[str, int | float]:
    """Return metrics describing the positive-feedback transformation."""
    positives = int(labeled_interactions["is_positive"].sum())
    total = len(labeled_interactions)
    return {
        "interactions": total,
        "positive_interactions": positives,
        "negative_or_neutral_interactions": total - positives,
        "positive_interaction_rate": round(positives / total, 6) if total else 0.0,
    }


def main() -> None:
    """Label local processed interactions and save them as a Parquet dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTERACTIONS_FILE)
    parser.add_argument("--output", type=Path, default=LABELED_INTERACTIONS_FILE)
    parser.add_argument(
        "--threshold", type=float, default=InteractionPolicy().positive_rating_threshold
    )
    arguments = parser.parse_args()

    policy = InteractionPolicy(arguments.threshold)
    labeled = label_interactions(pd.read_parquet(arguments.input), policy)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(arguments.output, index=False)
    summary = summarize_labels(labeled)
    (arguments.output.parent / SUMMARY_FILE.name).write_text(
        json.dumps({"policy": policy.__dict__, **summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
