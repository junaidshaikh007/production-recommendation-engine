"""Generate leakage-safe item and training-history features for recommenders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.config import PROCESSED_DATA_DIR
from recommender.data.preprocess import ITEMS_FILE
from recommender.data.splitting import DEFAULT_SPLITS_DIR

DEFAULT_FEATURES_DIR = PROCESSED_DATA_DIR / "features"


def generate_item_features(items: pd.DataFrame) -> pd.DataFrame:
    """Derive content and availability features from item metadata only."""
    required_columns = {"item_id", "title", "description", "features", "item_text", "price"}
    missing_columns = required_columns - set(items.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Cannot generate item features; missing columns: {missing}.")

    item_features = pd.DataFrame({"item_id": items["item_id"]})
    for source_column, feature_name in (
        ("title", "title_length"),
        ("description", "description_length"),
        ("features", "feature_text_length"),
        ("item_text", "item_text_length"),
    ):
        item_features[feature_name] = items[source_column].fillna("").str.len().astype("int32")

    item_features["has_item_text"] = (item_features["item_text_length"] > 0).astype("int8")
    item_features["has_price"] = items["price"].notna().astype("int8")
    item_features["price_log1p"] = (
        np.log1p(items["price"].clip(lower=0)).fillna(0).astype("float32")
    )
    return item_features


def generate_training_history_features(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate behavior strictly from the training period for later ranking features."""
    required_columns = {"user_id", "item_id", "is_positive"}
    missing_columns = required_columns - set(train.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Cannot generate history features; missing columns: {missing}.")

    def aggregate(entity_column: str, prefix: str) -> pd.DataFrame:
        grouped = train.groupby(entity_column, as_index=False).agg(
            interactions=("is_positive", "size"),
            positive_interactions=("is_positive", "sum"),
        )
        grouped[f"{prefix}_interaction_count"] = grouped.pop("interactions").astype("int32")
        grouped[f"{prefix}_positive_count"] = grouped.pop("positive_interactions").astype("int32")
        grouped[f"{prefix}_positive_rate"] = (
            grouped[f"{prefix}_positive_count"] / grouped[f"{prefix}_interaction_count"]
        ).astype("float32")
        return grouped

    return aggregate("user_id", "user"), aggregate("item_id", "item")


def save_features(
    item_features: pd.DataFrame,
    user_history_features: pd.DataFrame,
    item_history_features: pd.DataFrame,
    output_dir: Path = DEFAULT_FEATURES_DIR,
) -> dict[str, int]:
    """Persist generated feature tables and return their row counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    item_features.to_parquet(output_dir / "item_features.parquet", index=False)
    user_history_features.to_parquet(output_dir / "user_training_features.parquet", index=False)
    item_history_features.to_parquet(output_dir / "item_training_features.parquet", index=False)
    summary = {
        "item_features": len(item_features),
        "user_training_features": len(user_history_features),
        "item_training_features": len(item_history_features),
    }
    (output_dir / "feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """Generate local model features using items and the chronological train split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS_DIR / "train.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    arguments = parser.parse_args()

    item_features = generate_item_features(pd.read_parquet(arguments.items))
    user_features, item_history_features = generate_training_history_features(
        pd.read_parquet(arguments.train)
    )
    summary = save_features(
        item_features, user_features, item_history_features, arguments.output_dir
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
