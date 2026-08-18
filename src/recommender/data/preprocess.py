"""Create clean interaction and item datasets from the raw Amazon files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from recommender.config import PROCESSED_DATA_DIR
from recommender.data.inspection import METADATA_FILE, REVIEWS_FILE, read_json_lines

INTERACTIONS_FILE = PROCESSED_DATA_DIR / "interactions.parquet"
ITEMS_FILE = PROCESSED_DATA_DIR / "items.parquet"
SUMMARY_FILE = PROCESSED_DATA_DIR / "preprocessing_summary.json"


def clean_interactions(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Select, validate, deduplicate, and order user-item interactions."""
    rows = [
        {
            "user_id": record.get("user_id"),
            "item_id": record.get("parent_asin"),
            "rating": record.get("rating"),
            "timestamp_ms": record.get("timestamp"),
            "verified_purchase": record.get("verified_purchase"),
            "helpful_votes": record.get("helpful_vote", record.get("helpful_votes", 0)),
        }
        for record in records
    ]
    interactions = pd.DataFrame(rows)
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating", "timestamp_ms"])
    interactions = interactions.astype(
        {
            "user_id": "string",
            "item_id": "string",
            "rating": "float32",
            "timestamp_ms": "int64",
            "verified_purchase": "boolean",
            "helpful_votes": "int32",
        }
    )
    interactions = interactions.drop_duplicates(
        subset=["user_id", "item_id", "timestamp_ms"], keep="first"
    )
    interactions["event_time"] = pd.to_datetime(
        interactions["timestamp_ms"], unit="ms", utc=True
    )
    return interactions.sort_values(["user_id", "timestamp_ms"], kind="stable").reset_index(
        drop=True
    )


def join_text(value: Any) -> str:
    """Convert a string or list of strings into whitespace-normalized text."""
    if isinstance(value, list):
        value = " ".join(str(part) for part in value if part)
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def clean_items(records: Iterable[dict[str, Any]], reviewed_item_ids: set[str]) -> pd.DataFrame:
    """Keep relevant metadata and derive a text feature for each reviewed item."""
    rows = []
    for record in records:
        item_id = record.get("parent_asin")
        if item_id not in reviewed_item_ids:
            continue
        title = join_text(record.get("title"))
        description = join_text(record.get("description"))
        features = join_text(record.get("features"))
        store = join_text(record.get("store"))
        item_text = " ".join(part for part in [title, description, features, store] if part)
        rows.append(
            {
                "item_id": item_id,
                "title": title,
                "description": description,
                "features": features,
                "store": store,
                "price": record.get("price"),
                "average_rating": record.get("average_rating"),
                "rating_number": record.get("rating_number"),
                "item_text": item_text,
            }
        )

    items = pd.DataFrame(rows).drop_duplicates(subset=["item_id"], keep="first")
    items = items.astype(
        {
            "item_id": "string",
            "title": "string",
            "description": "string",
            "features": "string",
            "store": "string",
            "price": "float32",
            "average_rating": "float32",
            "rating_number": "Int32",
            "item_text": "string",
        }
    )
    return items.sort_values("item_id", kind="stable").reset_index(drop=True)


def preprocess(reviews_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build clean interactions and items DataFrames from raw source files."""
    interactions = clean_interactions(read_json_lines(reviews_path))
    items = clean_items(read_json_lines(metadata_path), set(interactions["item_id"]))
    return interactions, items


def save_processed_data(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, int]:
    """Persist processed data as Parquet and return row counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(output_dir / INTERACTIONS_FILE.name, index=False)
    items.to_parquet(output_dir / ITEMS_FILE.name, index=False)
    summary = {
        "interactions": len(interactions),
        "items": len(items),
        "items_without_text": int((items["item_text"].str.len() == 0).sum()),
    }
    (output_dir / SUMMARY_FILE.name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Run preprocessing against the project's downloaded raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=REVIEWS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    arguments = parser.parse_args()

    interactions, items = preprocess(arguments.reviews, arguments.metadata)
    summary = save_processed_data(interactions, items, arguments.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
