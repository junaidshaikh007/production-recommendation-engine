"""Inspect raw Amazon review and metadata files before preprocessing."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recommender.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

REVIEWS_FILE = RAW_DATA_DIR / "All_Beauty.jsonl.gz"
METADATA_FILE = RAW_DATA_DIR / "meta_All_Beauty.jsonl.gz"
SUMMARY_FILE = PROCESSED_DATA_DIR / "raw_data_inspection.json"


def read_json_lines(path: Path):
    """Yield JSON records from a UTF-8 gzip JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path.name}:{line_number}") from error


def inspect_reviews(path: Path) -> tuple[dict[str, Any], set[str]]:
    """Calculate interaction-quality statistics and return reviewed item IDs."""
    required_fields = ("user_id", "parent_asin", "rating", "timestamp")
    missing_fields: Counter[str] = Counter()
    rating_distribution: Counter[str] = Counter()
    user_ids: set[str] = set()
    item_ids: set[str] = set()
    event_ids: set[tuple[str | None, str | None, int | None]] = set()
    duplicate_events = 0
    timestamps: list[int] = []
    records = 0

    for record in read_json_lines(path):
        records += 1
        for field in required_fields:
            if record.get(field) in (None, ""):
                missing_fields[field] += 1

        user_id = record.get("user_id")
        item_id = record.get("parent_asin")
        timestamp = record.get("timestamp")
        if user_id:
            user_ids.add(user_id)
        if item_id:
            item_ids.add(item_id)
        if isinstance(timestamp, int):
            timestamps.append(timestamp)

        event_id = (user_id, item_id, timestamp)
        if event_id in event_ids:
            duplicate_events += 1
        event_ids.add(event_id)
        if record.get("rating") is not None:
            rating_distribution[str(record["rating"])] += 1

    return (
        {
            "records": records,
            "unique_users": len(user_ids),
            "unique_reviewed_items": len(item_ids),
            "exact_duplicate_events": duplicate_events,
            "missing_required_fields": dict(sorted(missing_fields.items())),
            "rating_distribution": dict(sorted(rating_distribution.items())),
            "first_interaction_utc": timestamp_to_iso(min(timestamps)) if timestamps else None,
            "last_interaction_utc": timestamp_to_iso(max(timestamps)) if timestamps else None,
        },
        item_ids,
    )


def inspect_metadata(path: Path) -> tuple[dict[str, Any], set[str]]:
    """Calculate metadata completeness statistics and return item IDs."""
    content_fields = ("title", "description", "features", "categories")
    missing_content: Counter[str] = Counter()
    item_ids: set[str] = set()
    records = 0

    for record in read_json_lines(path):
        records += 1
        item_id = record.get("parent_asin")
        if item_id:
            item_ids.add(item_id)
        for field in content_fields:
            if not record.get(field):
                missing_content[field] += 1

    return (
        {
            "records": records,
            "unique_metadata_items": len(item_ids),
            "missing_content_fields": dict(sorted(missing_content.items())),
        },
        item_ids,
    )


def timestamp_to_iso(timestamp_ms: int) -> str:
    """Convert a Unix timestamp in milliseconds to an ISO-8601 UTC string."""
    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC).isoformat()


def build_summary(reviews_path: Path, metadata_path: Path) -> dict[str, Any]:
    """Build a serializable inspection summary for the two raw dataset files."""
    reviews, reviewed_items = inspect_reviews(reviews_path)
    metadata, metadata_items = inspect_metadata(metadata_path)
    matched_items = reviewed_items & metadata_items

    return {
        "source_files": {
            "reviews": reviews_path.name,
            "metadata": metadata_path.name,
        },
        "reviews": reviews,
        "metadata": metadata,
        "join_coverage": {
            "reviewed_items_with_metadata": len(matched_items),
            "reviewed_item_metadata_coverage": round(
                len(matched_items) / len(reviewed_items), 6
            )
            if reviewed_items
            else 0.0,
        },
    }


def main() -> None:
    """Inspect the project raw files and save a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=REVIEWS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--output", type=Path, default=SUMMARY_FILE)
    arguments = parser.parse_args()

    summary = build_summary(arguments.reviews, arguments.metadata)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Inspection summary saved to: {arguments.output}")


if __name__ == "__main__":
    main()

