import gzip
import json
from pathlib import Path

from recommender.data.inspection import build_summary, timestamp_to_iso


def write_jsonl_gzip(path: Path, records: list[dict[str, object]]) -> None:
    """Create a small gzip JSONL fixture."""
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def test_build_summary_reports_quality_and_join_coverage(tmp_path: Path) -> None:
    reviews_path = tmp_path / "reviews.jsonl.gz"
    metadata_path = tmp_path / "metadata.jsonl.gz"
    review = {
        "user_id": "user-1",
        "parent_asin": "item-1",
        "rating": 5.0,
        "timestamp": 1_700_000_000_000,
    }
    write_jsonl_gzip(reviews_path, [review, review, {**review, "parent_asin": "item-2"}])
    write_jsonl_gzip(
        metadata_path,
        [
            {
                "parent_asin": "item-1",
                "title": "Product",
                "description": [],
                "features": [],
                "categories": [],
            },
            {
                "parent_asin": "item-3",
                "title": "",
                "description": [],
                "features": [],
                "categories": [],
            },
        ],
    )

    summary = build_summary(reviews_path, metadata_path)

    assert summary["reviews"]["records"] == 3
    assert summary["reviews"]["exact_duplicate_events"] == 1
    assert summary["reviews"]["unique_users"] == 1
    assert summary["join_coverage"]["reviewed_items_with_metadata"] == 1
    assert summary["join_coverage"]["reviewed_item_metadata_coverage"] == 0.5


def test_timestamp_to_iso_uses_utc() -> None:
    assert timestamp_to_iso(0) == "1970-01-01T00:00:00+00:00"
