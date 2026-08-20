"""Validate processed recommendation datasets before feature engineering."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from recommender.config import PROCESSED_DATA_DIR
from recommender.data.preprocess import INTERACTIONS_FILE, ITEMS_FILE


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding with its severity and affected row count."""

    severity: str
    dataset: str
    check: str
    message: str
    affected_rows: int = 0


@dataclass
class ValidationReport:
    """Collection of validation findings for the processed data contract."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True when no error-severity finding exists."""
        return not any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        severity: str,
        dataset: str,
        check: str,
        message: str,
        affected_rows: int = 0,
    ) -> None:
        """Add a validation finding."""
        self.issues.append(ValidationIssue(severity, dataset, check, message, affected_rows))

    def to_dict(self) -> dict[str, object]:
        """Convert the report into JSON-compatible data."""
        return {
            "is_valid": self.is_valid,
            "errors": sum(issue.severity == "error" for issue in self.issues),
            "warnings": sum(issue.severity == "warning" for issue in self.issues),
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    dataset: str,
    report: ValidationReport,
) -> bool:
    """Report missing columns and return whether all required columns exist."""
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        report.add(
            "error",
            dataset,
            "required_columns",
            f"Missing required columns: {', '.join(missing_columns)}.",
        )
        return False
    return True


def validate_interactions(interactions: pd.DataFrame, report: ValidationReport) -> None:
    """Validate interaction identifiers, ratings, timestamps, and duplicate events."""
    required_columns = {"user_id", "item_id", "rating", "timestamp_ms", "event_time"}
    if not validate_required_columns(interactions, required_columns, "interactions", report):
        return

    null_rows = int(interactions[list(required_columns)].isna().any(axis=1).sum())
    if null_rows:
        report.add(
            "error",
            "interactions",
            "null_required_values",
            "Required values are null.",
            null_rows,
        )

    invalid_ratings = int((~interactions["rating"].between(1.0, 5.0)).sum())
    if invalid_ratings:
        report.add(
            "error",
            "interactions",
            "rating_range",
            "Ratings must be between 1.0 and 5.0.",
            invalid_ratings,
        )

    invalid_timestamps = int((interactions["timestamp_ms"] <= 0).sum())
    if invalid_timestamps:
        report.add(
            "error",
            "interactions",
            "timestamp_range",
            "Timestamps must be positive Unix milliseconds.",
            invalid_timestamps,
        )

    duplicate_events = int(
        interactions.duplicated(subset=["user_id", "item_id", "timestamp_ms"]).sum()
    )
    if duplicate_events:
        report.add(
            "error",
            "interactions",
            "duplicate_events",
            "Duplicate user-item-timestamp events are not allowed.",
            duplicate_events,
        )


def validate_items(items: pd.DataFrame, report: ValidationReport) -> None:
    """Validate item uniqueness and flag items that lack content features."""
    required_columns = {"item_id", "item_text"}
    if not validate_required_columns(items, required_columns, "items", report):
        return

    null_item_ids = int(items["item_id"].isna().sum())
    if null_item_ids:
        report.add("error", "items", "null_item_ids", "Item IDs cannot be null.", null_item_ids)

    duplicate_items = int(items["item_id"].duplicated().sum())
    if duplicate_items:
        report.add(
            "error", "items", "duplicate_item_ids", "Item IDs must be unique.", duplicate_items
        )

    missing_text = int(items["item_text"].fillna("").str.strip().eq("").sum())
    if missing_text:
        report.add(
            "warning",
            "items",
            "missing_item_text",
            "Items without text cannot receive content-based recommendations.",
            missing_text,
        )


def validate_processed_data(interactions: pd.DataFrame, items: pd.DataFrame) -> ValidationReport:
    """Validate both processed datasets against their shared data contract."""
    report = ValidationReport()
    validate_interactions(interactions, report)
    validate_items(items, report)

    if "item_id" in interactions and "item_id" in items:
        unknown_items = int((~interactions["item_id"].isin(items["item_id"])).sum())
        if unknown_items:
            report.add(
                "error",
                "cross_dataset",
                "interaction_item_join",
                "Every interaction item must exist in the item dataset.",
                unknown_items,
            )
    return report


def main() -> None:
    """Validate the local processed Parquet datasets and save a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=INTERACTIONS_FILE)
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DATA_DIR / "validation_report.json"
    )
    arguments = parser.parse_args()

    report = validate_processed_data(
        pd.read_parquet(arguments.interactions), pd.read_parquet(arguments.items)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    if not report.is_valid:
        raise SystemExit("Processed data validation failed.")


if __name__ == "__main__":
    main()
