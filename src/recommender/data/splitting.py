"""Create chronological train, validation, and test datasets without future leakage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from recommender.config import PROCESSED_DATA_DIR
from recommender.data.interactions import LABELED_INTERACTIONS_FILE

DEFAULT_SPLITS_DIR = PROCESSED_DATA_DIR / "splits"


@dataclass(frozen=True)
class SplitConfig:
    """Chronological dataset proportions for model development."""

    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Ensure every split is non-empty and the fractions cover all rows."""
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(fraction <= 0 for fraction in fractions):
            raise ValueError("All split fractions must be positive.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0.")


def chronological_split(
    interactions: pd.DataFrame, config: SplitConfig = SplitConfig()
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into ordered periods using their original event times."""
    if "timestamp_ms" not in interactions:
        raise ValueError("Interactions must contain timestamp_ms for chronological splitting.")
    if len(interactions) < 3:
        raise ValueError("At least three interactions are required for a three-way split.")

    ordered = interactions.sort_values("timestamp_ms", kind="stable").reset_index(drop=True)
    train_target = int(len(ordered) * config.train_fraction)
    validation_target = int(len(ordered) * config.validation_fraction)
    if train_target == 0 or validation_target == 0:
        raise ValueError("Split fractions create an empty train, validation, or test dataset.")

    train_cutoff = ordered.iloc[train_target - 1]["timestamp_ms"]
    train = ordered.loc[ordered["timestamp_ms"] <= train_cutoff].copy()
    remaining = ordered.loc[ordered["timestamp_ms"] > train_cutoff].copy()
    if len(remaining) <= validation_target:
        raise ValueError("Not enough later interactions for validation and test datasets.")

    validation_cutoff = remaining.iloc[validation_target - 1]["timestamp_ms"]
    validation = remaining.loc[remaining["timestamp_ms"] <= validation_cutoff].copy()
    test = remaining.loc[remaining["timestamp_ms"] > validation_cutoff].copy()
    if validation.empty or test.empty:
        raise ValueError("Timestamp ties created an empty validation or test dataset.")
    return train, validation, test


def split_summary(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, object]:
    """Summarize split sizes, temporal boundaries, and cold-start exposure."""
    train_users = set(train["user_id"])
    train_items = set(train["item_id"])

    def describe(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "interactions": len(frame),
            "users": int(frame["user_id"].nunique()),
            "items": int(frame["item_id"].nunique()),
            "start_timestamp_ms": int(frame["timestamp_ms"].min()),
            "end_timestamp_ms": int(frame["timestamp_ms"].max()),
        }

    def cold_start(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "new_user_rate_vs_train": round((~frame["user_id"].isin(train_users)).mean(), 6),
            "new_item_rate_vs_train": round((~frame["item_id"].isin(train_items)).mean(), 6),
        }

    return {
        "train": describe(train),
        "validation": {**describe(validation), **cold_start(validation)},
        "test": {**describe(test), **cold_start(test)},
    }


def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: SplitConfig,
    output_dir: Path = DEFAULT_SPLITS_DIR,
) -> dict[str, object]:
    """Persist chronological splits and a machine-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    summary = {"config": asdict(config), **split_summary(train, validation, test)}
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Create local chronological splits from labeled interactions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LABELED_INTERACTIONS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    arguments = parser.parse_args()

    config = SplitConfig(
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
    )
    train, validation, test = chronological_split(pd.read_parquet(arguments.input), config)
    summary = save_splits(train, validation, test, config, arguments.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
