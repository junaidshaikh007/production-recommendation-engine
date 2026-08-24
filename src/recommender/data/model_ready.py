"""Fit training-only encoders and prepare numerical recommender-model inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recommender.config import PROJECT_ROOT
from recommender.data.encoding import (
    UNKNOWN_INDEX,
    EncoderBundle,
    encode_interactions,
    fit_encoders,
    save_encoders,
)
from recommender.data.splitting import DEFAULT_SPLITS_DIR

DEFAULT_OUTPUT_DIR = DEFAULT_SPLITS_DIR / "encoded"
DEFAULT_ENCODERS_FILE = PROJECT_ROOT / "artifacts" / "encoders.json"


def prepare_model_ready_splits(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], EncoderBundle, dict[str, object]]:
    """Encode splits from training data only and summarize unknown entities."""
    encoders = fit_encoders(train)
    encoded_splits = {
        "train": encode_interactions(train, encoders),
        "validation": encode_interactions(validation, encoders),
        "test": encode_interactions(test, encoders),
    }

    def unknown_rate(frame: pd.DataFrame, column: str) -> float:
        return round((frame[column] == UNKNOWN_INDEX).mean(), 6)

    summary = {
        "training_entities": {
            "users": encoders.user_encoder.size,
            "items": encoders.item_encoder.size,
        },
        "unknown_entity_rates": {
            split_name: {
                "users": unknown_rate(frame, "user_idx"),
                "items": unknown_rate(frame, "item_idx"),
            }
            for split_name, frame in encoded_splits.items()
        },
    }
    return encoded_splits, encoders, summary


def save_model_ready_splits(
    encoded_splits: dict[str, pd.DataFrame],
    encoders_path: Path,
    encoders: EncoderBundle,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Persist encoded splits, mappings, and an artifact summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in encoded_splits.items():
        frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)
    save_encoders(encoders, encoders_path)
    (output_dir / "encoding_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    """Create model-ready split files from local chronological split artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--encoders-output", type=Path, default=DEFAULT_ENCODERS_FILE)
    arguments = parser.parse_args()

    split_frames = {
        split_name: pd.read_parquet(arguments.splits_dir / f"{split_name}.parquet")
        for split_name in ("train", "validation", "test")
    }
    encoded_splits, encoders, summary = prepare_model_ready_splits(**split_frames)
    save_model_ready_splits(
        encoded_splits,
        arguments.encoders_output,
        encoders,
        summary,
        arguments.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
