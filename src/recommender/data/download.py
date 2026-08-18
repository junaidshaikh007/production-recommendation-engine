"""Download and validate the raw Amazon Reviews 2023 All Beauty dataset."""

from __future__ import annotations

import argparse
import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from recommender.config import RAW_DATA_DIR


@dataclass(frozen=True)
class DatasetFile:
    """A source file required for the recommendation-engine dataset."""

    filename: str
    url: str
    purpose: str


DATASET_FILES = (
    DatasetFile(
        filename="All_Beauty.jsonl.gz",
        url=(
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
            "raw/review_categories/All_Beauty.jsonl.gz"
        ),
        purpose="User review interactions",
    ),
    DatasetFile(
        filename="meta_All_Beauty.jsonl.gz",
        url=(
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
            "raw/meta_categories/meta_All_Beauty.jsonl.gz"
        ),
        purpose="Product metadata",
    ),
)


def download_file(dataset_file: DatasetFile, destination_dir: Path = RAW_DATA_DIR) -> Path:
    """Download one dataset file unless a valid local copy already exists."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / dataset_file.filename

    if destination.exists() and destination.stat().st_size > 0:
        validate_gzip(destination)
        print(f"Using existing file: {destination.name}")
        return destination

    print(f"Downloading {dataset_file.purpose}: {destination.name}")
    with urlopen(dataset_file.url, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)

    validate_gzip(destination)
    print(f"Saved {destination.name} ({destination.stat().st_size / 1_000_000:.1f} MB)")
    return destination


def validate_gzip(path: Path) -> None:
    """Ensure the file is a readable gzip stream containing at least one record."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as compressed_file:
            if not compressed_file.readline().strip():
                raise ValueError("file contains no records")
    except (gzip.BadGzipFile, OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid gzip dataset file: {path}") from error


def main() -> None:
    """Download either all raw data files or a specific dataset file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only product metadata.",
    )
    arguments = parser.parse_args()
    files = DATASET_FILES[1:] if arguments.metadata_only else DATASET_FILES

    for dataset_file in files:
        download_file(dataset_file)


if __name__ == "__main__":
    main()

