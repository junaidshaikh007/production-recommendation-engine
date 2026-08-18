import gzip
from pathlib import Path

import pytest

from recommender.data.download import DatasetFile, download_file, validate_gzip


def test_validate_gzip_accepts_file_with_json_record(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl.gz"
    with gzip.open(dataset_path, "wt", encoding="utf-8") as file:
        file.write('{"record": 1}\n')

    validate_gzip(dataset_path)


def test_validate_gzip_rejects_invalid_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl.gz"
    dataset_path.write_text("not a gzip file", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid gzip"):
        validate_gzip(dataset_path)


def test_download_reuses_existing_valid_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl.gz"
    with gzip.open(dataset_path, "wt", encoding="utf-8") as file:
        file.write('{"record": 1}\n')
    dataset_file = DatasetFile("dataset.jsonl.gz", "https://example.invalid/data", "test data")

    assert download_file(dataset_file, tmp_path) == dataset_path
