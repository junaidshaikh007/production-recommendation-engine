from pathlib import Path

import pandas as pd

from recommender.data.encoding import UNKNOWN_INDEX, load_encoders, save_encoders
from recommender.data.model_ready import prepare_model_ready_splits


def split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"]})
    validation = pd.DataFrame({"user_id": ["u1", "new-user"], "item_id": ["i2", "new-item"]})
    test = pd.DataFrame({"user_id": ["u2"], "item_id": ["i1"]})
    return train, validation, test


def test_prepare_model_ready_splits_marks_entities_unseen_in_training() -> None:
    encoded, _, summary = prepare_model_ready_splits(*split_data())

    assert encoded["train"]["user_idx"].tolist() == [0, 1]
    assert encoded["validation"]["user_idx"].tolist() == [0, UNKNOWN_INDEX]
    assert encoded["validation"]["item_idx"].tolist() == [1, UNKNOWN_INDEX]
    assert summary["unknown_entity_rates"]["validation"]["users"] == 0.5


def test_saved_encoders_are_reusable(tmp_path: Path) -> None:
    _, encoders, _ = prepare_model_ready_splits(*split_data())
    path = tmp_path / "encoders.json"

    save_encoders(encoders, path)
    restored = load_encoders(path)

    assert restored.user_encoder.transform(["u2", "missing"]) == [1, UNKNOWN_INDEX]
