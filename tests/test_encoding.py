import pandas as pd
import pytest

from recommender.data.encoding import (
    UNKNOWN_INDEX,
    EncoderBundle,
    encode_interactions,
    fit_encoders,
)


def training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user-1", "user-2", "user-1"],
            "item_id": ["item-1", "item-1", "item-2"],
        }
    )


def test_encoders_assign_contiguous_indices_and_mark_unknowns() -> None:
    encoders = fit_encoders(training_data())
    evaluation = pd.DataFrame(
        {
            "user_id": ["user-1", "new-user"],
            "item_id": ["item-2", "new-item"],
        }
    )

    encoded = encode_interactions(evaluation, encoders)

    assert encoders.user_encoder.size == 2
    assert encoders.item_encoder.size == 2
    assert encoded["user_idx"].tolist() == [0, UNKNOWN_INDEX]
    assert encoded["item_idx"].tolist() == [1, UNKNOWN_INDEX]


def test_encoder_bundle_round_trip_preserves_mappings() -> None:
    encoders = fit_encoders(training_data())

    restored = EncoderBundle.from_dict(encoders.to_dict())

    assert restored.item_encoder.inverse_transform([0, 1, UNKNOWN_INDEX]) == [
        "item-1",
        "item-2",
        None,
    ]


def test_encoder_requires_user_and_item_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        fit_encoders(pd.DataFrame({"user_id": ["user-1"]}))
