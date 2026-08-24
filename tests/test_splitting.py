import pandas as pd
import pytest

from recommender.data.splitting import SplitConfig, chronological_split, split_summary


def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9"],
            "item_id": ["i1", "i2", "i1", "i3", "i4", "i5", "i6", "i7", "i8", "i9"],
            "timestamp_ms": [900, 100, 800, 200, 700, 300, 600, 400, 500, 1_000],
        }
    )


def test_chronological_split_preserves_time_order() -> None:
    train, validation, test = chronological_split(interactions())

    assert len(train) == 8
    assert len(validation) == 1
    assert len(test) == 1
    assert train["timestamp_ms"].max() < validation["timestamp_ms"].min()
    assert validation["timestamp_ms"].max() < test["timestamp_ms"].min()


def test_split_summary_reports_new_users_and_items() -> None:
    train, validation, test = chronological_split(interactions())

    summary = split_summary(train, validation, test)

    assert summary["validation"]["new_user_rate_vs_train"] == 0.0
    assert summary["test"]["new_item_rate_vs_train"] == 1.0


def test_split_config_requires_fractions_to_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(0.7, 0.2, 0.2)
