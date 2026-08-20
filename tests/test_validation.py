import pandas as pd

from recommender.data.validation import validate_processed_data


def valid_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user-1", "user-2"],
            "item_id": ["item-1", "item-2"],
            "rating": [5.0, 3.0],
            "timestamp_ms": [1_000, 2_000],
            "event_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        }
    )


def test_valid_data_has_no_errors() -> None:
    items = pd.DataFrame({"item_id": ["item-1", "item-2"], "item_text": ["A", "B"]})

    report = validate_processed_data(valid_interactions(), items)

    assert report.is_valid
    assert report.to_dict()["errors"] == 0


def test_invalid_interactions_and_missing_item_are_reported() -> None:
    interactions = valid_interactions()
    interactions.loc[1, "rating"] = 6.0
    interactions.loc[1, "item_id"] = "missing-item"
    items = pd.DataFrame({"item_id": ["item-1"], "item_text": [""]})

    report = validate_processed_data(interactions, items)
    checks = {issue.check for issue in report.issues}

    assert not report.is_valid
    assert "rating_range" in checks
    assert "interaction_item_join" in checks
    assert "missing_item_text" in checks
