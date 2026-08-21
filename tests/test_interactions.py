import pandas as pd
import pytest

from recommender.data.interactions import InteractionPolicy, label_interactions, summarize_labels


def test_label_interactions_uses_configured_positive_threshold() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["user-1", "user-1"],
            "item_id": ["item-1", "item-2"],
            "rating": [3.0, 4.0],
            "timestamp_ms": [2_000, 1_000],
            "event_time": pd.to_datetime(["2024-01-02", "2024-01-01"], utc=True),
        }
    )

    labeled = label_interactions(interactions, InteractionPolicy(4.0))

    assert labeled["item_id"].tolist() == ["item-2", "item-1"]
    assert labeled["is_positive"].tolist() == [1, 0]
    assert summarize_labels(labeled)["positive_interaction_rate"] == 0.5


def test_policy_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between 1.0 and 5.0"):
        InteractionPolicy(5.1)
