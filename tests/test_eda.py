import pandas as pd

from recommender.analysis.eda import calculate_summary


def test_calculate_summary_reports_recommendation_sparsity() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["user-1", "user-1", "user-2"],
            "item_id": ["item-1", "item-2", "item-1"],
            "rating": [5.0, 3.0, 4.0],
        }
    )
    items = pd.DataFrame({"item_id": ["item-1", "item-2"]})

    summary = calculate_summary(interactions, items)

    assert summary["interactions"] == 3
    assert summary["positive_interaction_rate"] == 0.6667
    assert summary["matrix_density"] == 0.75
    assert summary["sparsity"] == 0.25
