import pandas as pd

from recommender.data.features import generate_item_features, generate_training_history_features


def test_generate_item_features_handles_missing_text_and_price() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "title": ["A title", None],
            "description": ["Description", ""],
            "features": ["Feature", ""],
            "item_text": ["A title Description Feature", ""],
            "price": [9.0, None],
        }
    )

    features = generate_item_features(items)

    assert features["has_item_text"].tolist() == [1, 0]
    assert features["has_price"].tolist() == [1, 0]
    assert features.loc[0, "price_log1p"] > 0
    assert features.loc[1, "price_log1p"] == 0


def test_training_history_features_use_only_train_interactions() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i1"],
            "is_positive": [1, 0, 1],
        }
    )

    user_features, item_features = generate_training_history_features(train)

    assert user_features.loc[user_features["user_id"] == "u1", "user_positive_rate"].item() == 0.5
    assert item_features.loc[item_features["item_id"] == "i1", "item_positive_count"].item() == 2
