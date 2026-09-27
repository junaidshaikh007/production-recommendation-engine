import pandas as pd

from recommender.models.popularity import PopularityRecommender, evaluate_popularity


def train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "item_id": ["i1", "i1", "i2", "i2", "i3"],
            "is_positive": [1, 1, 1, 0, 1],
            "timestamp_ms": [1, 2, 3, 4, 5],
        }
    )


def test_popularity_recommender_ranks_positive_interactions_first() -> None:
    model = PopularityRecommender().fit(train_data())

    assert model.recommend(k=3) == ["i1", "i2", "i3"]
    assert model.recommend(k=2, exclude_item_ids={"i1"}) == ["i2", "i3"]


def test_store_recommendations_fall_back_to_global_ranking() -> None:
    items = pd.DataFrame(
        {"item_id": ["i1", "i2", "i3"], "store": ["store-a", "store-b", "store-a"]}
    )
    model = PopularityRecommender().fit(train_data(), items)

    assert model.recommend_for_store("store-a", k=2) == ["i1", "i3"]
    assert model.recommend_for_store("missing-store", k=2) == ["i1", "i2"]


def test_evaluation_excludes_seen_items_from_known_users() -> None:
    model = PopularityRecommender().fit(train_data())
    test = pd.DataFrame({"user_id": ["u1"], "item_id": ["i2"], "is_positive": [1]})

    metrics = evaluate_popularity(model, train_data(), test, k=1)

    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
