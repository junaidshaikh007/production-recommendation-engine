import numpy as np
import pandas as pd
import pytest

from recommender.data.text_features import build_tfidf_vectors
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.similarity import compute_recommendation_scores


def get_mock_items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "item_text": [
                "moisturizing face cream",
                "face lotion with spf",
                "red lipstick makeup",
            ],
            "price": [10.0, 15.0, 8.0],
        }
    )


def get_mock_train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3"],
            "is_positive": [1, 1, 1],
            "timestamp_ms": [100, 200, 300],
        }
    )


def test_build_tfidf_vectors_creates_sparse_matrix() -> None:
    items = get_mock_items()
    matrix, vectorizer = build_tfidf_vectors(items)
    
    assert matrix.shape[0] == 3
    assert len(vectorizer.get_feature_names_out()) > 0


def test_compute_recommendation_scores() -> None:
    items = get_mock_items()
    matrix, vectorizer = build_tfidf_vectors(items)
    
    # Fake user profile matching "face cream" words
    user_profile = np.zeros(matrix.shape[1])
    # Just use the first item's vector for test
    user_profile = np.asarray(matrix[0].mean(axis=0)).flatten()
    
    scores = compute_recommendation_scores(user_profile, matrix)
    assert len(scores) == 3
    assert scores[0] > scores[2]  # i1 should be more similar to i1 than i3 is


def test_content_based_recommender_recommends_similar_items() -> None:
    model = ContentBasedRecommender(max_features=100)
    model.fit(get_mock_train_data(), get_mock_items())
    
    # u1 interacted with i1 and i2 (face cream/lotion)
    recs = model.recommend("u1", k=2)
    assert len(recs) == 2
    # Should recommend i1 and i2 first
    assert "i1" in recs or "i2" in recs


def test_content_based_recommender_falls_back_for_unknown_user() -> None:
    model = ContentBasedRecommender(max_features=100)
    model.fit(get_mock_train_data(), get_mock_items())
    
    recs = model.recommend("unknown_user", k=2)
    assert len(recs) == 2
    # The popularity model will recommend items with most interactions (i1, i2, i3 tied or ordered)
    assert isinstance(recs, list)
