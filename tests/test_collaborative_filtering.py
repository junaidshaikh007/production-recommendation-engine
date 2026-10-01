"""Unit tests for collaborative filtering logic and models."""

import numpy as np
import pandas as pd
import pytest

from recommender.data.matrix import create_interaction_matrix
from recommender.models.collaborative_filtering import CollaborativeFilteringRecommender
from recommender.models.matrix_factorization import MatrixFactorizationModel


def test_create_interaction_matrix() -> None:
    """Test converting a DataFrame into a sparse interaction matrix."""
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, -1],  # -1 should be ignored
            "item_idx": [1, 2, 0, 2],
            "rating": [5.0, 3.0, 4.0, 2.0],
        }
    )
    matrix = create_interaction_matrix(interactions, num_users=2, num_items=3, weight_column="rating")
    
    assert matrix.shape == (2, 3)
    dense = matrix.todense()
    
    # user 0, item 1 -> 5.0
    assert dense[0, 1] == 5.0
    # user 0, item 2 -> 3.0
    assert dense[0, 2] == 3.0
    # user 1, item 0 -> 4.0
    assert dense[1, 0] == 4.0
    # user 1, item 2 -> 0.0 (no interaction)
    assert dense[1, 2] == 0.0


def test_matrix_factorization_model() -> None:
    """Test the base SVD factorization logic."""
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "item_idx": [0, 1, 1, 2, 0],
            "rating": [5.0, 4.0, 4.0, 5.0, 1.0],
        }
    )
    matrix = create_interaction_matrix(interactions, num_users=3, num_items=3)
    
    # We must use a small n_factors since matrix is only 3x3
    model = MatrixFactorizationModel(n_factors=1)
    model.fit(matrix)
    
    assert model.user_factors is not None
    assert model.item_factors is not None
    assert model.user_factors.shape[1] == 1
    assert model.item_factors.shape[0] == 1
    
    # Predict for user 0
    scores = model.predict(0)
    assert len(scores) == 3


def test_collaborative_filtering_recommender() -> None:
    """Test the CollaborativeFilteringRecommender interface and cold-start fallback."""
    train_data = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i2", "i3"],
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 1, 2],
            "rating": [1.0, 1.0, 1.0, 1.0],
            "is_positive": [1, 1, 1, 1],
        }
    )
    items_data = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "item_text": ["text1", "text2", "text3", "text4"],
        }
    )
    
    recommender = CollaborativeFilteringRecommender(n_factors=1)
    recommender.fit(train_data, items_data)
    
    # Test known user recommendation
    recs_u1 = recommender.recommend("u1", k=2, exclude_item_ids={"i1", "i2"})
    assert len(recs_u1) == 2
    
    # Test cold-start user recommendation (should fallback to popularity)
    recs_cold = recommender.recommend("new_user", k=2)
    assert len(recs_cold) == 2
    assert "i2" in recs_cold  # i2 is the most popular item in train_data
