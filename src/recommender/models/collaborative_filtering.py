"""A collaborative filtering recommendation model using matrix factorization."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from recommender.data.matrix import create_interaction_matrix
from recommender.models.matrix_factorization import MatrixFactorizationModel
from recommender.models.popularity import PopularityRecommender


class CollaborativeFilteringRecommender:
    """Recommend items based on collaborative filtering (matrix factorization)."""

    def __init__(self, n_factors: int = 50, fallback_recommender: object | None = None) -> None:
        self.n_factors = n_factors
        self.mf_model = MatrixFactorizationModel(n_factors=self.n_factors)
        self.fallback_recommender = fallback_recommender or PopularityRecommender()

        self.user_id_to_idx: dict[str, int] = {}
        self.item_idx_to_id: dict[int, str] = {}
        self.is_fitted = False

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> CollaborativeFilteringRecommender:
        """Construct interaction matrix and fit the factorization model."""
        required_train_columns = {"user_id", "item_id", "user_idx", "item_idx"}
        missing_train = required_train_columns - set(train.columns)
        if missing_train:
            raise ValueError(f"train missing columns: {', '.join(sorted(missing_train))}")

        # Build ID mappings from training data
        self.user_id_to_idx = dict(zip(train["user_id"], train["user_idx"]))
        self.item_idx_to_id = dict(zip(train["item_idx"], train["item_id"]))

        # Get matrix dimensions safely
        num_users = train["user_idx"].max() + 1
        num_items = train["item_idx"].max() + 1

        # Create sparse interaction matrix
        interaction_matrix = create_interaction_matrix(
            interactions=train,
            num_users=num_users,
            num_items=num_items,
        )

        # Fit MF model
        self.mf_model.fit(interaction_matrix)

        # Fit fallback recommender
        self.fallback_recommender.fit(train, items)

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_item_ids: Iterable[str] = (),
    ) -> list[str]:
        """Return the top-K recommended items based on collaborative filtering."""
        if not self.is_fitted:
            raise ValueError("CollaborativeFilteringRecommender must be fitted before recommending.")
        if k <= 0:
            raise ValueError("k must be positive.")

        excluded = set(exclude_item_ids)

        if user_id not in self.user_id_to_idx:
            # Cold-start user: fallback to popularity
            return self.fallback_recommender.recommend(k=k, exclude_item_ids=excluded)

        user_idx = self.user_id_to_idx[user_id]
        
        # Predict scores for all items
        try:
            scores = self.mf_model.predict(user_idx)
        except IndexError:
            # Safe fallback if user index is out of bounds
            return self.fallback_recommender.recommend(k=k, exclude_item_ids=excluded)

        # Find top items using argsort
        top_indices = np.argsort(scores)[::-1]

        recommendations: list[str] = []
        for idx in top_indices:
            if len(recommendations) == k:
                break
            
            # Map index back to item_id, skip if unknown
            item_id = self.item_idx_to_id.get(idx)
            if item_id and item_id not in excluded:
                recommendations.append(item_id)

        # If we didn't find enough recommendations, fallback to popularity for the rest
        if len(recommendations) < k:
            fallback_excluded = excluded.union(recommendations)
            fallback_recs = self.fallback_recommender.recommend(
                k=k - len(recommendations), exclude_item_ids=fallback_excluded
            )
            recommendations.extend(fallback_recs)

        return recommendations
