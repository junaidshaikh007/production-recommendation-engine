"""A collaborative filtering recommendation model using matrix factorization."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.config import PROJECT_ROOT
from recommender.data.matrix import create_interaction_matrix
from recommender.data.preprocess import ITEMS_FILE
from recommender.data.splitting import DEFAULT_SPLITS_DIR
from recommender.evaluation.ranking import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommender.models.matrix_factorization import MatrixFactorizationModel
from recommender.models.popularity import PopularityRecommender

DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "baselines" / "collaborative_filtering_metrics.json"


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


def evaluate_collaborative_filtering(
    model: CollaborativeFilteringRecommender, train: pd.DataFrame, test: pd.DataFrame, k: int = 10
) -> dict[str, float | int]:
    """Evaluate top-K collaborative filtering rankings against future positive interactions."""
    test_positive = test.loc[test["is_positive"] == 1]
    relevant_by_user = {
        user_id: set(user_interactions["item_id"])
        for user_id, user_interactions in test_positive.groupby("user_id")
    }
    history_by_user: dict[str, set[str]] = defaultdict(set)
    for row in train[["user_id", "item_id"]].itertuples(index=False):
        history_by_user[row.user_id].add(row.item_id)

    catalog = set(model.item_idx_to_id.values())
    catalog_covered_users = sum(
        bool(relevant_items & catalog) for relevant_items in relevant_by_user.values()
    )
    evaluable_by_user = {
        user_id: relevant_items & catalog
        for user_id, relevant_items in relevant_by_user.items()
        if user_id in history_by_user and relevant_items & catalog
    }
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    map_scores: list[float] = []
    ndcg_scores: list[float] = []

    for user_id, relevant_items in evaluable_by_user.items():
        history = history_by_user.get(user_id)
        recommendations = model.recommend(user_id=user_id, k=k, exclude_item_ids=history or set())
        precision_scores.append(precision_at_k(recommendations, relevant_items, k))
        recall_scores.append(recall_at_k(recommendations, relevant_items, k))
        map_scores.append(map_at_k(recommendations, relevant_items, k))
        ndcg_scores.append(ndcg_at_k(recommendations, relevant_items, k))

    test_positive_users = len(relevant_by_user)
    user_count = len(evaluable_by_user)
    return {
        "test_positive_users": test_positive_users,
        "evaluated_users": user_count,
        "catalog_covered_user_rate": (
            round(catalog_covered_users / test_positive_users, 6) if test_positive_users else 0.0
        ),
        f"precision_at_{k}": round(sum(precision_scores) / user_count, 6) if user_count else 0.0,
        f"recall_at_{k}": round(sum(recall_scores) / user_count, 6) if user_count else 0.0,
        f"map_at_{k}": round(sum(map_scores) / user_count, 6) if user_count else 0.0,
        f"ndcg_at_{k}": round(sum(ndcg_scores) / user_count, 6) if user_count else 0.0,
    }


def main() -> None:
    """Train and evaluate the collaborative filtering model from local split artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS_DIR / "train.parquet")
    parser.add_argument("--test", type=Path, default=DEFAULT_SPLITS_DIR / "test.parquet")
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-factors", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    arguments = parser.parse_args()

    train = pd.read_parquet(arguments.train)
    test = pd.read_parquet(arguments.test)
    items = pd.read_parquet(arguments.items)

    print("Fitting CollaborativeFilteringRecommender...")
    model = CollaborativeFilteringRecommender(n_factors=arguments.n_factors).fit(train, items)

    print("Evaluating...")
    metrics = evaluate_collaborative_filtering(model, train, test, arguments.k)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
