"""A content-based recommendation model using item text metadata."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
import json
from collections import defaultdict
from pathlib import Path

from recommender.config import PROJECT_ROOT
from recommender.data.preprocess import ITEMS_FILE
from recommender.data.splitting import DEFAULT_SPLITS_DIR
from recommender.evaluation.ranking import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommender.data.text_features import build_tfidf_vectors
from recommender.models.similarity import build_user_profiles, compute_recommendation_scores
from recommender.models.popularity import PopularityRecommender

DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "baselines" / "content_based_metrics.json"


class ContentBasedRecommender:
    """Recommend items based on their text similarity to a user's past positive interactions."""

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self.item_ids: list[str] = []
        self.item_id_to_index: dict[str, int] = {}
        self.item_tfidf_matrix: csr_matrix | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.user_profiles: dict[str, np.ndarray] = {}
        
        # Fallback recommender for cold-start users
        self.fallback_recommender = PopularityRecommender()
        self.is_fitted = False

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> ContentBasedRecommender:
        """Build TF-IDF vectors for items and aggregate user profiles from interactions."""
        required_train_columns = {"user_id", "item_id", "is_positive"}
        missing_train = required_train_columns - set(train.columns)
        if missing_train:
            raise ValueError(f"train missing columns: {', '.join(sorted(missing_train))}")
            
        required_item_columns = {"item_id", "item_text"}
        missing_item = required_item_columns - set(items.columns)
        if missing_item:
            raise ValueError(f"items missing columns: {', '.join(sorted(missing_item))}")

        # Build TF-IDF vectors for items
        self.item_ids = items["item_id"].tolist()
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        self.item_tfidf_matrix, self.vectorizer = build_tfidf_vectors(
            items, text_column="item_text", max_features=self.max_features
        )
        
        # Build user profiles
        self.user_profiles = build_user_profiles(
            train_interactions=train,
            item_tfidf_matrix=self.item_tfidf_matrix,
            item_id_to_index=self.item_id_to_index,
        )
        
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
        """Return the top-K recommended items based on content similarity."""
        if not self.is_fitted:
            raise ValueError("ContentBasedRecommender must be fitted before recommending.")
        if k <= 0:
            raise ValueError("k must be positive.")
            
        excluded = set(exclude_item_ids)
        
        if user_id not in self.user_profiles:
            # Cold-start user: fallback to popularity
            return self.fallback_recommender.recommend(k=k, exclude_item_ids=excluded)
            
        # Get user profile and compute scores
        user_profile = self.user_profiles[user_id]
        scores = compute_recommendation_scores(user_profile, self.item_tfidf_matrix)
        
        # Find top k items (using argsort)
        # argsort sorts ascending, so we take the end and reverse it
        top_indices = np.argsort(scores)[::-1]
        
        recommendations: list[str] = []
        for idx in top_indices:
            if len(recommendations) == k:
                break
            item_id = self.item_ids[idx]
            if item_id not in excluded:
                recommendations.append(item_id)
                
        # If we didn't find enough recommendations, fallback to popularity for the rest
        if len(recommendations) < k:
            fallback_excluded = excluded.union(recommendations)
            fallback_recs = self.fallback_recommender.recommend(
                k=k - len(recommendations), exclude_item_ids=fallback_excluded
            )
            recommendations.extend(fallback_recs)
            
        return recommendations


def evaluate_content_based(
    model: ContentBasedRecommender, train: pd.DataFrame, test: pd.DataFrame, k: int = 10
) -> dict[str, float | int]:
    """Evaluate top-K content-based rankings against future positive interactions."""
    test_positive = test.loc[test["is_positive"] == 1]
    relevant_by_user = {
        user_id: set(user_interactions["item_id"])
        for user_id, user_interactions in test_positive.groupby("user_id")
    }
    history_by_user: dict[str, set[str]] = defaultdict(set)
    for row in train[["user_id", "item_id"]].itertuples(index=False):
        history_by_user[row.user_id].add(row.item_id)

    catalog = set(model.item_ids)
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
    """Train and evaluate the content-based baseline from local split artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS_DIR / "train.parquet")
    parser.add_argument("--test", type=Path, default=DEFAULT_SPLITS_DIR / "test.parquet")
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    arguments = parser.parse_args()

    train = pd.read_parquet(arguments.train)
    test = pd.read_parquet(arguments.test)
    items = pd.read_parquet(arguments.items)
    
    print("Fitting ContentBasedRecommender...")
    model = ContentBasedRecommender(max_features=arguments.max_features).fit(train, items)
    
    print("Evaluating...")
    metrics = evaluate_content_based(model, train, test, arguments.k)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
