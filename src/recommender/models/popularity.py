"""A user-independent popularity and trending recommendation baseline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from recommender.config import PROJECT_ROOT
from recommender.data.preprocess import ITEMS_FILE
from recommender.data.splitting import DEFAULT_SPLITS_DIR
from recommender.evaluation.ranking import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "baselines" / "popularity_metrics.json"


class PopularityRecommender:
    """Rank items by positive interactions observed in the training period."""

    def __init__(self, trending_window_days: int = 90) -> None:
        if trending_window_days <= 0:
            raise ValueError("trending_window_days must be positive.")
        self.trending_window_days = trending_window_days
        self.ranking = pd.DataFrame()
        self.trending_ranking = pd.DataFrame()
        self.store_rankings: dict[str, list[str]] = {}

    def fit(self, train: pd.DataFrame, items: pd.DataFrame | None = None) -> PopularityRecommender:
        """Fit global, recent-trending, and store-specific rankings on train data only."""
        required_columns = {"item_id", "is_positive", "timestamp_ms"}
        missing_columns = required_columns - set(train.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Cannot fit popularity baseline; missing columns: {missing}.")

        self.ranking = self._build_ranking(train)
        cutoff_ms = train["timestamp_ms"].max() - self.trending_window_days * 86_400_000
        recent_train = train.loc[train["timestamp_ms"] >= cutoff_ms]
        self.trending_ranking = self._build_ranking(recent_train)
        self.store_rankings = self._build_store_rankings(train, items)
        return self

    @staticmethod
    def _build_ranking(interactions: pd.DataFrame) -> pd.DataFrame:
        grouped = interactions.groupby("item_id", as_index=False).agg(
            interaction_count=("is_positive", "size"),
            positive_interaction_count=("is_positive", "sum"),
        )
        grouped["positive_rate"] = (
            grouped["positive_interaction_count"] / grouped["interaction_count"]
        )
        return grouped.sort_values(
            ["positive_interaction_count", "interaction_count", "item_id"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)

    def _build_store_rankings(
        self, train: pd.DataFrame, items: pd.DataFrame | None
    ) -> dict[str, list[str]]:
        if items is None or "store" not in items:
            return {}
        item_stores = items[["item_id", "store"]].dropna()
        item_stores = item_stores.loc[item_stores["store"].str.strip().ne("")]
        joined = train.merge(item_stores, on="item_id", how="inner")
        if joined.empty:
            return {}
        grouped = joined.groupby(["store", "item_id"], as_index=False).agg(
            interaction_count=("is_positive", "size"),
            positive_interaction_count=("is_positive", "sum"),
        )
        ranked = grouped.sort_values(
            ["store", "positive_interaction_count", "interaction_count", "item_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        return {
            str(store): group["item_id"].tolist() for store, group in ranked.groupby("store")
        }

    def recommend(
        self,
        k: int = 10,
        exclude_item_ids: Iterable[str] = (),
        strategy: str = "popular",
    ) -> list[str]:
        """Return the top unseen items using global popularity or recent trend ranking."""
        if self.ranking.empty:
            raise ValueError("PopularityRecommender must be fitted before recommending.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if strategy not in {"popular", "trending"}:
            raise ValueError("strategy must be 'popular' or 'trending'.")
        ranking = self.trending_ranking if strategy == "trending" else self.ranking
        excluded = set(exclude_item_ids)
        recommendations: list[str] = []
        for item_id in ranking["item_id"]:
            if item_id not in excluded:
                recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_for_store(
        self, store: str, k: int = 10, exclude_item_ids: Iterable[str] = ()
    ) -> list[str]:
        """Return store-specific popular items, falling back to global popularity."""
        excluded = set(exclude_item_ids)
        candidates = self.store_rankings.get(store, [])
        recommendations = [item_id for item_id in candidates if item_id not in excluded]
        if len(recommendations) < k:
            recommendations.extend(
                item_id
                for item_id in self.recommend(k=len(self.ranking), exclude_item_ids=excluded)
                if item_id not in recommendations
            )
        return recommendations[:k]


def evaluate_popularity(
    model: PopularityRecommender, train: pd.DataFrame, test: pd.DataFrame, k: int = 10
) -> dict[str, float | int]:
    """Evaluate top-K popularity rankings against future positive interactions."""
    test_positive = test.loc[test["is_positive"] == 1]
    relevant_by_user = {
        user_id: set(user_interactions["item_id"])
        for user_id, user_interactions in test_positive.groupby("user_id")
    }
    history_by_user: dict[str, set[str]] = defaultdict(set)
    for row in train[["user_id", "item_id"]].itertuples(index=False):
        history_by_user[row.user_id].add(row.item_id)

    catalog = set(model.ranking["item_id"])
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
    anonymous_recommendations = model.recommend(k=k)

    for user_id, relevant_items in evaluable_by_user.items():
        history = history_by_user.get(user_id)
        recommendations = (
            model.recommend(k=k, exclude_item_ids=history) if history else anonymous_recommendations
        )
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
    """Train and evaluate the popularity baseline from local split artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS_DIR / "train.parquet")
    parser.add_argument("--test", type=Path, default=DEFAULT_SPLITS_DIR / "test.parquet")
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    arguments = parser.parse_args()

    train = pd.read_parquet(arguments.train)
    test = pd.read_parquet(arguments.test)
    items = pd.read_parquet(arguments.items)
    model = PopularityRecommender().fit(train, items)
    metrics = evaluate_popularity(model, train, test, arguments.k)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
