"""A hybrid recommendation model combining content-based and collaborative filtering.

The hybrid model blends normalised scores from a ContentBasedRecommender and a
CollaborativeFilteringRecommender using configurable per-model weights.  Cold-start
routing is handled at each sub-model level; when both sub-models fall back to
popularity, the hybrid result is effectively popularity-ranked.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.config import PROJECT_ROOT
from recommender.data.preprocess import ITEMS_FILE
from recommender.data.splitting import DEFAULT_SPLITS_DIR
from recommender.evaluation.ranking import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.collaborative_filtering import CollaborativeFilteringRecommender
from recommender.models.hybrid_scores import blend_scores
from recommender.models.popularity import PopularityRecommender

DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "baselines" / "hybrid_metrics.json"


class HybridRecommender:
    """Combine content-based and collaborative filtering scores into a single ranking.

    Parameters
    ----------
    cb_weight:
        Weight assigned to the content-based model after score normalisation.
        Must be non-negative.  Default is 0.5 (equal blend).
    cf_weight:
        Weight assigned to the collaborative filtering model after score
        normalisation.  Must be non-negative.  Default is 0.5 (equal blend).
    cb_max_features:
        Maximum TF-IDF vocabulary size forwarded to ContentBasedRecommender.
    cf_n_factors:
        Number of latent factors forwarded to CollaborativeFilteringRecommender.
    """

    def __init__(
        self,
        cb_weight: float = 0.5,
        cf_weight: float = 0.5,
        cb_max_features: int = 5000,
        cf_n_factors: int = 50,
    ) -> None:
        if cb_weight < 0 or cf_weight < 0:
            raise ValueError("cb_weight and cf_weight must be non-negative.")
        if cb_weight + cf_weight == 0:
            raise ValueError("At least one of cb_weight or cf_weight must be positive.")

        self.cb_weight = cb_weight
        self.cf_weight = cf_weight

        self.cb_model = ContentBasedRecommender(max_features=cb_max_features)
        self.cf_model = CollaborativeFilteringRecommender(n_factors=cf_n_factors)
        self.fallback_recommender = PopularityRecommender()

        # Cache item universe for scoring
        self._all_item_ids: list[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> HybridRecommender:
        """Fit both sub-models and the popularity fallback on *train* data.

        Parameters
        ----------
        train:
            User-item interactions with columns ``user_id``, ``item_id``,
            ``is_positive``, ``timestamp_ms``, ``user_idx``, ``item_idx``.
        items:
            Item metadata with columns ``item_id`` and ``item_text``.
        """
        self.cb_model.fit(train, items)
        self.cf_model.fit(train, items)
        self.fallback_recommender.fit(train, items)

        self._all_item_ids = items["item_id"].tolist()
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def _cb_scores(self, user_id: str, excluded: set[str]) -> dict[str, float]:
        """Return raw content-based cosine scores for all non-excluded items."""
        if not self.cb_model.is_fitted:
            return {}
        if user_id not in self.cb_model.user_profiles:
            return {}

        from recommender.models.similarity import compute_recommendation_scores

        user_profile = self.cb_model.user_profiles[user_id]
        scores = compute_recommendation_scores(user_profile, self.cb_model.item_tfidf_matrix)
        return {
            item_id: float(scores[idx])
            for item_id, idx in self.cb_model.item_id_to_index.items()
            if item_id not in excluded
        }

    def _cf_scores(self, user_id: str, excluded: set[str]) -> dict[str, float]:
        """Return raw collaborative filtering scores for all non-excluded items."""
        if not self.cf_model.is_fitted:
            return {}
        if user_id not in self.cf_model.user_id_to_idx:
            return {}

        user_idx = self.cf_model.user_id_to_idx[user_id]
        try:
            scores = self.cf_model.mf_model.predict(user_idx)
        except IndexError:
            return {}

        return {
            item_id: float(scores[item_idx])
            for item_idx, item_id in self.cf_model.item_idx_to_id.items()
            if item_id not in excluded
        }

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_item_ids: Iterable[str] = (),
    ) -> list[str]:
        """Return the top-K item IDs ranked by blended hybrid score.

        Cold-start routing
        ------------------
        * If the user has no profile in *either* sub-model, the popularity
          fallback is returned directly.
        * If the user is known to only one sub-model, that model's scores are
          used alone (its weight is effectively 1.0).

        Parameters
        ----------
        user_id:
            The identifier of the user to recommend for.
        k:
            Number of items to return.  Must be positive.
        exclude_item_ids:
            Items to suppress from the output (typically the user's history).
        """
        if not self.is_fitted:
            raise ValueError("HybridRecommender must be fitted before recommending.")
        if k <= 0:
            raise ValueError("k must be positive.")

        excluded = set(exclude_item_ids)

        cb_scores = self._cb_scores(user_id, excluded)
        cf_scores = self._cf_scores(user_id, excluded)

        # Both models have no signal — full cold-start fallback
        if not cb_scores and not cf_scores:
            return self.fallback_recommender.recommend(k=k, exclude_item_ids=excluded)

        # Build score dicts and weights for whichever models have signal
        score_dicts: list[dict[str, float]] = []
        weights: list[float] = []
        if cb_scores:
            score_dicts.append(cb_scores)
            weights.append(self.cb_weight)
        if cf_scores:
            score_dicts.append(cf_scores)
            weights.append(self.cf_weight)

        blended = blend_scores(score_dicts=score_dicts, weights=weights)

        # Sort by descending blended score
        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)

        recommendations: list[str] = [item_id for item_id, _ in ranked[:k]]

        # Top up with popularity if blended pool is smaller than k
        if len(recommendations) < k:
            fallback_excluded = excluded | set(recommendations)
            fallback_recs = self.fallback_recommender.recommend(
                k=k - len(recommendations), exclude_item_ids=fallback_excluded
            )
            recommendations.extend(fallback_recs)

        return recommendations

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialize the fitted HybridRecommender to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> HybridRecommender:
        """Load a serialized HybridRecommender from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

def evaluate_hybrid(
    model: HybridRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> dict[str, float | int]:
    """Evaluate top-K hybrid rankings against future positive interactions."""
    test_positive = test.loc[test["is_positive"] == 1]
    relevant_by_user = {
        user_id: set(group["item_id"])
        for user_id, group in test_positive.groupby("user_id")
    }
    history_by_user: dict[str, set[str]] = defaultdict(set)
    for row in train[["user_id", "item_id"]].itertuples(index=False):
        history_by_user[row.user_id].add(row.item_id)

    catalog = set(model._all_item_ids)
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
        recommendations = model.recommend(
            user_id=user_id, k=k, exclude_item_ids=history or set()
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


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def main() -> None:
    """Train and evaluate the hybrid recommender from local split artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS_DIR / "train.parquet")
    parser.add_argument("--test", type=Path, default=DEFAULT_SPLITS_DIR / "test.parquet")
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cb-weight", type=float, default=0.5)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--n-factors", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    arguments = parser.parse_args()

    train = pd.read_parquet(arguments.train)
    test = pd.read_parquet(arguments.test)
    items = pd.read_parquet(arguments.items)

    print("Fitting HybridRecommender...")
    model = HybridRecommender(
        cb_weight=arguments.cb_weight,
        cf_weight=arguments.cf_weight,
        cb_max_features=arguments.max_features,
        cf_n_factors=arguments.n_factors,
    ).fit(train, items)

    print("Evaluating...")
    metrics = evaluate_hybrid(model, train, test, arguments.k)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
