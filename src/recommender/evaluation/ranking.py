"""Ranking metrics used to evaluate top-K recommendation results."""

from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    """Return the fraction of the first K recommendations that are relevant."""
    recommendations = list(recommended)[:k]
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(item_id in relevant for item_id in recommendations) / k


def recall_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    """Return the fraction of relevant items retrieved in the first K results."""
    if not relevant:
        return 0.0
    return sum(item_id in relevant for item_id in list(recommended)[:k]) / len(relevant)


def map_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    """Return average precision, normalized by available relevant top-K items."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item_id in enumerate(list(recommended)[:k], start=1):
        if item_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def ndcg_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    """Return normalized discounted cumulative gain for binary relevance."""
    if not relevant:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, item_id in enumerate(list(recommended)[:k], start=1)
        if item_id in relevant
    )
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal_dcg

