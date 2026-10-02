"""Weighted score blending utility for the hybrid recommender.

Given score dictionaries from two or more sub-models (each mapping item_id ->
raw score), this module normalises each model's scores to [0, 1] and combines
them into a single blended score using configurable per-model weights.

Typical usage
-------------
>>> from recommender.models.hybrid_scores import blend_scores
>>> cb_scores  = {"item_a": 0.9, "item_b": 0.4, "item_c": 0.1}
>>> cf_scores  = {"item_a": 0.3, "item_b": 0.8, "item_c": 0.6}
>>> blended = blend_scores(
...     score_dicts=[cb_scores, cf_scores],
...     weights=[0.4, 0.6],
... )
"""

from __future__ import annotations

from collections import defaultdict


def _min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    """Return a copy of *scores* with values normalised to [0, 1].

    If all values are identical (including the degenerate single-item case),
    every item receives a score of 1.0 so that it is still eligible for
    recommendation.
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    score_range = max_score - min_score

    if score_range == 0.0:
        return {item_id: 1.0 for item_id in scores}

    return {
        item_id: (raw - min_score) / score_range
        for item_id, raw in scores.items()
    }


def blend_scores(
    score_dicts: list[dict[str, float]],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Merge multiple item-score dictionaries into one blended score per item.

    Each model's scores are independently min-max normalised before blending.
    Items that appear in only a subset of models receive a score of 0.0 for
    the models that did not rank them — this penalises items unseen by a model
    without discarding them entirely.

    Parameters
    ----------
    score_dicts:
        One dictionary per sub-model mapping ``item_id -> raw score``.  The
        list must contain at least one entry.
    weights:
        Per-model weights applied after normalisation.  Must be the same
        length as *score_dicts*.  Values need not sum to 1 — they are
        normalised internally.  When *None*, equal weights are used.

    Returns
    -------
    dict[str, float]
        Mapping of ``item_id -> blended score`` across all items seen by any
        model.

    Raises
    ------
    ValueError
        If *score_dicts* is empty, *weights* length does not match
        *score_dicts*, or any weight is negative.
    """
    if not score_dicts:
        raise ValueError("score_dicts must contain at least one score dictionary.")

    n_models = len(score_dicts)

    if weights is None:
        weights = [1.0] * n_models
    else:
        weights = list(weights)

    if len(weights) != n_models:
        raise ValueError(
            f"weights length ({len(weights)}) must match score_dicts length ({n_models})."
        )
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative.")

    total_weight = sum(weights)
    if total_weight == 0.0:
        raise ValueError("At least one weight must be positive.")

    # Normalise weights so they sum to 1
    normalised_weights = [w / total_weight for w in weights]

    # Normalise each model's raw scores to [0, 1]
    normalised_dicts = [_min_max_normalise(sd) for sd in score_dicts]

    # Accumulate weighted scores; items missing from a model contribute 0
    blended: dict[str, float] = defaultdict(float)
    for norm_scores, weight in zip(normalised_dicts, normalised_weights):
        for item_id, score in norm_scores.items():
            blended[item_id] += weight * score

    return dict(blended)
