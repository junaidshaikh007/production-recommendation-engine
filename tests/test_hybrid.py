"""Unit tests for hybrid scoring and HybridRecommender."""

import pandas as pd
import pytest

from recommender.models.hybrid_scores import blend_scores, _min_max_normalise
from recommender.models.hybrid import HybridRecommender, evaluate_hybrid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def get_mock_items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "item_text": [
                "moisturizing face cream",
                "face lotion with spf",
                "red lipstick makeup",
                "hair conditioner repair",
            ],
            "price": [10.0, 15.0, 8.0, 12.0],
        }
    )


def get_mock_train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id":    ["u1", "u1", "u2", "u2", "u3"],
            "item_id":    ["i1", "i2", "i2", "i3", "i1"],
            "user_idx":   [0,    0,    1,    1,    2],
            "item_idx":   [0,    1,    1,    2,    0],
            "is_positive":[1,    1,    1,    1,    1],
            "timestamp_ms":[100, 200,  300,  400,  500],
            "rating":     [1.0,  1.0,  1.0,  1.0,  1.0],
        }
    )


def get_mock_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id":     ["u1", "u2"],
            "item_id":     ["i3", "i4"],
            "is_positive": [1,    1],
            "timestamp_ms":[600,  700],
        }
    )


# ---------------------------------------------------------------------------
# blend_scores / _min_max_normalise tests
# ---------------------------------------------------------------------------

def test_min_max_normalise_standard() -> None:
    """Scores should be scaled to [0, 1]."""
    scores = {"a": 0.0, "b": 5.0, "c": 10.0}
    result = _min_max_normalise(scores)
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(0.5)
    assert result["c"] == pytest.approx(1.0)


def test_min_max_normalise_constant_values() -> None:
    """When all scores are identical every item should receive 1.0."""
    scores = {"x": 3.0, "y": 3.0, "z": 3.0}
    result = _min_max_normalise(scores)
    assert all(v == pytest.approx(1.0) for v in result.values())


def test_min_max_normalise_empty() -> None:
    assert _min_max_normalise({}) == {}


def test_blend_scores_equal_weights() -> None:
    """With equal weights the blended score should be the mean of normalised scores."""
    cb = {"i1": 1.0, "i2": 0.0}
    cf = {"i1": 0.0, "i2": 1.0}
    blended = blend_scores([cb, cf], weights=[1.0, 1.0])
    # Both normalised to [0,1]; blended i1 == 0.5, i2 == 0.5
    assert blended["i1"] == pytest.approx(0.5)
    assert blended["i2"] == pytest.approx(0.5)


def test_blend_scores_asymmetric_weights() -> None:
    """Higher weight on CB should push i1 above i2 when CB prefers i1."""
    cb = {"i1": 1.0, "i2": 0.0}
    cf = {"i1": 0.0, "i2": 1.0}
    blended = blend_scores([cb, cf], weights=[0.8, 0.2])
    assert blended["i1"] > blended["i2"]


def test_blend_scores_item_union() -> None:
    """Items appearing in only one model should still be in the output."""
    cb = {"i1": 0.9, "i2": 0.1}
    cf = {"i2": 0.8, "i3": 0.5}
    blended = blend_scores([cb, cf])
    assert "i1" in blended
    assert "i2" in blended
    assert "i3" in blended


def test_blend_scores_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        blend_scores([])


def test_blend_scores_weight_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="weights length"):
        blend_scores([{"i1": 1.0}], weights=[0.5, 0.5])


def test_blend_scores_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        blend_scores([{"i1": 1.0}], weights=[-1.0])


def test_blend_scores_all_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        blend_scores([{"i1": 1.0}], weights=[0.0])


# ---------------------------------------------------------------------------
# HybridRecommender tests
# ---------------------------------------------------------------------------

def test_hybrid_recommender_returns_k_items() -> None:
    """Hybrid should return exactly k recommendations for known users."""
    model = HybridRecommender(cb_weight=0.5, cf_weight=0.5)
    model.fit(get_mock_train_data(), get_mock_items())
    recs = model.recommend("u1", k=3)
    assert len(recs) == 3


def test_hybrid_recommender_no_duplicates() -> None:
    """Returned list must contain unique item IDs."""
    model = HybridRecommender()
    model.fit(get_mock_train_data(), get_mock_items())
    recs = model.recommend("u1", k=4)
    assert len(recs) == len(set(recs))


def test_hybrid_recommender_excludes_seen_items() -> None:
    """Items in exclude_item_ids must not appear in recommendations."""
    model = HybridRecommender()
    model.fit(get_mock_train_data(), get_mock_items())
    excluded = {"i1", "i2"}
    recs = model.recommend("u1", k=2, exclude_item_ids=excluded)
    assert not set(recs) & excluded


def test_hybrid_recommender_cold_start_fallback() -> None:
    """Unknown users should receive recommendations via the popularity fallback."""
    model = HybridRecommender()
    model.fit(get_mock_train_data(), get_mock_items())
    recs = model.recommend("unknown_user", k=2)
    assert len(recs) == 2
    assert isinstance(recs, list)


def test_hybrid_recommender_invalid_k_raises() -> None:
    model = HybridRecommender()
    model.fit(get_mock_train_data(), get_mock_items())
    with pytest.raises(ValueError, match="k must be positive"):
        model.recommend("u1", k=0)


def test_hybrid_recommender_not_fitted_raises() -> None:
    model = HybridRecommender()
    with pytest.raises(ValueError, match="must be fitted"):
        model.recommend("u1", k=2)


def test_hybrid_recommender_invalid_weights_raises() -> None:
    with pytest.raises(ValueError):
        HybridRecommender(cb_weight=-1.0, cf_weight=0.5)


def test_hybrid_recommender_zero_sum_weights_raises() -> None:
    with pytest.raises(ValueError):
        HybridRecommender(cb_weight=0.0, cf_weight=0.0)


def test_hybrid_cb_only_weight() -> None:
    """Setting cf_weight=0 should still produce valid recommendations."""
    model = HybridRecommender(cb_weight=1.0, cf_weight=0.0)
    model.fit(get_mock_train_data(), get_mock_items())
    recs = model.recommend("u1", k=2)
    assert len(recs) == 2


def test_hybrid_cf_only_weight() -> None:
    """Setting cb_weight=0 should still produce valid recommendations."""
    model = HybridRecommender(cb_weight=0.0, cf_weight=1.0)
    model.fit(get_mock_train_data(), get_mock_items())
    recs = model.recommend("u1", k=2)
    assert len(recs) == 2


# ---------------------------------------------------------------------------
# evaluate_hybrid tests
# ---------------------------------------------------------------------------

def test_evaluate_hybrid_returns_expected_keys() -> None:
    """evaluate_hybrid should return all standard ranking metric keys."""
    model = HybridRecommender()
    model.fit(get_mock_train_data(), get_mock_items())
    metrics = evaluate_hybrid(model, get_mock_train_data(), get_mock_test_data(), k=2)
    assert "evaluated_users" in metrics
    assert "precision_at_2" in metrics
    assert "recall_at_2" in metrics
    assert "map_at_2" in metrics
    assert "ndcg_at_2" in metrics


def test_evaluate_hybrid_metric_range() -> None:
    """All ranking metric values should be in [0, 1]."""
    model = HybridRecommender()
    model.fit(get_mock_train_data(), get_mock_items())
    metrics = evaluate_hybrid(model, get_mock_train_data(), get_mock_test_data(), k=2)
    for key in ("precision_at_2", "recall_at_2", "map_at_2", "ndcg_at_2"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"
