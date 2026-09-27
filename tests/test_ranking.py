import math

from recommender.evaluation.ranking import map_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics_score_relevant_early_results_higher() -> None:
    recommended = ["item-1", "item-2", "item-3"]
    relevant = {"item-1", "item-3"}

    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert math.isclose(map_at_k(recommended, relevant, 3), (1 + 2 / 3) / 2)
    assert math.isclose(ndcg_at_k(recommended, relevant, 3), 0.919720789)
