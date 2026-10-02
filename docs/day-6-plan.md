# Day 6: Hybrid Recommender

## Objective

Combine the content-based and collaborative filtering models built in Days 4 and 5 into a unified hybrid recommender. The hybrid model will blend signals from both approaches to improve ranking quality, handle cold-start scenarios more robustly, and establish the strongest single model before serving.

## 5-Step Implementation Plan

1. **Day 6 Planning and Documentation** (this file)
2. **Hybrid Scoring Strategy**: Design and implement a weighted score blending utility that merges ranked candidate lists from the content-based and collaborative filtering models into a single unified score per item.
3. **HybridRecommender Class**: Build the `HybridRecommender` class in the `recommender.models` package implementing the standard model interface, with configurable weights for each sub-model and automatic cold-start fallback routing.
4. **Unit Testing**: Write unit tests for the hybrid model, weight blending logic, and cold-start fallback paths (`tests/test_hybrid.py`).
5. **Evaluation and Metrics**: Evaluate the hybrid model against the test set, compare Precision@K, Recall@K, MAP@K, and NDCG@K against Day 4 (content-based) and Day 5 (collaborative filtering) baselines, and update the README with Day 6 status.
