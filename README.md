# Personalized Recommendation Engine

A production-style machine learning system that recommends relevant products by combining behavioral interactions with item metadata.

## Problem

Given a user and their historical product interactions, the system ranks products the user is most likely to find relevant. It is designed for implicit and explicit product-feedback signals such as ratings, reviews, clicks, saves, and purchases.

## Project goals

- Compare popularity, content-based, collaborative-filtering, matrix-factorization, and hybrid recommenders.
- Evaluate models with ranking metrics: Precision@K, Recall@K, MAP@K, and NDCG@K.
- Handle cold-start users and items.
- Serve recommendations through a FastAPI application and a simple dashboard.
- Keep training and serving reproducible, testable, and deployment-ready.

## Dataset

We will use the Amazon Reviews 2023 **All Beauty** subset. It provides:

- **Users** — anonymized reviewers.
- **Items** — beauty products identified by parent ASIN.
- **Interactions** — ratings and review timestamps.
- **Metadata** — titles, descriptions, categories, and product attributes where available.

The raw dataset will never be committed. Instructions and provenance will be maintained in the project documentation.

## Day 1 plan

Day 1 establishes the project foundation in six small parts:

1. Project identity and scope (this commit)
2. Repository layout and dependency setup
3. Dataset download and provenance instructions
4. Dataset inspection utilities
5. Cleaning and processed-data contract
6. Exploratory data analysis and Day 1 documentation

We will complete one part at a time.

## Planned system architecture

```text
Interactions + item metadata
            |
            v
Data pipeline --> recommenders --> evaluation --> model artifacts
                                |
                                v
                         FastAPI + dashboard
```

## Evaluation Metrics

Performance comparison of Top-10 recommendations across models:

| Model | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|-------|--------------|-----------|---------|---------|
| Popularity (Day 3) | 0.00207 | 0.02075 | 0.00443 | 0.00816 |
| Content-Based (Day 4) | 0.00090 | 0.00883 | 0.00311 | 0.00445 |
| Collaborative Filtering (Day 5) | 0.00099 | 0.00992 | 0.00323 | 0.00479 |
| **Hybrid (Day 6)** | **0.00125** | **0.01235** | **0.00412** | **0.00612** |

## Serving & Deployment

To run the API and access the dashboard:
1. Ensure your environment has the required dependencies (`pip install fastapi uvicorn`).
2. Start the Uvicorn server from the project root:
   ```bash
   uvicorn recommender.api.app:app --reload
   ```
3. Open your browser and navigate to `http://localhost:8000/dashboard` to interact with the engine.

## Status

**Current milestone:** Day 7 complete — FastAPI serving and web dashboard implemented!
