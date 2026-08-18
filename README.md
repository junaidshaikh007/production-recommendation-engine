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

## Status

**Current milestone:** Day 1, Part 2 — repository foundation complete.
