# Day 1: Planning and Dataset Foundation

## Recommendation objective

For each user, produce a ranked top-K list of unseen items that maximizes the likelihood of a relevant future interaction.

## Definitions

| Concept | Meaning in this project |
| --- | --- |
| User | An Amazon customer/reviewer, identified by an anonymized user ID. |
| Item | A beauty product, identified by its parent ASIN. |
| Interaction | A timestamped rating/review event between one user and one item. |
| Relevance | Initially, a held-out positive interaction; the exact threshold will be documented with the preprocessing pipeline. |
| Candidate | A product considered for recommendation before final ranking. |

## Metrics

We will evaluate ranked top-K lists using the following metrics:

| Metric | Answers |
| --- | --- |
| Precision@K | What fraction of the K recommendations are relevant? |
| Recall@K | How many of a user's relevant held-out items were retrieved? |
| MAP@K | Are relevant recommendations consistently ranked early? |
| NDCG@K | How good is the ranking while discounting lower positions? |

## Why this dataset

Amazon Reviews 2023 All Beauty contains both interaction events and product metadata. This makes it suitable for an honest comparison between behavioral recommendation methods and metadata-driven methods, including cold-start item recommendations.

## Scope boundary

Day 1 creates the project foundation, documents the dataset and evaluation goal, and prepares for data acquisition. Model implementation begins only after the raw data and its quality have been examined.

