# Day 3: Popularity baseline recommender

The popularity baseline is the first production-relevant recommendation strategy. It needs no user history, so it works for anonymous and new users while establishing a minimum performance bar for every later model.

## Strategies

- **Popular:** ranks products by positive interactions, then total interactions, from the training period.
- **Trending:** applies the same ranking to the most recent 90 training days.
- **Store-specific:** ranks products within an item store, then falls back to global popularity.

The raw product `categories` field is empty in this dataset, so a fake category baseline would be misleading. Store-specific popularity provides meaningful metadata segmentation without inventing data.

## Evaluation

The baseline is trained on `train.parquet` and evaluated against future positive interactions in `test.parquet`. Previously seen training items are excluded for users who have prior history.

```powershell
.\.venv\Scripts\python.exe -m recommender.models.popularity
```

The command writes Git-ignored metrics to `artifacts/baselines/popularity_metrics.json` and reports Precision@10, Recall@10, MAP@10, NDCG@10, and catalog coverage.

