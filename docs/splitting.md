# Chronological data splits

The pipeline uses a global time-based 80/10/10 split:

```text
earliest 80% of interactions  -> train
next 10%                     -> validation
latest 10%                   -> test
```

This mirrors a deployed recommender: it learns from history and predicts future behavior. Random splits would let a model see later interactions during training, which produces overly optimistic offline results.

The split summary records how many evaluation users and items were unseen in training. These are deliberately retained as cold-start cases. Personalized models will evaluate their known-entity subset separately, while the hybrid system will later use fallback strategies for new users and items.

Generated files are stored under `data/processed/splits/`:

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `split_summary.json`

