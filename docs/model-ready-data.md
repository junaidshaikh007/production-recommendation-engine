# Model-ready recommendation data

The model-ready pipeline fits user and item encoders only on `train.parquet`, then applies those exact mappings to validation and test data.

```powershell
.\.venv\Scripts\python.exe -m recommender.data.model_ready
```

It writes Git-ignored artifacts:

- `artifacts/encoders.json` — user and item ID mappings for reproducible training and serving.
- `data/processed/splits/encoded/*.parquet` — original split data plus `user_idx` and `item_idx`.
- `data/processed/splits/encoded/encoding_summary.json` — unknown-entity rates.

The value `-1` represents an entity that was not available in training. Matrix-factorization and neural models will exclude this value from their embedding lookup; the later cold-start strategy will handle it with non-personalized or content signals.

