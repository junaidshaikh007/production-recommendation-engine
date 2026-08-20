# Data validation

Day 2 starts with a strict check of the Day 1 processed-data contract. Run it with:

```powershell
.\.venv\Scripts\python.exe -m recommender.data.validation
```

The command writes `data/processed/validation_report.json` and exits with an error if a required invariant fails.

## Enforced rules

- Interactions must contain user ID, item ID, rating, timestamp, and event time.
- Ratings must be in the inclusive range 1.0–5.0.
- Interaction timestamps must be positive.
- `(user_id, item_id, timestamp_ms)` must be unique.
- Item IDs must be unique.
- Every interaction item must exist in the item dataset.

Items with missing `item_text` are a warning rather than an error: they remain available to collaborative and popularity models but cannot use content-based features.

