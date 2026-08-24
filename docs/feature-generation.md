# Feature generation

Day 2 produces three feature tables using only permitted information:

| Feature table | Source | Leakage protection |
| --- | --- | --- |
| `item_features.parquet` | Item title, description, feature text, and price availability | Does not use interaction outcomes |
| `user_training_features.parquet` | Training interactions | Derived only from the train period |
| `item_training_features.parquet` | Training interactions | Derived only from the train period |

Run feature generation with:

```powershell
.\.venv\Scripts\python.exe -m recommender.data.features
```

The history tables are **not** joined onto their own training events as model labels. They are profile summaries for later candidate ranking and cold-start strategy selection. When used for validation or test ranking, they represent information that was available before those periods.

The project intentionally does not use product `average_rating` or `rating_number` as a training feature. Those values are scraped product-level aggregates and may include future feedback relative to a historical interaction.

