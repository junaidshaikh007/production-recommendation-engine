# Processed-data contract

`python -m recommender.data.preprocess` produces two Git-ignored Parquet files in `data/processed/`.

## `interactions.parquet`

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | string | Anonymized reviewer identifier |
| `item_id` | string | Parent product identifier (`parent_asin`) |
| `rating` | float32 | Explicit rating from 1.0 to 5.0 |
| `timestamp_ms` | int64 | Original Unix event timestamp in milliseconds |
| `event_time` | UTC datetime | Timestamp converted for chronological analysis |
| `verified_purchase` | boolean | Whether Amazon marked the purchase as verified |
| `helpful_votes` | int32 | Helpful-vote count on the review |

Rows missing a required identifier, rating, or timestamp are removed. Duplicate events are defined by the same `(user_id, item_id, timestamp_ms)` and only the first occurrence is retained. Rows are sorted chronologically within each user.

## `items.parquet`

| Column | Type | Meaning |
| --- | --- | --- |
| `item_id` | string | Parent product identifier |
| `title`, `description`, `features`, `store` | string | Cleaned metadata fields |
| `price`, `average_rating`, `rating_number` | numeric | Product-level metadata |
| `item_text` | string | Concatenation of available textual metadata |

Only items observed in the interaction data are retained. `item_text` is the content representation used by the Day 4 content-based recommender and for new-item cold start.

## Leakage policy

The files preserve original timestamps and do not create train/test labels. Day 2 will apply chronological splits, ensuring a model never trains on interactions that occur after its validation or test target.

