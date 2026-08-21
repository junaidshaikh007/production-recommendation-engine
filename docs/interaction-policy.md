# Interaction-preprocessing policy

The raw source records explicit product ratings from 1 to 5. Most recommendation models in this project will rank a top-K list, so they need a binary definition of relevance.

## Default relevance label

```text
is_positive = 1 when rating >= 4.0
is_positive = 0 when rating < 4.0
```

The threshold is configurable through `InteractionPolicy`. It is deliberately applied to the event’s own rating only; it does not use a user’s future behavior, item popularity, or future timestamps.

## Why retain all events?

Low ratings are retained in the labeled dataset. They are useful for honest analysis, future explicit-rating experiments, and avoiding an artificial data distribution. Later top-K evaluation will treat only held-out positive events as relevant.

## Output

`interactions_labeled.parquet` contains the original validated columns plus:

| Column | Meaning |
| --- | --- |
| `is_positive` | Binary relevance label |
| `event_date` | UTC calendar date derived from the original event timestamp |

