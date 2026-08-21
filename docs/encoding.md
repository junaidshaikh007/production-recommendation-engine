# User and item encoding

Recommendation algorithms operate on compact numerical indices rather than long Amazon identifiers. `IndexEncoder` maps each string ID to a contiguous integer, while retaining original IDs in the transformed data.

## Leakage rule

Encoders are fitted **only on the training split**. Validation and test identifiers unseen in training receive the sentinel value `-1` (`UNKNOWN_INDEX`). This preserves a realistic cold-start boundary instead of allowing future identities to influence model dimensions or learned embeddings.

## Why not remove unknown entities?

Unknown entities are retained and explicitly marked. Later recommenders can choose an appropriate fallback, such as a content-based or popularity strategy, instead of silently dropping their evaluation examples.

