# Dataset: Amazon Reviews 2023 — All Beauty

## Source and attribution

This project uses the **All Beauty** subset of Amazon Reviews 2023, published by the UC San Diego McAuley Lab. The dataset documentation reports approximately 632,000 users, 112,600 items, and 701,500 reviews in this category.

- Dataset documentation: <https://amazon-reviews-2023.github.io/main.html>
- Reviews: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz>
- Product metadata: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz>
- Citation: Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*.

## Files

| Local filename | Source content | Role in this project |
| --- | --- | --- |
| `All_Beauty.jsonl.gz` | Reviews, ratings, timestamps, and reviewer IDs | User-item interaction data |
| `meta_All_Beauty.jsonl.gz` | Titles, descriptions, features, prices, and categories | Content features and cold-start item support |

The shared key is `parent_asin`; it represents the parent product and correctly connects review events to product metadata.

## Raw-data policy

Raw files belong in `data/raw/` and are intentionally excluded from version control because they are large third-party data. The download script is idempotent: if a valid file exists locally, it reuses it. It also checks each downloaded gzip file before continuing.

## Reproduction

The project downloads both files automatically during Day 1 setup. To repeat this manually if ever needed:

```powershell
.\.venv\Scripts\python.exe -m recommender.data.download
```

## Intended use and limitations

This is a public research dataset with anonymized reviewer IDs. Product metadata may be incomplete, and ratings are not equivalent to real-time commercial behavior. We will document filtering decisions and split data chronologically to avoid future interactions leaking into training.

