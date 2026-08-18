from recommender.data.preprocess import clean_interactions, clean_items


def test_clean_interactions_removes_exact_event_duplicates() -> None:
    records = [
        {
            "user_id": "user-1",
            "parent_asin": "item-1",
            "rating": 5.0,
            "timestamp": 2_000,
            "verified_purchase": True,
            "helpful_vote": 2,
        },
        {
            "user_id": "user-1",
            "parent_asin": "item-1",
            "rating": 5.0,
            "timestamp": 2_000,
            "verified_purchase": True,
            "helpful_vote": 2,
        },
        {
            "user_id": "user-1",
            "parent_asin": "item-2",
            "rating": 4.0,
            "timestamp": 1_000,
            "verified_purchase": False,
        },
    ]

    cleaned = clean_interactions(records)

    assert len(cleaned) == 2
    assert cleaned["item_id"].tolist() == ["item-2", "item-1"]
    assert cleaned["helpful_votes"].tolist() == [0, 2]


def test_clean_items_creates_normalized_content_text() -> None:
    records = [
        {
            "parent_asin": "item-1",
            "title": "  Bright  Serum ",
            "description": ["Hydrates ", " skin"],
            "features": ["Vitamin C"],
            "store": "Example Store",
            "price": 12.5,
            "average_rating": 4.8,
            "rating_number": 10,
        },
        {"parent_asin": "unused", "title": "Do not retain"},
    ]

    cleaned = clean_items(records, {"item-1"})

    assert cleaned["item_id"].tolist() == ["item-1"]
    assert cleaned.loc[0, "item_text"] == "Bright Serum Hydrates skin Vitamin C Example Store"
