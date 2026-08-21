"""Leakage-safe user and item ID encoders for recommendation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

UNKNOWN_INDEX = -1


@dataclass
class IndexEncoder:
    """Map stable string identifiers to contiguous integer indices."""

    index_by_value: dict[str, int] = field(default_factory=dict)

    def fit(self, values: Iterable[str]) -> IndexEncoder:
        """Learn an index from first-seen unique values."""
        self.index_by_value = {}
        for value in values:
            if pd.isna(value):
                raise ValueError("Cannot fit an encoder with null identifiers.")
            normalized_value = str(value)
            if normalized_value not in self.index_by_value:
                self.index_by_value[normalized_value] = len(self.index_by_value)
        return self

    @property
    def size(self) -> int:
        """Return the number of fitted identifiers."""
        return len(self.index_by_value)

    def transform(self, values: Iterable[str]) -> list[int]:
        """Encode values; values unseen during fitting receive UNKNOWN_INDEX."""
        if not self.index_by_value:
            raise ValueError("Encoder has not been fitted.")
        return [self.index_by_value.get(str(value), UNKNOWN_INDEX) for value in values]

    def inverse_transform(self, indices: Iterable[int]) -> list[str | None]:
        """Decode indices; UNKNOWN_INDEX and invalid indices become None."""
        values_by_index = {index: value for value, index in self.index_by_value.items()}
        return [values_by_index.get(index) for index in indices]

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Create a JSON-ready representation of the fitted mapping."""
        return {"index_by_value": self.index_by_value}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> IndexEncoder:
        """Restore a fitted encoder from JSON-ready data."""
        mapping = {key: int(value) for key, value in data["index_by_value"].items()}
        return cls(index_by_value=mapping)


@dataclass
class EncoderBundle:
    """Paired encoders for the two entity types used by recommendation models."""

    user_encoder: IndexEncoder
    item_encoder: IndexEncoder

    def to_dict(self) -> dict[str, dict[str, dict[str, int]]]:
        """Create a JSON-ready representation of both mappings."""
        return {
            "user_encoder": self.user_encoder.to_dict(),
            "item_encoder": self.item_encoder.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, dict[str, int]]]) -> EncoderBundle:
        """Restore an encoder bundle from JSON-ready data."""
        return cls(
            user_encoder=IndexEncoder.from_dict(data["user_encoder"]),
            item_encoder=IndexEncoder.from_dict(data["item_encoder"]),
        )


def fit_encoders(training_interactions: pd.DataFrame) -> EncoderBundle:
    """Fit user and item encoders strictly from a training interaction dataset."""
    required_columns = {"user_id", "item_id"}
    missing_columns = required_columns - set(training_interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Cannot fit encoders; missing columns: {missing}.")
    return EncoderBundle(
        user_encoder=IndexEncoder().fit(training_interactions["user_id"]),
        item_encoder=IndexEncoder().fit(training_interactions["item_id"]),
    )


def encode_interactions(interactions: pd.DataFrame, encoders: EncoderBundle) -> pd.DataFrame:
    """Add encoded user and item indices without discarding original identifiers."""
    encoded = interactions.copy()
    encoded["user_idx"] = pd.Series(
        encoders.user_encoder.transform(encoded["user_id"]), index=encoded.index, dtype="int32"
    )
    encoded["item_idx"] = pd.Series(
        encoders.item_encoder.transform(encoded["item_id"]), index=encoded.index, dtype="int32"
    )
    return encoded
