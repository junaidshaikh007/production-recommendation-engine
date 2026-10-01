"""Utilities to construct sparse interaction matrices for collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recommender.data.encoding import UNKNOWN_INDEX


def create_interaction_matrix(
    interactions: pd.DataFrame,
    num_users: int,
    num_items: int,
    weight_column: str = "rating",
) -> csr_matrix:
    """Convert an encoded interactions DataFrame into a sparse user-item matrix.

    Args:
        interactions: DataFrame containing encoded 'user_idx' and 'item_idx' columns.
        num_users: Total number of unique users (to define matrix shape).
        num_items: Total number of unique items (to define matrix shape).
        weight_column: Column name to use for matrix values. Defaults to 'rating'.
                       If missing, values default to 1.0.

    Returns:
        A SciPy CSR sparse matrix of shape (num_users, num_items).
        Interactions involving unknown users or items are ignored.
    """
    valid_mask = (interactions["user_idx"] != UNKNOWN_INDEX) & (
        interactions["item_idx"] != UNKNOWN_INDEX
    )
    valid_interactions = interactions[valid_mask]

    row_indices = valid_interactions["user_idx"].values
    col_indices = valid_interactions["item_idx"].values

    if weight_column in valid_interactions.columns:
        data = valid_interactions[weight_column].values
    else:
        data = np.ones(len(valid_interactions), dtype=np.float32)

    return csr_matrix((data, (row_indices, col_indices)), shape=(num_users, num_items))
