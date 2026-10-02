"""Similarity utilities for content-based recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profiles(
    train_interactions: pd.DataFrame,
    item_tfidf_matrix: csr_matrix,
    item_id_to_index: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Build user profiles by averaging the TF-IDF vectors of items they interacted with positively.

    Args:
        train_interactions: DataFrame containing training interactions (user_id, item_id, is_positive).
        item_tfidf_matrix: Sparse matrix of item TF-IDF vectors.
        item_id_to_index: Mapping from item_id to the row index in item_tfidf_matrix.

    Returns:
        A dictionary mapping user_id to their dense profile vector (1D numpy array).
    """
    # Filter only positive interactions
    positive_interactions = train_interactions[train_interactions["is_positive"] == 1]
    
    user_profiles = {}
    
    # Group by user
    for user_id, group in positive_interactions.groupby("user_id"):
        item_ids = group["item_id"].tolist()
        
        # Find valid indices for items that exist in our TF-IDF matrix
        valid_indices = [
            item_id_to_index[i_id] 
            for i_id in item_ids 
            if i_id in item_id_to_index
        ]
        
        if valid_indices:
            # Get the submatrix of TF-IDF vectors for this user's positive items
            user_items_matrix = item_tfidf_matrix[valid_indices]
            # Convert to dense numpy array before averaging to bypass scipy's
            # internal sparse matmul path which is broken on Python 3.14+.
            # The submatrix is small (few items × vocab), so this is safe.
            user_profile = user_items_matrix.toarray().mean(axis=0)
            user_profiles[user_id] = user_profile
            
    return user_profiles


def compute_recommendation_scores(
    user_profile: np.ndarray,
    item_tfidf_matrix: csr_matrix,
) -> np.ndarray:
    """
    Compute cosine similarity between a user profile and all item profiles.

    Args:
        user_profile: Dense 1D array representing the user's preferences.
        item_tfidf_matrix: Sparse matrix of item TF-IDF vectors.

    Returns:
        1D numpy array of similarity scores for each item in the matrix.
    """
    # Reshape user_profile to 2D for cosine_similarity
    user_profile_2d = user_profile.reshape(1, -1)
    
    # Compute similarity (returns a 1 x N array)
    similarities = cosine_similarity(user_profile_2d, item_tfidf_matrix)
    
    # Return as 1D array
    return similarities.flatten()
