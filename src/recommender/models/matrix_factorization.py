"""Base matrix factorization using Singular Value Decomposition (SVD)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class MatrixFactorizationModel:
    """A simple Matrix Factorization model based on truncated SVD."""

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, interaction_matrix: csr_matrix) -> MatrixFactorizationModel:
        """Factorize the sparse interaction matrix into user and item embeddings.

        Args:
            interaction_matrix: Sparse matrix of shape (n_users, n_items).
        """
        # Ensure k is smaller than the smallest dimension
        k = min(self.n_factors, min(interaction_matrix.shape) - 1)
        if k <= 0:
            raise ValueError("Matrix dimensions are too small for factorization.")

        # Compute truncated SVD
        u, s, vt = svds(interaction_matrix.astype(float), k=k)

        # Incorporate singular values into user and item factors
        s_sqrt = np.diag(np.sqrt(s))
        self.user_factors = np.dot(u, s_sqrt)
        self.item_factors = np.dot(s_sqrt, vt).T  # Transpose to shape (n_items, k)

        return self

    def predict(self, user_idx: int) -> np.ndarray:
        """Predict scores for all items for a given user index.

        Args:
            user_idx: Internal integer index of the user.

        Returns:
            1D array of predicted scores for all items.
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model is not fitted yet.")

        if user_idx < 0 or user_idx >= len(self.user_factors):
            raise IndexError("User index out of bounds.")

        user_vector = self.user_factors[user_idx]
        return np.dot(self.item_factors, user_vector)
