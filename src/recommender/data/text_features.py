"""Item text vectorization logic using TF-IDF."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def build_tfidf_vectors(
    items: pd.DataFrame, 
    text_column: str = "item_text", 
    max_features: int = 5000
) -> tuple[csr_matrix, TfidfVectorizer]:
    """
    Build TF-IDF vectors for items based on their text metadata.
    
    Args:
        items: DataFrame containing item metadata.
        text_column: The column containing the text to vectorize (e.g., combined title + description).
        max_features: Maximum number of features (vocabulary size) for TF-IDF.
        
    Returns:
        A tuple containing the sparse TF-IDF matrix and the fitted vectorizer.
    """
    if text_column not in items.columns:
        raise ValueError(f"Column '{text_column}' not found in items DataFrame.")
        
    # Handle missing text by filling with empty string
    corpus = items[text_column].fillna("")
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        lowercase=True,
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer
