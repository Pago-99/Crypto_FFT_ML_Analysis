"""
Module: modeling.dimensionality_reduction

Defines dimensionality reduction utilities (PCA, t-SNE).
"""
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional


def pca_reduce(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce feature dimensionality via PCA.

    Returns an array of shape (n_samples, n_components).
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def tsne_reduce(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: Optional[int] = None,
    random_state: int = 0
) -> np.ndarray:
    """
    Reduce feature dimensionality via t-SNE, explicitly setting init and learning_rate
    to avoid FutureWarnings.
    """
    from sklearn.manifold import TSNE

    n_samples = X.shape[0]
    if perplexity is None:
        perp = min(30, n_samples - 1)
    else:
        perp = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        init='pca',            # future default
        learning_rate='auto',  # future default
        random_state=random_state
    )
    return tsne.fit_transform(X)