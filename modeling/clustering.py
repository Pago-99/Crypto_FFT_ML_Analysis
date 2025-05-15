"""
Module: modeling.clustering

Defines clustering methods for FFT features.
"""
from sklearn.cluster import KMeans, DBSCAN
import numpy as np


def kmeans_cluster(X: np.ndarray, n_clusters: int = 3, random_state: int = 0) -> np.ndarray:
    """
    Apply KMeans clustering and return cluster labels.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)
    return labels


def dbscan_cluster(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Apply DBSCAN clustering and return cluster labels (-1 for noise).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels