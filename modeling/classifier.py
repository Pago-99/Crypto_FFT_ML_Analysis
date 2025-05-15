"""
Module: modeling.classifier

Defines a classifier pipeline for predicting price direction.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


class PriceDirectionClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that predicts price direction (up/down) based on FFT features.

    Uses a standard scaler and logistic regression by default.
    """
    def __init__(self, C: float = 1.0, random_state: int = 0):
        self.C = C
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, random_state=self.random_state))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        return self.pipeline.score(X, y)