"""
Module: modeling.regressor

Defines a regressor pipeline for forecasting price changes.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np


class PriceChangeRegressor(BaseEstimator, RegressorMixin):
    """
    A regressor that forecasts the next price change (delta) based on FFT features.

    Uses a standard scaler and linear regression by default.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        return self.pipeline.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R^2 score."""
        return self.pipeline.score(X, y)