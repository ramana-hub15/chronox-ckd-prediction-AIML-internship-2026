"""Custom scikit-learn preprocessing transformers for ChronoX clinical inputs."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES: list[str] = [
    "age",
    "bp",
    "sg",
    "al",
    "bgr",
    "bu",
    "sc",
    "hemo",
    "htn",
]

CONTINUOUS_FEATURES: list[str] = ["age", "bp", "bgr", "bu", "sc", "hemo"]
ORDINAL_FEATURES: list[str] = ["sg", "al"]
BINARY_FEATURES: list[str] = ["htn"]


class ClinicalOutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorize continuous clinical features using training quantiles.

    Learns lower and upper quantile bounds during ``fit`` so test/validation
    data is clipped without data leakage.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ClinicalOutlierClipper":
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float).copy()
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)
