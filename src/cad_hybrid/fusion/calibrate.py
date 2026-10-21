"""Calibrate each branch so its probabilities mean what they say.

Fit on a held-out validation set (never the training data), then apply to new
predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """Platt (sigmoid) or isotonic calibration of raw branch scores."""

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._model = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Calibrator":
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(scores, y)
        else:
            self._model = LogisticRegression()
            self._model.fit(scores.reshape(-1, 1), y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            return self._model.predict(scores)
        return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
