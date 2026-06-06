from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Ensemble anomaly detector: IsolationForest OR z-score (|z| > 3)."""

    def __init__(
        self,
        contamination: float = 0.05,
        method: str = "ensemble",
    ) -> None:
        """
        Args:
            contamination: Expected fraction of outliers (passed to IsolationForest).
            method: 'isolation_forest', 'zscore', or 'ensemble'.
        """
        self.contamination = contamination
        self.method = method
        self._isoforest: IsolationForest | None = None
        self._ref_mean: np.ndarray | None = None
        self._ref_std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit on reference (baseline) data."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._ref_mean = X.mean(axis=0)
        self._ref_std = X.std(axis=0)
        # Avoid division by zero in score()
        self._ref_std = np.where(self._ref_std == 0, 1.0, self._ref_std)

        if self.method in ("isolation_forest", "ensemble"):
            self._isoforest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
            )
            self._isoforest.fit(X)

        return self

    def score(self, X: np.ndarray) -> dict:
        """Detect anomalies in *X*.

        Returns:
            anomaly_rate: fraction of samples flagged
            anomaly_mask: bool array, True = anomalous
            scores: raw scores (lower = more anomalous for IsolationForest)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = len(X)
        mask = np.zeros(n, dtype=bool)
        raw_scores = np.zeros(n, dtype=float)

        if self.method in ("isolation_forest", "ensemble") and self._isoforest is not None:
            preds = self._isoforest.predict(X)            # -1 = anomaly
            iso_mask = preds == -1
            raw_scores = self._isoforest.decision_function(X)
            mask |= iso_mask

        if self.method in ("zscore", "ensemble") and self._ref_mean is not None:
            z = np.abs((X - self._ref_mean) / self._ref_std)
            zscore_mask = z.max(axis=1) > 3.0
            mask |= zscore_mask

        return {
            "anomaly_rate": float(mask.sum() / max(n, 1)),
            "anomaly_mask": mask,
            "scores": raw_scores,
        }
