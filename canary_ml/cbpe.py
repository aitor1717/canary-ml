from __future__ import annotations

import numpy as np


def estimate_accuracy(probas: np.ndarray) -> float:
    """Estimate classifier accuracy as mean max-class probability.

    This equals true accuracy only when the model's predicted probabilities are
    well-calibrated (i.e. a prediction of 0.9 is correct ~90% of the time).
    Most classifiers are overconfident — if your model hasn't been through
    calibration (e.g. sklearn's CalibratedClassifierCV), this estimate will
    overstate true accuracy. Use it as a directional signal, not a precise number.

    Works identically for binary (shape n×2) and multi-class (shape n×k).
    If a 1-D array is passed it is interpreted as P(positive) for binary.
    """
    arr = np.asarray(probas, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"probas must be shape (n, k) with k≥2, got {arr.shape}")
    return float(np.mean(np.max(arr, axis=1)))
