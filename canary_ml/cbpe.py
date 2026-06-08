from __future__ import annotations

import numpy as np


def estimate_accuracy(probas: np.ndarray) -> float:
    """Estimate classifier accuracy from predicted probabilities (CBPE).

    For each sample the expected probability of a correct prediction equals
    max(p_i) across all classes — you always predict the most confident class,
    and that confidence is the probability of being right.  Averaging over the
    batch gives an unbiased accuracy estimate when probabilities are calibrated.

    Works identically for binary (shape n×2) and multi-class (shape n×k).
    If a 1-D array is passed it is interpreted as P(positive) for binary.
    """
    arr = np.asarray(probas, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"probas must be shape (n, k) with k≥2, got {arr.shape}")
    return float(np.mean(np.max(arr, axis=1)))
