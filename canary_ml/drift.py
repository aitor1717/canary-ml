from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Any


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict[str, dict[str, Any]]:
    """KS test per feature. Flags drift when p_value < 0.05."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)

    n_features = ref.shape[1]
    results: dict[str, dict[str, Any]] = {}
    for i in range(n_features):
        stat, p = stats.ks_2samp(ref[:, i], cur[:, i])
        results[str(i)] = {
            "statistic": float(stat),
            "p_value": float(p),
            "drifted": bool(p < 0.05),
        }
    return results


def psi_score(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index across all features (mean PSI).

    < 0.1 stable · 0.1–0.2 moderate · > 0.2 significant shift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)

    n_features = ref.shape[1]
    psi_values: list[float] = []

    # Fewer bins for small batches: keep ~10 samples per bin on average
    effective_bins = min(bins, max(2, len(cur) // 10))

    for i in range(n_features):
        r = ref[:, i]
        c = cur[:, i]

        # Quantile-based bins on reference — equal-count partitioning, handles skew
        breaks = np.unique(np.percentile(r, np.linspace(0, 100, effective_bins + 1)))
        if len(breaks) < 3:
            psi_values.append(0.0)   # constant feature, no drift measurable
            continue
        # Extend to cover any current values outside the reference range
        breaks[0]  = min(breaks[0],  c.min()) - 1e-9
        breaks[-1] = max(breaks[-1], c.max()) + 1e-9

        ref_counts, _ = np.histogram(r, bins=breaks)
        cur_counts, _ = np.histogram(c, bins=breaks)

        ref_pct = ref_counts / max(len(r), 1)
        cur_pct = cur_counts / max(len(c), 1)

        # Replace zeros to avoid log(0) and division by zero
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        psi_values.append(psi)

    return float(np.mean(psi_values)) if psi_values else 0.0


def chi2_drift(reference: np.ndarray, current: np.ndarray) -> dict[str, dict[str, Any]]:
    """Chi-square test per categorical feature."""
    ref = np.asarray(reference)
    cur = np.asarray(current)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)

    n_features = ref.shape[1]
    results: dict[str, dict[str, Any]] = {}
    for i in range(n_features):
        categories = np.union1d(np.unique(ref[:, i]), np.unique(cur[:, i]))
        ref_counts = np.array([np.sum(ref[:, i] == c) for c in categories])
        cur_counts = np.array([np.sum(cur[:, i] == c) for c in categories])

        contingency = np.vstack([ref_counts, cur_counts])
        try:
            chi2, p, *_ = stats.chi2_contingency(contingency)
        except ValueError:
            chi2, p = 0.0, 1.0

        results[str(i)] = {
            "statistic": float(chi2),
            "p_value": float(p),
            "drifted": bool(p < 0.05),
        }
    return results


def is_categorical(ref_col: np.ndarray, cur_col: np.ndarray | None = None, threshold: int = 20) -> bool:
    """Heuristic: treat a column as categorical if it has few unique values across both splits."""
    uniq = np.unique(ref_col)
    if cur_col is not None:
        uniq = np.union1d(uniq, np.unique(cur_col))
    return len(uniq) <= threshold


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    categorical_threshold: int = 20,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Compute PSI + per-feature KS/chi2, routing by column type.

    Returns (psi, ks_results).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)

    psi = psi_score(ref, cur, bins=bins)

    # Route each column to the right test
    ks_cols_ref, ks_cols_cur = [], []
    chi2_cols_ref, chi2_cols_cur = [], []
    col_map: dict[str, str] = {}  # col_idx -> 'ks' | 'chi2'

    for i in range(ref.shape[1]):
        if is_categorical(ref[:, i], cur[:, i], threshold=categorical_threshold):
            chi2_cols_ref.append(ref[:, i])
            chi2_cols_cur.append(cur[:, i])
            col_map[str(i)] = "chi2"
        else:
            ks_cols_ref.append(ref[:, i])
            ks_cols_cur.append(cur[:, i])
            col_map[str(i)] = "ks"

    results: dict[str, dict[str, Any]] = {}

    if ks_cols_ref:
        r = np.column_stack(ks_cols_ref)
        c = np.column_stack(ks_cols_cur)
        ks_r = ks_drift(r, c)
        idx = [k for k, v in col_map.items() if v == "ks"]
        for local_i, global_i in enumerate(idx):
            results[global_i] = ks_r[str(local_i)]

    if chi2_cols_ref:
        r = np.column_stack(chi2_cols_ref)
        c = np.column_stack(chi2_cols_cur)
        chi_r = chi2_drift(r, c)
        idx = [k for k, v in col_map.items() if v == "chi2"]
        for local_i, global_i in enumerate(idx):
            results[global_i] = chi_r[str(local_i)]

    return psi, results
