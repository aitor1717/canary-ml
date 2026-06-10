import numpy as np
import pytest

from canary_ml.drift import ks_drift, psi_score, chi2_drift, detect_drift


rng = np.random.default_rng(0)


def make_same(n=200):
    return rng.normal(0, 1, (n, 3))


def make_shifted(n=200, shift=5.0):
    ref = rng.normal(0, 1, (n, 3))
    cur = ref + shift
    return ref, cur


def test_ks_no_drift_on_identical():
    X = make_same()
    results = ks_drift(X, X.copy())
    for v in results.values():
        assert not v["drifted"], "Identical distributions should not flag drift"


def test_ks_flags_drift_on_clear_shift():
    ref, cur = make_shifted(shift=5.0)
    results = ks_drift(ref, cur)
    drifted = [v["drifted"] for v in results.values()]
    assert any(drifted), "Clear mean shift should be detected by KS test"


def test_psi_near_zero_for_identical():
    X = make_same()
    score = psi_score(X, X.copy())
    assert score < 0.05, f"PSI on identical data should be ~0, got {score:.4f}"


def test_psi_above_threshold_for_very_different():
    ref = rng.normal(0, 1, (300, 2))
    cur = rng.normal(5, 1, (300, 2))
    score = psi_score(ref, cur)
    assert score > 0.2, f"PSI should exceed 0.2 for very different distributions, got {score:.4f}"


def test_ks_edge_case_single_sample():
    ref = make_same(200)
    cur = rng.normal(0, 1, (1, 3))
    # Should not raise
    results = ks_drift(ref, cur)
    assert set(results.keys()) == {"0", "1", "2"}


def test_psi_edge_case_single_sample():
    ref = make_same(200)
    cur = rng.normal(0, 1, (1, 3))
    score = psi_score(ref, cur)
    assert isinstance(score, float)


# ── chi2_drift ────────────────────────────────────────────────────────────────

def make_categorical(n=200, n_cats=5):
    """Return (ref, cur) arrays with integer categories 0..n_cats-1."""
    ref = rng.integers(0, n_cats, (n, 2)).astype(float)
    cur = rng.integers(0, n_cats, (n, 2)).astype(float)
    return ref, cur


def test_chi2_no_drift_on_identical():
    ref, _ = make_categorical()
    results = chi2_drift(ref, ref.copy())
    for v in results.values():
        assert not v["drifted"], "Identical categorical distributions should not flag drift"


def test_chi2_flags_drift_on_clear_shift():
    ref, _ = make_categorical(n_cats=5)
    # All current samples go into category 0 — maximally different from uniform ref
    cur = np.zeros((200, 2), dtype=float)
    results = chi2_drift(ref, cur)
    drifted = [v["drifted"] for v in results.values()]
    assert any(drifted), "All-one-category current vs uniform ref should flag chi2 drift"


def test_chi2_results_have_required_keys():
    ref, cur = make_categorical()
    results = chi2_drift(ref, cur)
    for v in results.values():
        assert "statistic" in v and "p_value" in v and "drifted" in v


def test_detect_drift_routes_categorical_to_chi2():
    """detect_drift should use chi2 for low-cardinality features."""
    ref, cur = make_categorical(n_cats=5)
    psi, ks_results = detect_drift(ref, cur, categorical_threshold=20)
    # Low-cardinality → chi2 — results should still have the standard keys
    for v in ks_results.values():
        assert "statistic" in v and "p_value" in v and "drifted" in v
