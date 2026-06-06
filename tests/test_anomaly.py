import numpy as np
import pytest

from canary_ml.anomaly import AnomalyDetector


rng = np.random.default_rng(42)


def make_clean(n=300):
    return rng.normal(0, 1, (n, 4))


def test_flags_obvious_outliers():
    X_ref = make_clean()
    detector = AnomalyDetector(contamination=0.05).fit(X_ref)

    outliers = rng.normal(0, 1, (10, 4)) + 10  # 10-sigma away
    result = detector.score(outliers)

    assert result["anomaly_rate"] > 0.5, "Should flag most extreme outliers"
    assert result["anomaly_mask"].sum() > 0


def test_low_rate_on_clean_data():
    X_ref = make_clean()
    detector = AnomalyDetector(contamination=0.05).fit(X_ref)

    X_clean = rng.normal(0, 1, (200, 4))
    result = detector.score(X_clean)

    assert result["anomaly_rate"] < 0.15, "Anomaly rate on in-distribution data should be low"


def test_fit_score_runs_without_error():
    X = make_clean()
    detector = AnomalyDetector().fit(X)
    result = detector.score(X[:20])

    assert "anomaly_rate" in result
    assert "anomaly_mask" in result
    assert "scores" in result
    assert len(result["anomaly_mask"]) == 20


def test_zscore_method():
    X_ref = make_clean()
    detector = AnomalyDetector(method="zscore").fit(X_ref)

    extreme = np.full((5, 4), 100.0)
    result = detector.score(extreme)
    assert result["anomaly_rate"] == 1.0


def test_isolation_forest_method():
    X_ref = make_clean()
    detector = AnomalyDetector(method="isolation_forest").fit(X_ref)
    result = detector.score(X_ref[:10])
    assert "anomaly_rate" in result
