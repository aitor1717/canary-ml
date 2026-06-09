import tempfile

import numpy as np
import pytest

from canary_ml import ModelMonitor
from canary_ml.cbpe import estimate_accuracy


# ── Unit tests for estimate_accuracy ─────────────────────────────────────────

def test_binary_perfect_confidence():
    # All predictions with 100% confidence → estimated accuracy = 1.0
    probas = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert estimate_accuracy(probas) == pytest.approx(1.0)


def test_binary_random_confidence():
    # Known values: max(0.9,0.1)=0.9, max(0.4,0.6)=0.6, max(0.7,0.3)=0.7 → mean=0.733...
    probas = np.array([[0.1, 0.9], [0.4, 0.6], [0.3, 0.7]])
    expected = (0.9 + 0.6 + 0.7) / 3
    assert estimate_accuracy(probas) == pytest.approx(expected)


def test_multiclass():
    # 3-class: max of each row
    probas = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    expected = (0.8 + 0.7 + 0.4) / 3
    assert estimate_accuracy(probas) == pytest.approx(expected)


def test_1d_binary_input():
    # 1-D array treated as P(positive)
    probas = np.array([0.9, 0.6, 0.7])
    expected = (0.9 + 0.6 + 0.7) / 3
    assert estimate_accuracy(probas) == pytest.approx(expected)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        estimate_accuracy(np.array([[0.5]]))  # single column


# ── Integration tests via ModelMonitor ───────────────────────────────────────

rng = np.random.default_rng(42)
X_ref = rng.normal(0, 1, (300, 4))
y_ref = rng.integers(0, 2, 300)
X_test = rng.normal(0, 1, (80, 4))


class _ProbaModel:
    """Minimal model with predict_proba for testing."""
    def fit(self, X, y): return self
    def predict(self, X):
        return (X[:, 0] > 0).astype(int)
    def predict_proba(self, X):
        p = np.clip(0.5 + X[:, 0] * 0.4, 0.05, 0.95)
        return np.column_stack([1 - p, p])


class _NoProbaModel:
    """Model without predict_proba."""
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def make_proba_monitor(tmp_path):
    model = _ProbaModel()
    return ModelMonitor(
        model=model,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )


def test_cbpe_enabled_when_predict_proba_available(tmp_path):
    monitor = make_proba_monitor(tmp_path)
    assert monitor._cbpe_enabled is True
    assert monitor._reference_estimated_accuracy is not None
    assert 0.5 < monitor._reference_estimated_accuracy <= 1.0


def test_cbpe_disabled_without_predict_proba(tmp_path):
    monitor = ModelMonitor(
        model=_NoProbaModel(),
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )
    assert monitor._cbpe_enabled is False
    assert monitor._reference_estimated_accuracy is None


def test_report_has_estimated_accuracy_after_predict(tmp_path):
    monitor = make_proba_monitor(tmp_path)
    monitor.predict(X_test)
    monitor._flush()
    report = monitor.get_report()
    assert report.estimated_accuracy is not None
    assert 0.5 < report.estimated_accuracy <= 1.0
    assert report.reference_accuracy == pytest.approx(monitor._reference_estimated_accuracy)
    assert report.performance_delta == pytest.approx(
        report.estimated_accuracy - report.reference_accuracy
    )


def test_report_cbpe_fields_none_without_predict_proba(tmp_path):
    monitor = ModelMonitor(
        model=_NoProbaModel(),
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )
    monitor.predict(X_test)
    monitor._flush()
    report = monitor.get_report()
    assert report.estimated_accuracy is None
    assert report.performance_delta is None
    assert report.performance_alert is False


def test_performance_alert_fires_on_large_drop(tmp_path):
    model = _ProbaModel()
    monitor = ModelMonitor(
        model=model,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
        performance_threshold=0.0,  # any drop triggers alert
    )
    # Inject reference accuracy higher than any real estimate
    monitor._reference_estimated_accuracy = 1.0
    monitor.predict(X_test)
    monitor._flush()
    report = monitor.get_report()
    assert report.performance_alert is True
    assert report.alert_triggered is True


def test_performance_alert_does_not_fire_within_threshold(tmp_path):
    monitor = make_proba_monitor(tmp_path)
    # Set a very large threshold so no alert fires
    monitor._performance_threshold = 1.0
    monitor.predict(X_test)
    monitor._flush()
    report = monitor.get_report()
    assert report.performance_alert is False


def test_callback_fires_on_performance_alert(tmp_path):
    fired = []
    model = _ProbaModel()
    monitor = ModelMonitor(
        model=model,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
        performance_threshold=0.0,
        on_alert=lambda r: fired.append(r),
    )
    monitor._reference_estimated_accuracy = 1.0
    monitor.predict(X_test)
    monitor._flush()
    assert len(fired) == 1
    assert fired[0].performance_alert is True
