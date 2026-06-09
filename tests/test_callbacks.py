"""Tests for on_alert callback and output distribution monitoring."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from canary_ml import ModelMonitor
from canary_ml.report import DriftReport


rng = np.random.default_rng(99)
X_ref = rng.normal(0, 1, (200, 4))
y_ref = rng.integers(0, 2, 200)


def make_monitor(tmp_path, **kwargs):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_ref, y_ref)
    return ModelMonitor(
        model=clf,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
        **kwargs,
    )


# ── on_alert callback ─────────────────────────────────────────────────────────

def test_on_alert_called_when_alert_triggered(tmp_path):
    fired = []
    monitor = make_monitor(tmp_path, alert_threshold=0.0, on_alert=fired.append)
    # threshold=0.0 means any PSI > 0 triggers; drifted batch should fire
    monitor.predict(rng.normal(5, 1, (50, 4)))
    monitor._flush()
    assert len(fired) >= 1
    assert isinstance(fired[0], DriftReport)


def test_on_alert_not_called_when_stable(tmp_path):
    fired = []
    monitor = make_monitor(tmp_path, alert_threshold=999.0, on_alert=fired.append)
    monitor.predict(rng.normal(0, 1, (50, 4)))
    monitor._flush()
    assert len(fired) == 0


def test_on_alert_receives_correct_report(tmp_path):
    reports = []
    monitor = make_monitor(tmp_path, alert_threshold=0.0, on_alert=reports.append)
    monitor.predict(rng.normal(5, 1, (50, 4)))
    monitor._flush()
    assert reports[0].alert_triggered is True
    assert isinstance(reports[0].psi_score, float)


def test_on_alert_exception_does_not_propagate(tmp_path):
    def bad_callback(r):
        raise RuntimeError("callback failure")

    monitor = make_monitor(tmp_path, alert_threshold=0.0, on_alert=bad_callback)
    # Should not raise — callback errors are swallowed
    result = monitor.predict(rng.normal(5, 1, (50, 4)))
    monitor._flush()
    assert result is not None


def test_no_on_alert_is_fine(tmp_path):
    monitor = make_monitor(tmp_path)
    monitor.predict(rng.normal(0, 1, (50, 4)))
    monitor._flush()


# ── output distribution monitoring ───────────────────────────────────────────

def test_output_ks_present_in_log(tmp_path):
    """output_ks should appear in the log when the model supports predict."""
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_ref, y_ref)
    monitor = ModelMonitor(
        model=clf,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )
    monitor.predict(rng.normal(0, 1, (50, 4)))
    monitor._flush()
    entry = monitor.get_history(1)[-1]
    assert "output_ks" in entry


def test_output_ks_drifted_on_shifted_outputs(tmp_path):
    """A model that always predicts one class vs another should show output drift."""
    class AlwaysZero:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    class AlwaysOne:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    # Reference monitor predicts all 0s; then we swap to all 1s
    monitor = ModelMonitor(
        model=AlwaysZero(),
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )
    # Override model after baseline captured to simulate concept drift
    monitor._model = AlwaysOne()
    monitor.predict(rng.normal(0, 1, (100, 4)))
    monitor._flush()

    report = monitor.get_report()
    assert report is not None
    assert report.output_ks is not None
    assert report.output_ks["drifted"] is True


def test_output_ks_stable_on_clean_data(tmp_path):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_ref, y_ref)
    monitor = ModelMonitor(
        model=clf,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )
    monitor.predict(rng.normal(0, 1, (80, 4)))
    monitor._flush()
    report = monitor.get_report()
    # DummyClassifier always predicts the same class — outputs identical,
    # KS statistic should be ~0
    if report.output_ks is not None:
        assert report.output_ks["statistic"] < 0.2
