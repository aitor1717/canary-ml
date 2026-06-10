import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from canary_ml import ModelMonitor


rng = np.random.default_rng(7)
X_ref = rng.normal(0, 1, (200, 5))
y_ref = rng.integers(0, 2, 200)
X_test = rng.normal(0, 1, (40, 5))


def make_monitor(tmp_path):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_ref, y_ref)
    return ModelMonitor(
        model=clf,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )


def test_predict_returns_same_as_direct_model(tmp_path):
    monitor = make_monitor(tmp_path)
    clf = monitor._model
    expected = clf.predict(X_test)
    result = monitor.predict(X_test)
    np.testing.assert_array_equal(result, expected)


def test_get_report_populates_after_predict(tmp_path):
    monitor = make_monitor(tmp_path)
    assert monitor.get_report() is None
    monitor.predict(X_test)
    monitor.wait()
    report = monitor.get_report()
    assert report is not None
    assert report.n_samples == len(X_test)
    assert isinstance(report.psi_score, float)


def test_log_written_to_disk(tmp_path):
    monitor = make_monitor(tmp_path)
    monitor.predict(X_test)
    monitor.wait()
    log_file = tmp_path / "monitor.jsonl"
    assert log_file.exists()
    assert log_file.stat().st_size > 0


def test_predict_never_raises_even_if_internals_fail(tmp_path):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_ref, y_ref)

    monitor = ModelMonitor(
        model=clf,
        reference_data=X_ref,
        log_path=str(tmp_path),
        verbose=False,
    )

    # Corrupt the anomaly detector to force an internal error
    monitor._detector._isoforest = None
    monitor._detector._ref_mean = None
    monitor._detector._ref_std = None

    # Should still return predictions without raising
    result = monitor.predict(X_test)
    monitor.wait()
    assert result is not None
    assert len(result) == len(X_test)


def test_get_history_returns_list(tmp_path):
    monitor = make_monitor(tmp_path)
    monitor.predict(X_test)
    monitor.predict(X_test)
    monitor.wait()
    history = monitor.get_history(n=5)
    assert isinstance(history, list)
    assert len(history) <= 5


def test_reset_baseline(tmp_path):
    monitor = make_monitor(tmp_path)
    new_ref = rng.normal(5, 1, (100, 5))
    monitor.reset_baseline(new_ref)
    assert monitor._reference.shape == (100, 5)
