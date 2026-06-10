"""Integration tests: spin up the real HTTP server and hit its endpoints."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from canary_ml import ModelMonitor
from canary_ml.server import start as start_server

PORT = 18501  # non-standard port to avoid clashing with a running dev server


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Start a server with two logged batches; yield the base URL."""
    log = tmp_path_factory.mktemp("srv_logs")

    rng = np.random.default_rng(0)
    X_ref = rng.normal(0, 1, (100, 4))
    y_ref = rng.integers(0, 2, 100)
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_ref, y_ref)

    monitor = ModelMonitor(
        model=clf,
        reference_data=X_ref,
        log_path=str(log),
        verbose=False,
    )
    monitor.predict(rng.normal(0, 1, (30, 4)))
    monitor.predict(rng.normal(2, 1, (30, 4)))  # drifted batch
    monitor.wait()  # ensure both batches are written before server starts

    start_server(log, port=PORT)
    # Brief wait for the daemon thread to bind
    time.sleep(0.15)
    yield f"http://localhost:{PORT}"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def test_dashboard_html_served(live_server):
    with urllib.request.urlopen(live_server + "/", timeout=5) as r:
        body = r.read().decode()
    assert "canary" in body.lower()
    assert r.status == 200


def test_api_data_returns_list(live_server):
    data = _get(live_server + "/api/data?n=10")
    assert isinstance(data, list)
    assert len(data) == 2  # we logged exactly 2 batches


def test_api_data_entry_has_required_keys(live_server):
    data = _get(live_server + "/api/data?n=10")
    entry = data[-1]
    for key in ("timestamp", "psi_score", "ks_results", "anomaly_rate",
                "drift_detected", "alert_triggered", "n_samples"):
        assert key in entry, f"missing key: {key}"


def test_api_data_n_param_limits_results(live_server):
    data = _get(live_server + "/api/data?n=1")
    assert len(data) == 1


def test_api_reference_returns_list(live_server):
    ref = _get(live_server + "/api/reference")
    assert isinstance(ref, list)
    assert len(ref) > 0


def test_api_reference_rows_match_feature_count(live_server):
    ref = _get(live_server + "/api/reference")
    data = _get(live_server + "/api/data?n=1")
    # Each reference row should have same feature count as feature_sample rows
    sample = data[-1].get("feature_sample", [[]])
    if sample:
        assert len(ref[0]) == len(sample[0])


def test_unknown_route_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(live_server + "/nonexistent", timeout=5)
    assert exc_info.value.code == 404
