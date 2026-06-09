"""Tests for format_alert — terminal panel title and content."""
from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from canary_ml.alerts import format_alert
from canary_ml.report import DriftReport


def _report(**kwargs) -> DriftReport:
    defaults = dict(
        timestamp="2026-01-01T00:00:00+00:00",
        n_samples=100,
        psi_score=0.05,
        ks_results={"0": {"statistic": 0.1, "p_value": 0.3, "drifted": False}},
        anomaly_rate=0.02,
        drift_detected=False,
        alert_triggered=False,
        alert_reasons=[],
    )
    defaults.update(kwargs)
    return DriftReport(**defaults)


def _capture(report: DriftReport) -> str:
    buf = StringIO()
    console = Console(file=buf, highlight=False, width=120)
    # Temporarily replace the module-level console
    import canary_ml.alerts as _alerts
    orig = _alerts._console
    _alerts._console = console
    try:
        format_alert(report)
    finally:
        _alerts._console = orig
    return buf.getvalue()


def test_stable_shows_stable_label():
    out = _capture(_report())
    assert "STABLE" in out


def test_drift_warning_when_drift_detected_but_no_alert():
    out = _capture(_report(drift_detected=True))
    assert "DRIFT WARNING" in out


def test_drift_alert_title_for_drift_reason():
    out = _capture(_report(
        alert_triggered=True,
        alert_reasons=["drift"],
        psi_score=0.5,
        drift_detected=True,
    ))
    assert "DRIFT ALERT" in out


def test_anomaly_alert_title_for_anomaly_reason():
    out = _capture(_report(
        alert_triggered=True,
        alert_reasons=["anomaly"],
        anomaly_rate=0.2,
    ))
    assert "ANOMALY ALERT" in out


def test_performance_alert_title_for_performance_reason():
    out = _capture(_report(
        alert_triggered=True,
        alert_reasons=["performance"],
        performance_alert=True,
        estimated_accuracy=0.7,
        reference_accuracy=0.9,
        performance_delta=-0.2,
    ))
    assert "PERFORMANCE ALERT" in out


def test_generic_alert_title_for_multiple_reasons():
    out = _capture(_report(
        alert_triggered=True,
        alert_reasons=["drift", "anomaly"],
        psi_score=0.5,
        anomaly_rate=0.2,
    ))
    assert "ALERT" in out
    assert "DRIFT ALERT" not in out
    assert "ANOMALY ALERT" not in out


def test_psi_score_shown_in_panel():
    out = _capture(_report(psi_score=0.341))
    assert "0.341" in out


def test_estimated_accuracy_shown_when_present():
    out = _capture(_report(
        estimated_accuracy=0.87,
        reference_accuracy=0.90,
        performance_delta=-0.03,
    ))
    assert "87" in out
