from __future__ import annotations

import json
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scipy import stats as _scipy_stats

from canary_ml.alerts import format_alert
from canary_ml.anomaly import AnomalyDetector
from canary_ml.cbpe import estimate_accuracy
from canary_ml.drift import detect_drift
from canary_ml.report import DriftReport
from canary_ml.storage import MonitorLog

_MIN_PSI_SAMPLES = 200
_log = logging.getLogger(__name__)


class ModelMonitor:
    """Drop-in wrapper that adds monitoring as a side effect of prediction.

    Usage::

        monitor = ModelMonitor(model=clf, reference_data=X_train)
        predictions = monitor.predict(X_new)   # identical to clf.predict(X_new)
    """

    def __init__(
        self,
        model: Any,
        reference_data: Any,
        feature_names: list[str] | None = None,
        alert_threshold: float = 0.2,
        anomaly_contamination: float = 0.05,
        log_path: str | os.PathLike = "./canary_logs",
        verbose: bool = True,
        on_alert: Callable[[DriftReport], None] | None = None,
        performance_threshold: float = 0.05,
        categorical_threshold: int = 20,
        store_samples: bool = True,
    ) -> None:
        self._model = model
        self._reference = np.asarray(reference_data, dtype=float)
        if self._reference.ndim == 1:
            self._reference = self._reference.reshape(-1, 1)

        self._feature_names = feature_names
        self._alert_threshold = alert_threshold
        self._anomaly_contamination = anomaly_contamination
        self._performance_threshold = performance_threshold
        self._categorical_threshold = categorical_threshold
        self._store_samples = store_samples
        self._verbose = verbose
        self._on_alert = on_alert
        self._log_path = Path(log_path)
        self._log = MonitorLog(log_path)
        self._report: DriftReport | None = None
        self._report_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="canary")

        self._detector = AnomalyDetector(contamination=anomaly_contamination)
        self._detector.fit(self._reference)

        # Capture reference output distribution for output drift detection.
        self._reference_outputs: np.ndarray | None = _safe_predict(model, self._reference)

        # CBPE — estimate reference accuracy from predict_proba if available.
        self._cbpe_enabled = False
        self._reference_estimated_accuracy: float | None = None
        if hasattr(model, "predict_proba"):
            try:
                ref_probas = model.predict_proba(self._reference)
                self._reference_estimated_accuracy = estimate_accuracy(ref_probas)
                self._cbpe_enabled = True
            except Exception:  # noqa: BLE001
                pass

        self._save_reference_sample()

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, X: Any) -> np.ndarray:
        """Run model prediction. Monitoring runs in a background thread."""
        result = self._model.predict(X)
        probas: np.ndarray | None = None
        if self._cbpe_enabled:
            try:
                probas = self._model.predict_proba(X)
            except Exception:  # noqa: BLE001
                _log.debug("predict_proba failed", exc_info=True)
        arr = _to_2d(X)
        self._executor.submit(self._monitor_safe, arr, result, probas)
        return result

    def predict_proba(self, X: Any) -> np.ndarray:
        """Passthrough to model.predict_proba if supported."""
        probas = getattr(self._model, "predict_proba")(X)
        outputs: np.ndarray | None = None
        try:
            outputs = self._model.predict(X)
        except Exception:  # noqa: BLE001
            _log.debug("predict failed inside predict_proba", exc_info=True)
        arr = _to_2d(X)
        self._executor.submit(self._monitor_safe, arr, outputs, probas)
        return probas

    def get_report(self) -> DriftReport | None:
        """Return the most recent DriftReport (None before first predict)."""
        with self._report_lock:
            return self._report

    def _flush(self) -> None:
        """Block until all pending monitoring tasks finish. For testing only."""
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="canary")

    def get_history(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last *n* log entries."""
        return self._log.read_last(n)

    def reset_baseline(self, new_data: Any) -> None:
        """Replace the reference distribution and refit the anomaly detector."""
        self._reference = np.asarray(new_data, dtype=float)
        if self._reference.ndim == 1:
            self._reference = self._reference.reshape(-1, 1)
        self._detector.fit(self._reference)
        self._reference_outputs = _safe_predict(self._model, self._reference)
        self._save_reference_sample()

    def serve_dashboard(self, port: int = 8501) -> None:
        """Launch the canary dashboard server in a background thread."""
        from canary_ml import server as _server
        _server.start(self._log_path, port=port)
        from rich.console import Console
        Console().print(f"[bold yellow]canary[/bold yellow] dashboard → http://localhost:{port}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _monitor_safe(self, arr: np.ndarray, outputs: Any, probas: np.ndarray | None) -> None:
        try:
            self._monitor(arr, outputs, probas=probas)
        except Exception as exc:  # noqa: BLE001
            self._log.append({"error": str(exc), "timestamp": _now()})

    def _monitor(self, arr: np.ndarray, outputs: Any, probas: np.ndarray | None = None) -> None:
        if len(arr) < _MIN_PSI_SAMPLES and self._verbose:
            from rich.console import Console
            Console().print(
                f"[bold yellow]canary[/bold yellow] warning: batch size {len(arr)} "
                f"is below {_MIN_PSI_SAMPLES} — PSI may be unreliable. "
                f"Use drift_detected (KS-based) for small batches."
            )

        anomaly_result = self._detector.score(arr)
        psi, ks_results = detect_drift(self._reference, arr, categorical_threshold=self._categorical_threshold)

        # Output distribution drift — always KS so the statistic is bounded [0,1]
        output_ks: dict[str, Any] | None = None
        if self._reference_outputs is not None and outputs is not None:
            try:
                out_arr = np.asarray(outputs, dtype=float).ravel()
                ref_out = self._reference_outputs.ravel()
                stat, p = _scipy_stats.ks_2samp(ref_out, out_arr)
                output_ks = {"statistic": float(stat), "p_value": float(p), "drifted": bool(p < 0.05)}
            except Exception:  # noqa: BLE001
                _log.debug("output KS failed", exc_info=True)

        # Confidence estimate — label-free performance proxy
        estimated_accuracy: float | None = None
        performance_delta: float | None = None
        performance_alert = False
        if probas is not None and self._reference_estimated_accuracy is not None:
            try:
                estimated_accuracy = estimate_accuracy(probas)
                performance_delta = estimated_accuracy - self._reference_estimated_accuracy
                performance_alert = performance_delta < -abs(self._performance_threshold)
            except Exception:  # noqa: BLE001
                _log.debug("confidence estimate failed", exc_info=True)

        drift_detected = any(v.get("drifted") for v in ks_results.values())
        anomaly_alert = anomaly_result["anomaly_rate"] > min(0.1, self._anomaly_contamination * 3)
        # PSI alert is suppressed for small batches — variance is too high to be reliable
        psi_alert = psi > self._alert_threshold and len(arr) >= _MIN_PSI_SAMPLES
        alert_triggered = psi_alert or performance_alert or anomaly_alert

        alert_reasons: list[str] = []
        if psi_alert:
            alert_reasons.append("drift")
        if anomaly_alert:
            alert_reasons.append("anomaly")
        if performance_alert:
            alert_reasons.append("performance")

        report = DriftReport(
            timestamp=_now(),
            n_samples=len(arr),
            psi_score=psi,
            ks_results=ks_results,
            anomaly_rate=anomaly_result["anomaly_rate"],
            drift_detected=drift_detected,
            alert_triggered=alert_triggered,
            alert_reasons=alert_reasons,
            output_ks=output_ks,
            estimated_accuracy=estimated_accuracy,
            reference_accuracy=self._reference_estimated_accuracy,
            performance_delta=performance_delta,
            performance_alert=performance_alert,
        )
        with self._report_lock:
            self._report = report

        entry = report.to_dict()
        if self._store_samples:
            sample_rows = arr.tolist()
            if len(sample_rows) > 500:
                sample_rows = random.sample(sample_rows, 500)
            entry["feature_sample"] = sample_rows
        entry["feature_names"] = self._feature_names
        self._log.append(entry)

        if alert_triggered and self._on_alert is not None:
            try:
                self._on_alert(report)
            except Exception:  # noqa: BLE001
                _log.debug("on_alert callback raised", exc_info=True)

        if self._verbose and (report.drift_detected or report.alert_triggered):
            format_alert(report)

    def _save_reference_sample(self) -> None:
        self._log_path.mkdir(parents=True, exist_ok=True)
        ref = self._reference
        sample = ref.tolist() if len(ref) <= 500 else random.sample(ref.tolist(), 500)
        ref_file = self._log_path / "reference.json"
        ref_file.write_text(json.dumps(sample))


def _to_2d(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _safe_predict(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Run model.predict on X without raising — returns None on failure."""
    try:
        sample = X if len(X) <= 500 else X[np.random.choice(len(X), 500, replace=False)]
        return np.asarray(model.predict(sample), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
