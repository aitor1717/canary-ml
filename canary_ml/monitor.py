from __future__ import annotations

import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from canary_ml.alerts import check_alert, format_alert
from canary_ml.anomaly import AnomalyDetector
from canary_ml.drift import detect_drift
from canary_ml.report import DriftReport
from canary_ml.storage import MonitorLog


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
    ) -> None:
        self._model = model
        self._reference = np.asarray(reference_data, dtype=float)
        if self._reference.ndim == 1:
            self._reference = self._reference.reshape(-1, 1)

        self._feature_names = feature_names
        self._alert_threshold = alert_threshold
        self._verbose = verbose
        self._log_path = Path(log_path)
        self._log = MonitorLog(log_path)
        self._report: DriftReport | None = None

        self._detector = AnomalyDetector(contamination=anomaly_contamination)
        self._detector.fit(self._reference)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, X: Any) -> np.ndarray:
        """Run model prediction. Monitoring is a non-blocking side effect."""
        result = self._model.predict(X)
        try:
            self._monitor(X)
        except Exception as exc:  # noqa: BLE001
            self._log.append({"error": str(exc), "timestamp": _now()})
        return result

    def predict_proba(self, X: Any) -> np.ndarray:
        """Passthrough to model.predict_proba if supported."""
        result = getattr(self._model, "predict_proba")(X)
        try:
            self._monitor(X)
        except Exception as exc:  # noqa: BLE001
            self._log.append({"error": str(exc), "timestamp": _now()})
        return result

    def get_report(self) -> DriftReport | None:
        """Return the most recent DriftReport (None before first predict)."""
        return self._report

    def get_history(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last *n* log entries."""
        return self._log.read_last(n)

    def reset_baseline(self, new_data: Any) -> None:
        """Replace the reference distribution and refit the anomaly detector."""
        self._reference = np.asarray(new_data, dtype=float)
        if self._reference.ndim == 1:
            self._reference = self._reference.reshape(-1, 1)
        self._detector.fit(self._reference)

    def serve_dashboard(self, port: int = 8501) -> None:
        """Launch the Streamlit dashboard in a background subprocess."""
        dashboard = Path(__file__).parent.parent / "dashboard_app.py"
        env = os.environ.copy()
        env["CANARY_LOG_PATH"] = str(self._log_path)
        streamlit_bin = Path(__file__).parent.parent / ".venv" / "bin" / "streamlit"
        if not streamlit_bin.exists():
            streamlit_bin = "streamlit"
        subprocess.Popen(
            [str(streamlit_bin), "run", str(dashboard), "--server.port", str(port)],
            env=env,
        )
        from rich.console import Console
        Console().print(f"[bold yellow]canary[/bold yellow] dashboard → http://localhost:{port}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _monitor(self, X: Any) -> None:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        anomaly_result = self._detector.score(arr)
        psi, ks_results = detect_drift(self._reference, arr)

        drift_detected = any(v.get("drifted") for v in ks_results.values())
        alert_triggered = check_alert(
            DriftReport(
                timestamp=_now(),
                n_samples=len(arr),
                psi_score=psi,
                ks_results=ks_results,
                anomaly_rate=anomaly_result["anomaly_rate"],
                drift_detected=drift_detected,
                alert_triggered=False,  # placeholder for threshold check
            ),
            self._alert_threshold,
        )

        report = DriftReport(
            timestamp=_now(),
            n_samples=len(arr),
            psi_score=psi,
            ks_results=ks_results,
            anomaly_rate=anomaly_result["anomaly_rate"],
            drift_detected=drift_detected,
            alert_triggered=alert_triggered,
        )
        self._report = report

        sample_rows = arr.tolist()
        if len(sample_rows) > 500:
            sample_rows = random.sample(sample_rows, 500)

        entry = report.to_dict()
        entry["feature_sample"] = sample_rows
        self._log.append(entry)

        if self._verbose and (report.drift_detected or report.alert_triggered):
            format_alert(report)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
