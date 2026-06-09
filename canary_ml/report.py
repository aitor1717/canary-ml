from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriftReport:
    """Snapshot of a single monitoring pass."""

    timestamp: str
    n_samples: int
    psi_score: float
    ks_results: dict[str, Any]
    anomaly_rate: float
    drift_detected: bool
    alert_triggered: bool
    output_ks: dict[str, Any] | None = None
    # CBPE fields — None when model has no predict_proba
    estimated_accuracy: float | None = None
    reference_accuracy: float | None = None
    performance_delta: float | None = None
    performance_alert: bool = False
    # Which conditions triggered the alert: "drift", "anomaly", "performance"
    alert_reasons: list[str] = field(default_factory=list)

    @property
    def features_drifted(self) -> int:
        """Number of features where KS/chi² p-value < 0.05."""
        return sum(1 for v in self.ks_results.values() if v.get("drifted"))

    def summary(self) -> str:
        """One-line human-readable summary."""
        drifted = sum(1 for v in self.ks_results.values() if v.get("drifted"))
        parts = [
            f"DriftReport | psi={self.psi_score:.2f}",
            f"features_drifted={drifted}",
            f"anomaly_rate={self.anomaly_rate * 100:.1f}%",
        ]
        if self.estimated_accuracy is not None:
            parts.append(f"est_acc={self.estimated_accuracy:.1%}")
        if self.performance_alert:
            parts.append("PERF_ALERT")
        status = "ALERT" if self.alert_triggered else ("WARN" if self.drift_detected else "OK")
        parts.append(status)
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dict (suitable for JSON logging)."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "psi_score": self.psi_score,
            "ks_results": self.ks_results,
            "anomaly_rate": self.anomaly_rate,
            "drift_detected": self.drift_detected,
            "alert_triggered": self.alert_triggered,
        }
        if self.alert_reasons:
            d["alert_reasons"] = self.alert_reasons
        if self.output_ks is not None:
            d["output_ks"] = self.output_ks
        if self.estimated_accuracy is not None:
            d["estimated_accuracy"] = round(self.estimated_accuracy, 4)
            d["reference_accuracy"] = round(self.reference_accuracy, 4) if self.reference_accuracy is not None else None
            d["performance_delta"] = round(self.performance_delta, 4) if self.performance_delta is not None else None
            d["performance_alert"] = self.performance_alert
        return d

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def keys(self):
        return self.to_dict().keys()
