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

    def summary(self) -> str:
        """One-line human-readable summary."""
        drifted = sum(1 for v in self.ks_results.values() if v.get("drifted"))
        status = "ALERT" if self.alert_triggered else ("WARN" if self.drift_detected else "OK")
        return (
            f"DriftReport | psi={self.psi_score:.2f} "
            f"| features_drifted={drifted} "
            f"| anomaly_rate={self.anomaly_rate * 100:.1f}% "
            f"| {status}"
        )

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
        if self.output_ks is not None:
            d["output_ks"] = self.output_ks
        return d
