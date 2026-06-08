# canary-ml

**Drop-in drift and anomaly monitoring for production ML models.**

[![PyPI](https://img.shields.io/pypi/v/canary-ml)](https://pypi.org/project/canary-ml/)
[![Python](https://img.shields.io/pypi/pyversions/canary-ml)](https://pypi.org/project/canary-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)](tests/)

One line wraps your model. Every `.predict()` call logs drift metrics, detects anomalies, and can fire an alert — without adding latency or requiring any infrastructure.

```python
from canary_ml import ModelMonitor

monitor = ModelMonitor(model=clf, reference_data=X_train, log_path="./logs")
predictions = monitor.predict(X_new)   # identical to clf.predict(X_new)
```

[Live demo](https://aitor1717.github.io/canary-ml/demo.html) · [Docs & guide](https://aitor1717.github.io/canary-ml/guide.html) · [Landing page](https://aitor1717.github.io/canary-ml/)

---

## Why canary-ml

Most monitoring tools need a sidecar process, a managed database, or a cloud account. canary-ml logs to a local `.jsonl` file and serves its own dashboard from a single stdlib HTTP server — nothing to deploy, nothing to pay for.

| | canary-ml | Evidently | NannyML | WhyLabs |
|---|---|---|---|---|
| Zero infrastructure | ✅ | ✅ | ✅ | Cloud |
| Single-line wrap | ✅ | ❌ | ❌ | ❌ |
| Ships own dashboard | ✅ | Report files | ✅ | Cloud |
| PSI + KS + chi² | ✅ | ✅ | ✅ | ✅ |
| Anomaly detection | ✅ | ❌ | ❌ | ❌ |
| on_alert callback | ✅ | ❌ | ❌ | ❌ |
| Python-only install | ✅ | ✅ | ✅ | Agent |

---

## Install

```bash
pip install canary-ml
```

Requires Python 3.9+. Dependencies: numpy, scipy, scikit-learn, rich.

Optional Keras support:

```bash
pip install "canary-ml[keras]"
```

---

## Quickstart

```python
from canary_ml import ModelMonitor

monitor = ModelMonitor(
    model=your_model,           # any sklearn-compatible model
    reference_data=X_train,     # baseline distribution
    alert_threshold=0.2,        # PSI threshold for alerts (default 0.2)
    log_path="./canary_logs",   # where to write monitor.jsonl
    verbose=True,               # print rich alert panels to stdout
)

# Drop-in replacement — monitoring is a side effect of predict()
predictions = monitor.predict(X_new)

# Inspect the latest report
report = monitor.get_report()
print(report.summary())
# DriftReport | psi=0.41 | features_drifted=3/8 | anomaly_rate=3.2% | ALERT

# Launch the live dashboard (opens in background thread)
monitor.serve_dashboard(port=8501)
# → http://localhost:8501
```

---

## How it works

### Data drift (PSI + KS / chi²)

Each `.predict()` call runs three tests against the reference distribution:

- **PSI (Population Stability Index)** — quantile-based global measure. PSI < 0.1 is stable, 0.1–0.2 is moderate, > 0.2 triggers an alert.
- **KS test** — per-feature Kolmogorov-Smirnov test for continuous features (p < 0.05 = drift).
- **Chi² test** — per-feature chi-squared test for categorical features (≤ 20 unique values).

### Anomaly detection

An ensemble of **Isolation Forest** (fitted on `reference_data`) and a **z-score detector** (flags samples where any feature exceeds |z| > 3) scores each batch. The anomaly rate is the fraction of samples flagged by either detector.

### Alert callback

```python
def my_alert(report):
    send_slack(f"Drift alert: PSI={report.psi_score:.2f}")

monitor = ModelMonitor(..., on_alert=my_alert)
```

`on_alert` fires when `psi_score > alert_threshold`. It receives the full `DriftReport`.

### Dashboard

```python
monitor.serve_dashboard(port=8501)
```

Serves a zero-dependency HTML/JS dashboard at `localhost:8501`. Auto-refreshes every 5 seconds. Shows PSI timeline, per-feature KS heatmap, distribution comparison, and alert log. Can also run standalone:

```bash
python -m canary_ml.server ./canary_logs 8501
```

---

## API reference

### `ModelMonitor`

```python
ModelMonitor(
    model,                      # sklearn-compatible model with .predict()
    reference_data,             # np.ndarray or pd.DataFrame, shape (n, features)
    alert_threshold=0.2,        # PSI threshold for alert
    log_path="./canary_logs",   # directory for monitor.jsonl and reference.json
    verbose=False,              # print rich alert panels on alert
    on_alert=None,              # callable(DriftReport) fired on alert
)
```

| Method | Returns | Description |
|---|---|---|
| `.predict(X)` | same as model | Runs model + monitoring as a side effect |
| `.get_report()` | `DriftReport \| None` | Latest monitoring report |
| `.serve_dashboard(port=8501)` | — | Starts dashboard server in background thread |

### `DriftReport`

| Attribute | Type | Description |
|---|---|---|
| `psi_score` | `float` | Global PSI vs reference |
| `drift_detected` | `bool` | `True` if PSI > alert_threshold |
| `ks_results` | `dict` | Per-feature `{statistic, p_value, drift}` |
| `features_drifted` | `int` | Count of features with KS p < 0.05 |
| `anomaly_rate` | `float` | Fraction of samples flagged as anomalies |
| `alert` | `bool` | `True` if drift_detected or anomaly_rate > 0.05 |
| `timestamp` | `str` | ISO 8601 |
| `.summary()` | `str` | Human-readable one-liner |

---

## Log format

`<log_path>/monitor.jsonl` — one JSON object per `.predict()` call:

```json
{
  "timestamp": "2026-06-07T14:23:01",
  "psi_score": 0.41,
  "drift_detected": true,
  "ks_results": {"0": {"statistic": 0.38, "p_value": 0.001, "drift": true}},
  "anomaly_rate": 0.032,
  "alert": true,
  "feature_sample": [[...], ...]
}
```

`<log_path>/reference.json` — up to 500 rows of baseline data for distribution comparison.

---

## Testing

```bash
pip install -e ".[dev]"
pytest                        # 32 tests
pytest --cov=canary_ml        # with coverage
```

---

## Roadmap

**v1.1** — Label-free performance estimation (CBPE): estimate model accuracy/F1 from predicted confidence scores without ground truth labels. Alerts when estimated performance degrades, not just when inputs shift.

---

## License

MIT © Aitor Bazo
