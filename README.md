# canary-ml

**Drop-in drift and anomaly monitoring for production ML models.**

[![PyPI](https://img.shields.io/pypi/v/canary-ml)](https://pypi.org/project/canary-ml/)
[![Python](https://img.shields.io/pypi/pyversions/canary-ml)](https://pypi.org/project/canary-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)](tests/)

One line wraps your model. Every `.predict()` call logs drift metrics, detects anomalies, and fires an alert when something shifts. Monitoring runs in a background thread — your inference latency is unaffected. No infrastructure required.

[Project page](https://aitor1717.github.io/canary-ml/) · [Guide & manual](https://aitor1717.github.io/canary-ml/guide.html) · [Live demo](https://aitor1717.github.io/canary-ml/demo.html)

---

## Install

```bash
pip install canary-ml
```

Requires Python 3.9–3.12. Dependencies: numpy, scipy, scikit-learn, rich.

Keras/TensorFlow model monitoring also requires TensorFlow (Python 3.9–3.12 only):

```bash
pip install canary-ml[keras]
```

---

## Quickstart

```python
from canary_ml import ModelMonitor

monitor = ModelMonitor(
    model=your_model,           # any sklearn-compatible model
    reference_data=X_train,     # baseline distribution
    alert_threshold=0.2,        # PSI threshold for alerts
    log_path="./canary_logs",
    verbose=True,
)

# Drop-in replacement — monitoring runs in the background
predictions = monitor.predict(X_new)
monitor.wait()  # block until background thread finishes

# Inspect the latest report
report = monitor.get_report()
print(report.summary())
# DriftReport | psi=0.41 | features_drifted=3/8 | anomaly_rate=3.2% | ALERT

# Launch the live dashboard
monitor.serve_dashboard(port=8501)
# → http://localhost:8501
```

---

## What it monitors

- **PSI** — global distribution shift. < 0.1 stable · 0.1–0.2 moderate · > 0.2 alert. Requires ≥ 200 samples per batch; use `drift_detected` (KS-based) for smaller batches.
- **KS test** — per-feature Kolmogorov-Smirnov (continuous features, p < 0.05 = drift). Note: with many features, expect ~5% false positives per feature under the null; `drift_detected` and `features_drifted` will occasionally fire on clean data at scale.
- **Chi² test** — per-feature chi-squared (categorical features, ≤ 20 unique values).
- **Anomaly detection** — ensemble of Isolation Forest + z-score (|z| > 3).
- **Confidence estimate** — label-free accuracy proxy from predicted probabilities. Accurate when probabilities are well-calibrated; overestimates if the model is overconfident.

---

## Alert callback

```python
def my_alert(report):
    send_slack(f"Drift alert: PSI={report.psi_score:.2f}")

monitor = ModelMonitor(..., on_alert=my_alert)
```

---

## Dashboard

```python
monitor.serve_dashboard(port=8501)
```

Stdlib HTTP server, no extra dependencies. Auto-refreshes every 5 seconds. Can also run standalone:

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
    alert_threshold=0.2,        # PSI threshold for drift alert
    performance_threshold=0.05, # accuracy drop (pp) below reference that fires a perf alert
    anomaly_contamination=0.05, # expected fraction of anomalies; alert fires at 4×
    categorical_threshold=20,   # max unique values for a feature to be treated as categorical
    store_samples=True,         # set False to skip storing raw feature rows (recommended in PII-sensitive envs)
    log_path="./canary_logs",
    verbose=True,               # default True — set False to suppress console output
    on_alert=None,              # callable(DriftReport) fired on alert
)
```

| Method | Returns | Description |
|---|---|---|
| `.predict(X)` | same as model | Runs model; monitoring queued in background thread |
| `.wait()` | — | Block until background monitoring tasks complete |
| `.get_report()` | `DriftReport \| None` | Latest monitoring report |
| `.serve_dashboard(port=8501)` | — | Starts dashboard server in background thread |

### `DriftReport`

| Attribute | Type | Description |
|---|---|---|
| `psi_score` | `float` | Global PSI vs reference |
| `drift_detected` | `bool` | `True` if any feature's KS/chi² p < 0.05 (soft warning) |
| `ks_results` | `dict` | Per-feature `{statistic, p_value, drifted}` |
| `features_drifted` | `int` | Count of features with p < 0.05 (computed property) |
| `anomaly_rate` | `float` | Fraction of samples flagged as anomalies |
| `alert_triggered` | `bool` | `True` if PSI > threshold, anomaly rate is high, or performance drops |
| `alert_reasons` | `list[str]` | Which conditions fired: `"drift"`, `"anomaly"`, `"performance"` |
| `estimated_accuracy` | `float \| None` | Confidence estimate; `None` if no `predict_proba` |
| `reference_accuracy` | `float \| None` | Confidence estimate on reference data |
| `performance_delta` | `float \| None` | `estimated_accuracy − reference_accuracy` |
| `output_ks` | `dict\|None` | KS test on prediction distribution vs. reference; `None` if unavailable |
| `performance_alert` | `bool` | `True` if delta < −performance_threshold |
| `timestamp` | `str` | ISO 8601 |

`DriftReport` is not directly JSON-serialisable. Use `report.to_dict()` for logging or `json.dumps(report.to_dict())`. Dict-style access (`report["psi_score"]`) is also supported.

---

## Testing

```bash
pip install -e ".[dev]"
pytest                        # 52 tests
pytest --cov=canary_ml
```

---

## License

MIT © Aitor Bazo
