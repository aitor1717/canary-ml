# canary-ml

Lightweight drift and anomaly monitoring for production ML models.

## Install

```bash
pip install canary-ml
```

## Quickstart

```python
from canary_ml import ModelMonitor

monitor = ModelMonitor(
    model=your_model,
    reference_data=X_train,
    alert_threshold=0.2,
    log_path="./canary_logs"
)

# drop-in replacement — monitoring is a side effect
predictions = monitor.predict(X_new)

report = monitor.get_report()
print(report.summary())
# DriftReport | psi=0.41 | features_drifted=3 | anomaly_rate=3.2% | ALERT

monitor.serve_dashboard(port=8501)
```

## Features

- Data Drift Detection — KS test, PSI, chi-square per feature with configurable thresholds
- Anomaly Detection — Isolation Forest and z-score ensemble on inputs and outputs
- Zero Latency — monitoring is a side effect; inference path stays unchanged
- Live Dashboard — zero-dep HTML/JS, ships with the package, no cloud account needed

## Why

Most ML monitoring tools require a database, a cloud account, or a separate deployment pipeline. canary-ml wraps your model with a single line of code and starts logging drift metrics immediately — to a local JSON-lines file, with no external dependencies.

The dashboard (`monitor.serve_dashboard()`) reads from that file and auto-refreshes every 5 seconds. You can run it on a laptop, in a Docker container, or on any machine with the package installed.

## Roadmap

### v1.1
- Label-free performance estimation: estimate model accuracy/F1 from confidence score distributions without ground truth labels, using confidence-based performance estimation (CBPE). Alerts when estimated performance degrades, not just when inputs shift.

## License

MIT
