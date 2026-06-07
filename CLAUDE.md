# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_monitor.py

# Run a single test
pytest tests/test_monitor.py::test_get_report_populates_after_predict

# Run with coverage
pytest --cov=canary_ml

# Launch the dashboard server directly
.venv/bin/python -m canary_ml.server ./canary_logs 8501
```

## Architecture

**canary-ml** is a drop-in model monitoring wrapper. The public API is `ModelMonitor` (re-exported from `canary_ml/__init__.py`), which wraps any sklearn-compatible model and makes monitoring a side effect of `.predict()`.

### Data flow

1. `ModelMonitor.__init__` fits an `AnomalyDetector` on reference data and writes a `reference.json` sample to disk (used by the dashboard for distribution comparison).
2. `ModelMonitor.predict(X)` calls the underlying model, then `_monitor(X)` in a try/except — monitoring failures never propagate to the caller.
3. `_monitor` runs `detect_drift` (PSI + per-feature KS/chi² routing) and `AnomalyDetector.score`, assembles a `DriftReport`, and appends it as a JSON-lines entry via `MonitorLog`.
4. If verbose and drift/alert detected, `format_alert` prints a rich panel.

### Module responsibilities

- **`monitor.py`** — `ModelMonitor`: the only entry point users interact with. Orchestrates all other modules.
- **`drift.py`** — `detect_drift`: routes continuous features to KS test and categorical features (≤20 unique values) to chi², computes global PSI. All three functions (`ks_drift`, `chi2_drift`, `psi_score`) operate on numpy arrays with shape `(n_samples, n_features)`.
- **`anomaly.py`** — `AnomalyDetector`: ensemble of IsolationForest + z-score (|z| > 3). `fit` on reference; `score` on new batches.
- **`storage.py`** — `MonitorLog`: append-only `.jsonl` file at `<log_path>/monitor.jsonl`.
- **`report.py`** — `DriftReport` dataclass: snapshot of one monitoring pass. Alert threshold logic is in `alerts.py`; triggering is PSI-based (`psi_score > threshold`).
- **`alerts.py`** — `check_alert` (returns bool) and `format_alert` (prints rich panel).
- **`server.py`** — Stdlib `http.server` serving `dashboard.html` at `/`, `/api/data?n=N`, and `/api/reference`. CORS open. `monitor.serve_dashboard()` starts this in a daemon thread. Can also run standalone: `python -m canary_ml.server <log_path> <port>`.

### Log format

Each line in `monitor.jsonl` is a `DriftReport.to_dict()` plus a `feature_sample` key (up to 500 rows of raw input, for distribution plots). The `reference.json` file holds up to 500 rows of baseline data.

### PSI thresholds

< 0.1 stable · 0.1–0.2 moderate · > 0.2 significant shift (triggers alert at default `alert_threshold=0.2`).
