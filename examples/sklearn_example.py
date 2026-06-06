"""sklearn quickstart — breast cancer dataset with artificial drift."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from canary_ml import ModelMonitor

# ── Train ─────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

monitor = ModelMonitor(
    model=clf,
    reference_data=X_train,
    alert_threshold=0.2,
    log_path="./canary_logs",
)

# ── Clean batch — no alert expected ──────────────────────────────────────────
print("\n--- Clean batch (no drift expected) ---")
preds_clean = monitor.predict(X_test[:50])
report = monitor.get_report()
print(report.summary())

# ── Drifted batch — alert should fire ────────────────────────────────────────
print("\n--- Drifted batch (drift alert expected) ---")
X_drifted = X_test.copy()
drift_cols = [0, 1, 2]
rng = np.random.default_rng(99)
for col in drift_cols:
    X_drifted[:, col] = X_drifted[:, col] * 1.5 + rng.normal(0, 0.5, len(X_drifted))

preds_drifted = monitor.predict(X_drifted)
report = monitor.get_report()
print(report.summary())

print("\n--- History (last 2 entries) ---")
for entry in monitor.get_history(n=2):
    print(f"  psi={entry['psi_score']:.3f}  alert={entry['alert_triggered']}")

print("\nLaunching dashboard...")
monitor.serve_dashboard(port=8501)
input("\nPress Enter to exit.\n")
