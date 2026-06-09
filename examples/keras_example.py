"""Keras quickstart — identical ModelMonitor interface with a simple MLP."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from canary_ml import ModelMonitor

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise SystemExit(
        "TensorFlow is not installed. Run: pip install canary-ml"
    )

# ── Data ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Build & train MLP ─────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1,  activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# ── Monitor — identical to the sklearn example ────────────────────────────────
monitor = ModelMonitor(
    model=model,
    reference_data=X_train,
    alert_threshold=0.2,
    log_path="./canary_logs_keras",
)

print("\n--- Clean batch ---")
preds = monitor.predict(X_test[:50])
print(monitor.get_report().summary())

# Introduce drift
X_drifted = X_test.copy()
rng = np.random.default_rng(77)
for col in [0, 1, 2]:
    X_drifted[:, col] = X_drifted[:, col] * 1.5 + rng.normal(0, 0.5, len(X_drifted))

print("\n--- Drifted batch ---")
monitor.predict(X_drifted)
print(monitor.get_report().summary())

monitor.serve_dashboard(port=8502)
input("\nPress Enter to exit.\n")
