"""Streamlit dashboard for canary-ml. Launch via ModelMonitor.serve_dashboard()."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from canary_ml.storage import MonitorLog

# ── Design tokens ────────────────────────────────────────────────────────────
C = {
    "bg":        "#0d0d0b",
    "surface":   "#131310",
    "surfaceHi": "#1a1a17",
    "border":    "#1e1e1a",
    "borderHi":  "#2a2a24",
    "text":      "#cac6b4",
    "muted":     "#56564e",
    "yellow":    "#d4a827",
    "yellowDim": "#7a6018",
    "red":       "#bf4040",
    "redDim":    "#7a2828",
    "green":     "#4a8a54",
}

PLOTLY_BASE = dict(
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["surface"],
    font=dict(family="Fira Code, monospace", color=C["muted"], size=10),
    margin=dict(l=40, r=20, t=20, b=30),
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="canary · monitoring",
    page_icon="🐦",
    layout="wide",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800;900&family=Fira+Code:wght@300;400;500&display=swap');
  html, body, [class*="css"] {{
      background: {C["bg"]} !important;
      color: {C["text"]};
      font-family: 'Fira Code', monospace;
  }}
  .block-container {{ padding: 0 1.5rem 2rem !important; max-width: 1440px; }}
  div[data-testid="metric-container"] {{
      background: {C["surface"]};
      border: 1px solid {C["borderHi"]};
      border-radius: 5px;
      padding: 14px 18px;
  }}
  div[data-testid="metric-container"] label {{
      font-size: 10px !important;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {C["muted"]} !important;
  }}
  div[data-testid="metric-container"] [data-testid="metric-value"] {{
      font-family: 'Barlow Condensed', sans-serif !important;
      font-weight: 700;
      font-size: 2rem !important;
  }}
  .canary-header {{
      position: sticky; top: 0; z-index: 100;
      background: rgba(13,13,11,0.95);
      border-bottom: 1px solid {C["border"]};
      padding: 13px 0 13px;
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 20px;
  }}
  .canary-logo {{
      font-family: 'Barlow Condensed', sans-serif;
      font-weight: 800; font-size: 18px;
      letter-spacing: 0.06em; text-transform: uppercase;
      color: {C["yellow"]};
  }}
  .badge-alert  {{ font-size:10px; color:{C["red"]};    border:1px solid {C["redDim"]};    border-radius:2px; padding:2px 8px; letter-spacing:.08em; text-transform:uppercase; }}
  .badge-warn   {{ font-size:10px; color:{C["yellow"]}; border:1px solid {C["yellowDim"]}; border-radius:2px; padding:2px 8px; letter-spacing:.08em; text-transform:uppercase; }}
  .badge-stable {{ font-size:10px; color:{C["muted"]};  border:1px solid {C["border"]};    border-radius:2px; padding:2px 8px; letter-spacing:.08em; text-transform:uppercase; }}
  .panel {{ background:{C["surface"]}; border:1px solid {C["borderHi"]}; border-radius:5px; padding:18px 20px; margin-bottom:18px; }}
  .panel-title {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:{C["muted"]}; margin-bottom:14px; }}
  stDataFrame {{ background: {C["surface"]} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
log_path = os.environ.get("CANARY_LOG_PATH", "./canary_logs")
log = MonitorLog(log_path)
entries = log.read_all()

if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = 0

# ── Header ────────────────────────────────────────────────────────────────────
if entries:
    latest = entries[-1]
    psi_latest = latest.get("psi_score", 0.0)
    anom_latest = latest.get("anomaly_rate", 0.0)

    if latest.get("alert_triggered"):
        badge = '<span class="badge-alert">● drift detected</span>'
    elif latest.get("drift_detected"):
        badge = '<span class="badge-warn">● monitoring</span>'
    else:
        badge = '<span class="badge-stable">stable</span>'

    ts = latest.get("timestamp", "")[:19].replace("T", " ")
    n = latest.get("n_samples", 0)
    model_name = Path(log_path).name or "model"

    st.markdown(f"""
    <div class="canary-header">
      <div style="display:flex;align-items:center;gap:20px;">
        <span class="canary-logo">canary</span>
        <span style="color:{C["border"]};font-size:16px;">|</span>
        <span style="font-size:12px;color:{C["muted"]};">{model_name}</span>
        {badge}
      </div>
      <div style="font-size:11px;color:{C["muted"]};">
        last batch: <span style="color:{C["text"]};">{ts}</span>
        &nbsp;·&nbsp; {n:,} samples
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="canary-header">
      <span class="canary-logo">canary</span>
      <span style="font-size:11px;color:{C["muted"]};">no data yet — run monitor.predict()</span>
    </div>
    """, unsafe_allow_html=True)

if not entries:
    st.stop()

# ── Stat cards ────────────────────────────────────────────────────────────────
psi_val  = latest.get("psi_score", 0.0)
ks_val   = max((v.get("statistic", 0) for v in latest.get("ks_results", {}).values()), default=0.0)
anom_val = latest.get("anomaly_rate", 0.0) * 100
batch_n  = latest.get("n_samples", 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("PSI score",     f"{psi_val:.3f}",  "threshold: 0.20")
col2.metric("KS statistic",  f"{ks_val:.3f}",   f"features drifted: {sum(1 for v in latest.get('ks_results', {}).values() if v.get('drifted'))}")
col3.metric("Anomaly rate",  f"{anom_val:.1f}%", None)
col4.metric("Batch size",    f"{batch_n:,}",     None)

# Color metric values via JS-free CSS targeting (approximate — Streamlit limitation)
psi_color  = C["red"] if psi_val > 0.2 else C["text"]
ks_color   = C["red"] if ks_val  > 0.2 else C["text"]
anom_color = C["yellow"] if anom_val > 2 else C["text"]
st.markdown(f"""
<style>
  div[data-testid="metric-container"]:nth-child(1) [data-testid="metric-value"] {{ color:{psi_color}; }}
  div[data-testid="metric-container"]:nth-child(2) [data-testid="metric-value"] {{ color:{ks_color}; }}
  div[data-testid="metric-container"]:nth-child(3) [data-testid="metric-value"] {{ color:{anom_color}; }}
  div[data-testid="metric-container"]:nth-child(4) [data-testid="metric-value"] {{ color:{C["text"]}; }}
</style>
""", unsafe_allow_html=True)

st.write("")

# ── Drift timeline ────────────────────────────────────────────────────────────
last30 = entries[-30:]
batches    = [e.get("timestamp", "")[-8:][:5] for e in last30]
psi_series = [e.get("psi_score", 0.0) for e in last30]
anom_series = [e.get("anomaly_rate", 0.0) * 100 for e in last30]

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=batches, y=psi_series,
    name="PSI", line=dict(color=C["yellow"], width=1.5),
    yaxis="y1",
))
fig_timeline.add_trace(go.Scatter(
    x=batches, y=anom_series,
    name="anomaly %", line=dict(color="#6b9ed4", width=1.5),
    yaxis="y2",
))
fig_timeline.add_hline(
    y=0.2, line_dash="dash", line_color=C["yellowDim"],
    annotation_text="threshold", annotation_font_color=C["yellowDim"],
    annotation_font_size=9, yref="y",
)
fig_timeline.update_layout(
    **PLOTLY_BASE,
    height=180,
    xaxis=dict(tickfont=dict(size=9), gridcolor=C["border"], linecolor=C["border"]),
    yaxis=dict(title="PSI", gridcolor=C["border"], linecolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
    yaxis2=dict(title="anomaly %", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, font=dict(size=10)),
    showlegend=True,
)

st.markdown('<div class="panel"><div class="panel-title">Drift Timeline · last 30 batches</div>', unsafe_allow_html=True)
st.plotly_chart(fig_timeline, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ── Heatmap + Distributions ───────────────────────────────────────────────────
left, right = st.columns(2)

# Build per-feature PSI history (last 12 windows)
last12 = entries[-12:]
all_feature_keys: list[str] = []
for e in last12:
    for k in e.get("ks_results", {}):
        if k not in all_feature_keys:
            all_feature_keys.append(k)

feature_labels: list[str] = []

def _feat_label(idx: str) -> str:
    try:
        return f"feature_{int(idx)}"
    except ValueError:
        return idx

for k in all_feature_keys:
    feature_labels.append(_feat_label(k))

# Build PSI heatmap matrix (features × windows)
# Approximate per-feature PSI from ks statistic (we store global PSI, not per-feature)
# Use KS statistic as a proxy for visual intensity
z_matrix: list[list[float]] = []
for feat_k in all_feature_keys:
    row = []
    for e in last12:
        ks = e.get("ks_results", {}).get(feat_k, {}).get("statistic", 0.0)
        row.append(ks)
    z_matrix.append(row)

batch_labels = [e.get("timestamp", "")[-8:][:5] for e in last12]

# Custom colorscale: transparent below 0.1, yellow 0.1–0.2, red above 0.2
colorscale = [
    [0.0,  "rgba(0,0,0,0)"],
    [0.09, "rgba(0,0,0,0)"],
    [0.1,  f"rgba(212,168,39,0.3)"],
    [0.2,  f"rgba(212,168,39,0.9)"],
    [0.21, f"rgba(191,64,64,0.5)"],
    [1.0,  f"rgba(191,64,64,1.0)"],
]

fig_heat = go.Figure(go.Heatmap(
    z=z_matrix,
    x=batch_labels,
    y=feature_labels,
    colorscale=colorscale,
    zmin=0, zmax=0.5,
    showscale=False,
    hovertemplate="feature: %{y}<br>window: %{x}<br>KS: %{z:.3f}<extra></extra>",
))
fig_heat.update_layout(
    **PLOTLY_BASE,
    height=max(180, len(feature_labels) * 28 + 60),
    xaxis=dict(tickfont=dict(size=8), side="top"),
    yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
)

with left:
    st.markdown('<div class="panel"><div class="panel-title">Feature Drift Map · KS per window</div>', unsafe_allow_html=True)
    # Render feature selection via click workaround (radio buttons)
    sel = st.radio(
        "Select feature:",
        options=list(range(len(feature_labels))),
        format_func=lambda i: feature_labels[i] if i < len(feature_labels) else str(i),
        index=st.session_state.selected_feature,
        label_visibility="collapsed",
        horizontal=False,
    )
    if sel != st.session_state.selected_feature:
        st.session_state.selected_feature = sel
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# Distribution comparison
with right:
    st.markdown('<div class="panel"><div class="panel-title">Distribution Shift</div>', unsafe_allow_html=True)

    # Determine which features to show: selected from heatmap, or top-3 by KS
    top_features = sorted(
        all_feature_keys,
        key=lambda k: latest.get("ks_results", {}).get(k, {}).get("statistic", 0),
        reverse=True,
    )[:3]

    sel_key = all_feature_keys[st.session_state.selected_feature] if st.session_state.selected_feature < len(all_feature_keys) else None
    show_keys = [sel_key] if sel_key else top_features

    feature_samples = latest.get("feature_sample")
    if feature_samples and all_feature_keys:
        arr = np.array(feature_samples, dtype=float)
        for feat_k in show_keys[:3]:
            try:
                col_idx = int(feat_k)
            except (ValueError, TypeError):
                continue
            if col_idx >= arr.shape[1]:
                continue

            col_data = arr[:, col_idx]
            label = _feat_label(feat_k)

            # Build histogram traces: baseline approximated from ks results isn't stored,
            # so we show current distribution vs a synthetic baseline (centered at 0)
            # For a real baseline, store reference sample in the first log entry.
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=col_data,
                name="current",
                marker_color=C["yellow"],
                opacity=0.75,
                nbinsx=25,
            ))
            fig_dist.update_layout(
                **PLOTLY_BASE,
                height=100,
                showlegend=False,
                margin=dict(l=30, r=10, t=10, b=20),
                bargap=0.05,
                xaxis=dict(tickfont=dict(size=8), gridcolor=C["border"]),
                yaxis=dict(tickfont=dict(size=8), gridcolor=C["border"]),
            )
            st.markdown(f'<div style="font-size:10px;color:{C["muted"]};letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;">{label}</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(f'<span style="font-size:11px;color:{C["muted"]};">No feature samples in log yet.</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Alert log ─────────────────────────────────────────────────────────────────
alert_entries = [e for e in reversed(entries) if e.get("alert_triggered") or e.get("drift_detected")][:20]

if alert_entries:
    rows = []
    for e in alert_entries:
        ks_max = max((v.get("statistic", 0) for v in e.get("ks_results", {}).values()), default=0.0)
        drifted = sum(1 for v in e.get("ks_results", {}).values() if v.get("drifted"))
        status = "ALERT" if e.get("alert_triggered") else "WARN"
        rows.append({
            "Time":           e.get("timestamp", "")[:19].replace("T", " "),
            "PSI":            round(e.get("psi_score", 0), 3),
            "KS max":         round(ks_max, 3),
            "Features drifted": drifted,
            "Status":         status,
        })

    df = pd.DataFrame(rows)

    def color_status(val: str) -> str:
        if val == "ALERT":
            return f"color: {C['red']}"
        return f"color: {C['yellow']}"

    styled = df.style.applymap(color_status, subset=["Status"])

    st.markdown('<div class="panel"><div class="panel-title">Alert Log · last 20</div>', unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(5)
st.rerun()
