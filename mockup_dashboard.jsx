import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, CartesianGrid
} from "recharts";

const C = {
  bg:        '#0d0d0b',
  surface:   '#131310',
  surfaceHi: '#1a1a17',
  border:    '#1e1e1a',
  borderHi:  '#2a2a24',
  text:      '#cac6b4',
  muted:     '#56564e',
  yellow:    '#d4a827',
  yellowDim: '#7a6018',
  red:       '#bf4040',
  redDim:    '#7a2828',
  green:     '#4a8a54',
};

// ── FAKE DATA ─────────────────────────────────────────────────────────────────
const seed = (i) => Math.sin(i * 9301 + 49297) * 0.5 + 0.5;

const timelineData = Array.from({ length: 30 }, (_, i) => {
  const d = i >= 21 ? (i - 21) / 8 : 0;
  return {
    batch: `#${2818 + i}`,
    psi:   +( d > 0 ? 0.03 + d * 0.40 + seed(i) * 0.02 : 0.02 + seed(i) * 0.02 ).toFixed(3),
    anom:  +( d > 0 ? 0.8  + d * 3.8  + seed(i+5) * 0.3 : 0.6 + seed(i) * 0.3 ).toFixed(2),
  };
});

const features = [
  { name: 'feature_importance', ks: 0.38, psi: 0.41, drifted: true  },
  { name: 'avg_session_len',    ks: 0.31, psi: 0.33, drifted: true  },
  { name: 'request_latency',    ks: 0.24, psi: 0.21, drifted: true  },
  { name: 'cache_hit_rate',     ks: 0.09, psi: 0.06, drifted: false },
  { name: 'error_rate',         ks: 0.07, psi: 0.05, drifted: false },
  { name: 'throughput_rps',     ks: 0.06, psi: 0.04, drifted: false },
  { name: 'memory_usage',       ks: 0.05, psi: 0.03, drifted: false },
  { name: 'cpu_utilization',    ks: 0.04, psi: 0.03, drifted: false },
];

const heatRows = [
  { name: 'feature_importance', vals: [0.02,0.03,0.02,0.03,0.03,0.04,0.03,0.07,0.14,0.26,0.38,0.41] },
  { name: 'avg_session_len',    vals: [0.03,0.02,0.04,0.02,0.04,0.05,0.04,0.11,0.21,0.29,0.33,0.33] },
  { name: 'request_latency',    vals: [0.02,0.03,0.02,0.04,0.03,0.03,0.04,0.06,0.09,0.13,0.19,0.24] },
  { name: 'cache_hit_rate',     vals: [0.02,0.03,0.02,0.04,0.03,0.02,0.03,0.04,0.03,0.05,0.05,0.06] },
  { name: 'error_rate',         vals: [0.03,0.02,0.03,0.02,0.04,0.03,0.02,0.03,0.04,0.03,0.04,0.05] },
  { name: 'throughput_rps',     vals: [0.02,0.03,0.02,0.03,0.02,0.03,0.04,0.03,0.03,0.04,0.04,0.04] },
  { name: 'memory_usage',       vals: [0.03,0.02,0.03,0.03,0.02,0.04,0.03,0.02,0.03,0.04,0.03,0.03] },
  { name: 'cpu_utilization',    vals: [0.02,0.03,0.02,0.02,0.03,0.02,0.03,0.03,0.02,0.03,0.04,0.04] },
];

const batchLabels = ['b-11','b-10','b-9','b-8','b-7','b-6','b-5','b-4','b-3','b-2','b-1','now'];

const alerts = [
  { time: '14:32:07', batch: '#2847', psi: '0.41', ks: '0.38', level: 'ALERT'  },
  { time: '14:20:51', batch: '#2846', psi: '0.38', ks: '0.33', level: 'ALERT'  },
  { time: '14:08:34', batch: '#2845', psi: '0.28', ks: '0.24', level: 'WARN'   },
  { time: '13:56:12', batch: '#2844', psi: '0.22', ks: '0.19', level: 'WARN'   },
  { time: '13:43:48', batch: '#2843', psi: '0.14', ks: '0.11', level: 'WARN'   },
];

// ── HELPERS ───────────────────────────────────────────────────────────────────
function heatColor(v) {
  if (v < 0.1)  return `rgba(74,138,84,${0.15 + v * 0.5})`;
  if (v < 0.2)  return `rgba(212,168,39,${0.2 + (v-0.1)*3})`;
  return `rgba(191,64,64,${0.3 + (v-0.2)*2})`;
}

function gauss(x, m, s) {
  return Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
}

function distPath(mean, std, w, h, xMin=0, xMax=1, N=60) {
  const pts = Array.from({ length: N+1 }, (_, i) => {
    const x = xMin + (i / N) * (xMax - xMin);
    return [x, gauss(x, mean, std)];
  });
  const maxY = Math.max(...pts.map(p => p[1]));
  const coords = pts.map(([x, y]) => [
    ((x - xMin) / (xMax - xMin)) * w,
    h - (y / maxY) * h * 0.88,
  ]);
  return coords.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
}

// ── SUB-COMPONENTS ────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, level }) {
  const valColor = level === 'alert' ? C.red : level === 'warn' ? C.yellow : C.text;
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.borderHi}`, borderRadius: 5, padding: '16px 20px' }}>
      <div style={{ fontFamily: 'Fira Code, monospace', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', color: C.muted, marginBottom: 6 }}>{label}</div>
      <div style={{ fontFamily: 'Barlow Condensed, sans-serif', fontWeight: 700, fontSize: 32, lineHeight: 1, color: valColor, letterSpacing: '0.01em' }}>{value}</div>
      {sub && <div style={{ fontFamily: 'Fira Code, monospace', fontSize: 10, color: C.muted, marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: C.surfaceHi, border: `1px solid ${C.borderHi}`, borderRadius: 4, padding: '10px 14px', fontFamily: 'Fira Code, monospace', fontSize: 11 }}>
      <div style={{ color: C.muted, marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <span style={{ color: C.text }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
};

function DistPanel({ feature, baselineMean, currentMean, baselineStd=0.11, currentStd=0.13 }) {
  const W = 200, H = 72;
  const bp = distPath(baselineMean, baselineStd, W, H);
  const cp = distPath(currentMean,  currentStd,  W, H);
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontFamily: 'Fira Code, monospace', fontSize: 10, color: C.muted, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>{feature}</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ display: 'block', height: 64 }}>
        <defs>
          <linearGradient id={`bg-${feature}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.yellow} stopOpacity="0.12"/>
            <stop offset="100%" stopColor={C.yellow} stopOpacity="0"/>
          </linearGradient>
        </defs>
        {/* Baseline fill */}
        <path d={`${bp} L${W},${H} L0,${H} Z`} fill="rgba(50,50,44,0.4)" stroke="none"/>
        {/* Current fill */}
        <path d={`${cp} L${W},${H} L0,${H} Z`} fill={`url(#bg-${feature})`} stroke="none"/>
        {/* Baseline line */}
        <path d={bp} fill="none" stroke="rgba(86,86,78,0.7)" strokeWidth="1.5"/>
        {/* Current line */}
        <path d={cp} fill="none" stroke={C.yellow} strokeWidth="1.5"/>
        {/* Axis */}
        <line x1="0" y1={H-1} x2={W} y2={H-1} stroke={C.border} strokeWidth="1"/>
      </svg>
    </div>
  );
}

// ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [selectedFeature, setSelectedFeature] = useState(0);
  const latest = timelineData[timelineData.length - 1];

  return (
    <div style={{ background: C.bg, minHeight: '100vh', color: C.text, fontFamily: 'Fira Code, monospace' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800;900&family=Fira+Code:wght@300;400;500&display=swap');
        ::-webkit-scrollbar { width: 6px; background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.borderHi}; border-radius: 3px; }
        * { box-sizing: border-box; }
      `}</style>

      {/* HEADER */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: '14px 28px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(13,13,11,0.95)', position: 'sticky', top: 0, zIndex: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <span style={{ fontFamily: 'Barlow Condensed, sans-serif', fontWeight: 800, fontSize: 18, letterSpacing: '0.06em', textTransform: 'uppercase', color: C.yellow }}>canary</span>
          <span style={{ color: C.border }}>|</span>
          <span style={{ fontSize: 12, color: C.muted }}>production_model_v3</span>
          <span style={{ fontSize: 10, color: C.red, border: `1px solid ${C.redDim}`, borderRadius: 2, padding: '2px 8px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>● drift detected</span>
        </div>
        <div style={{ fontSize: 11, color: C.muted }}>last batch: <span style={{ color: C.text }}>14:32:07</span> · #2847 · 1,240 samples</div>
      </div>

      <div style={{ padding: '24px 28px', maxWidth: 1440, margin: '0 auto' }}>

        {/* STAT CARDS */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 20 }}>
          <StatCard label="PSI score"      value="0.41" sub="threshold: 0.20"         level="alert" />
          <StatCard label="KS statistic"   value="0.38" sub="features drifted: 3"     level="alert" />
          <StatCard label="Anomaly rate"   value="4.2%" sub="↑ from baseline 0.8%"    level="warn"  />
          <StatCard label="Batch size"     value="1,240" sub="baseline avg: 1,180"     level="ok"    />
        </div>

        {/* TIMELINE */}
        <div style={{ background: C.surface, border: `1px solid ${C.borderHi}`, borderRadius: 5, padding: '18px 20px', marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 16 }}>
            <div style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted }}>Drift Timeline · last 30 batches</div>
            <div style={{ display: 'flex', gap: 20, fontSize: 10 }}>
              <span style={{ color: C.yellow }}>── PSI score</span>
              <span style={{ color: '#6b9ed4' }}>── anomaly rate %</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={timelineData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 5" stroke={C.border} vertical={false}/>
              <XAxis dataKey="batch" tick={{ fill: C.muted, fontSize: 9, fontFamily: 'Fira Code' }} tickLine={false} axisLine={{ stroke: C.border }} interval={4}/>
              <YAxis yAxisId="psi" domain={[0, 0.55]} tick={{ fill: C.muted, fontSize: 9, fontFamily: 'Fira Code' }} tickLine={false} axisLine={false}/>
              <YAxis yAxisId="anom" orientation="right" domain={[0, 6]} tick={{ fill: C.muted, fontSize: 9, fontFamily: 'Fira Code' }} tickLine={false} axisLine={false}/>
              <Tooltip content={<CustomTooltip/>}/>
              <ReferenceLine yAxisId="psi" y={0.2} stroke={C.yellowDim} strokeDasharray="4 4" label={{ value: 'threshold', fill: C.yellowDim, fontSize: 9, fontFamily: 'Fira Code' }}/>
              <Line yAxisId="psi"  type="monotone" dataKey="psi"  stroke={C.yellow}  strokeWidth={1.5} dot={false} name="PSI"/>
              <Line yAxisId="anom" type="monotone" dataKey="anom" stroke="#6b9ed4" strokeWidth={1.5} dot={false} name="anomaly %"/>
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* HEATMAP + DISTRIBUTIONS */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>

          {/* FEATURE HEATMAP */}
          <div style={{ background: C.surface, border: `1px solid ${C.borderHi}`, borderRadius: 5, padding: '18px 20px' }}>
            <div style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted, marginBottom: 14 }}>Feature Drift Map · PSI per window</div>
            {/* Time labels */}
            <div style={{ display: 'grid', gridTemplateColumns: '108px repeat(12, 1fr)', gap: 2, marginBottom: 4 }}>
              <div/>
              {batchLabels.map(l => (
                <div key={l} style={{ fontSize: 8, color: C.muted, textAlign: 'center', letterSpacing: '0.04em' }}>{l}</div>
              ))}
            </div>
            {heatRows.map((row, ri) => (
              <div
                key={row.name}
                onClick={() => setSelectedFeature(ri)}
                style={{ display: 'grid', gridTemplateColumns: '108px repeat(12, 1fr)', gap: 2, marginBottom: 2, cursor: 'pointer', opacity: selectedFeature === ri ? 1 : 0.85 }}
              >
                <div style={{ fontSize: 10, color: selectedFeature === ri ? C.yellow : C.muted, letterSpacing: '0.03em', display: 'flex', alignItems: 'center', paddingRight: 8, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {row.name}
                </div>
                {row.vals.map((v, ci) => (
                  <div
                    key={ci}
                    title={`PSI: ${v}`}
                    style={{
                      height: 20, borderRadius: 2,
                      background: heatColor(v),
                      border: selectedFeature === ri && ci === 11 ? `1px solid ${C.yellow}` : '1px solid transparent',
                    }}
                  />
                ))}
              </div>
            ))}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 14, fontSize: 9, color: C.muted }}>
              <span>PSI:</span>
              {[0.05, 0.1, 0.15, 0.2, 0.3, 0.4].map(v => (
                <span key={v} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 2, background: heatColor(v) }}/>
                  {v}
                </span>
              ))}
            </div>
          </div>

          {/* DISTRIBUTION COMPARISON */}
          <div style={{ background: C.surface, border: `1px solid ${C.borderHi}`, borderRadius: 5, padding: '18px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 14 }}>
              <div style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted }}>Distribution Shift</div>
              <div style={{ display: 'flex', gap: 14, fontSize: 10 }}>
                <span style={{ color: C.muted }}>── baseline</span>
                <span style={{ color: C.yellow }}>── current</span>
              </div>
            </div>
            {[
              { feat: 'feature_importance', bm: 0.38, cm: 0.64 },
              { feat: 'avg_session_len',    bm: 0.42, cm: 0.61 },
              { feat: 'request_latency',    bm: 0.35, cm: 0.53 },
            ].map(({ feat, bm, cm }) => (
              <DistPanel key={feat} feature={feat} baselineMean={bm} currentMean={cm}/>
            ))}
          </div>
        </div>

        {/* ALERT LOG */}
        <div style={{ background: C.surface, border: `1px solid ${C.borderHi}`, borderRadius: 5, padding: '16px 20px' }}>
          <div style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted, marginBottom: 14 }}>Alert Log</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                {['Time','Batch','PSI','KS stat','Status'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '4px 12px 8px 0', fontSize: 9, letterSpacing: '0.1em', textTransform: 'uppercase', color: C.muted, fontWeight: 400 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {alerts.map((a, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td style={{ padding: '9px 12px 9px 0', color: C.muted }}>{a.time}</td>
                  <td style={{ padding: '9px 12px 9px 0', color: C.text  }}>{a.batch}</td>
                  <td style={{ padding: '9px 12px 9px 0', color: parseFloat(a.psi) > 0.2 ? C.red : C.yellow }}>{a.psi}</td>
                  <td style={{ padding: '9px 12px 9px 0', color: C.text  }}>{a.ks}</td>
                  <td style={{ padding: '9px 0' }}>
                    <span style={{ fontSize: 9, letterSpacing: '0.08em', padding: '2px 8px', borderRadius: 2, border: `1px solid ${a.level === 'ALERT' ? C.redDim : C.yellowDim}`, color: a.level === 'ALERT' ? C.red : C.yellow }}>
                      {a.level}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

      </div>
    </div>
  );
}
