import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000/api";

const COLORS = {
  bg: "#050B1A", panel: "#0A1628", panelBorder: "#0E2040",
  cyan: "#00D4FF", amber: "#FFB347", green: "#39FF14",
  red: "#FF4757", slate: "#8899BB", slateLight: "#AAB8CC", white: "#E8F0FF",
};

const CLASS_META = {
  "Exoplanet Transit":  { icon: "◎", color: "#00D4FF" },
  "Eclipsing Binary":   { icon: "⊕", color: "#FFB347" },
  "Stellar Blend":      { icon: "◈", color: "#FF4757" },
  "Starspot":           { icon: "✦", color: "#39FF14" },
  "Stellar Variability":{ icon: "∿", color: "#A78BFA" },
};

const PIPELINE_STEPS = [
  { label: "MAST Archive Fetch",    desc: "lightkurve · SPOC 2-min cadence" },
  { label: "Outlier Removal",       desc: "5σ clip · NaN removal" },
  { label: "Flux Normalisation",    desc: "PDCSAP · CBV corrected" },
  { label: "BLS Periodogram",       desc: "Period grid: 0.5 – 27 d" },
  { label: "Transit Detection",     desc: "SNR & phase-fold" },
  { label: "AI Classification",     desc: "Heuristic ensemble classifier" },
];

// ── Canvas: main light curve ──────────────────────────────────────────────
function LightCurveCanvas({ width, height, lcData, hoveredX, setHoveredX }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !lcData) return;
    const { time, flux } = lcData;
    if (!time || time.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = width  * dpr;
    canvas.height = height * dpr;
    canvas.style.width  = width  + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const PAD = { l: 54, r: 14, t: 20, b: 24 };
    const W = width  - PAD.l - PAD.r;
    const H = height - PAD.t - PAD.b;

    const tMin = Math.min(...time), tMax = Math.max(...time);
    const fArr = flux.filter(isFinite);
    const fMin = Math.min(...fArr) - 0.002;
    const fMax = Math.max(...fArr) + 0.002;

    const toX = t => PAD.l + ((t - tMin) / (tMax - tMin)) * W;
    const toY = f => PAD.t + ((fMax - f) / (fMax - fMin)) * H;

    // Grid
    ctx.strokeStyle = "#0E2040"; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = PAD.t + (i / 5) * H;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + W, y); ctx.stroke();
      const label = (fMax - (i / 5) * (fMax - fMin)).toFixed(4);
      ctx.fillStyle = COLORS.slate;
      ctx.font = "9px 'Space Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(label, PAD.l - 4, y + 3);
    }
    for (let i = 0; i <= 6; i++) {
      const x = PAD.l + (i / 6) * W;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + H); ctx.stroke();
      const label = (tMin + (i / 6) * (tMax - tMin)).toFixed(1);
      ctx.fillStyle = COLORS.slate;
      ctx.font = "9px 'Space Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(label, x, PAD.t + H + 14);
    }

    // Plot — downsample if huge
    const stride = Math.max(1, Math.floor(time.length / 3000));
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < time.length; i += stride) {
      if (!isFinite(flux[i])) continue;
      const x = toX(time[i]), y = toY(flux[i]);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.cyan;
    ctx.lineWidth   = 0.8;
    ctx.shadowColor = COLORS.cyan;
    ctx.shadowBlur  = 2;
    ctx.stroke();
    ctx.shadowBlur  = 0;

    // Hover
    if (hoveredX !== null) {
      const t = tMin + ((hoveredX - PAD.l) / W) * (tMax - tMin);
      if (t >= tMin && t <= tMax) {
        let nearIdx = 0, nearDist = Infinity;
        for (let i = 0; i < time.length; i++) {
          const d = Math.abs(time[i] - t);
          if (d < nearDist) { nearDist = d; nearIdx = i; }
        }
        const hx = toX(time[nearIdx]), hy = toY(flux[nearIdx]);
        ctx.strokeStyle = "rgba(255,179,71,0.5)";
        ctx.lineWidth = 1; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(hx, PAD.t); ctx.lineTo(hx, PAD.t+H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PAD.l, hy); ctx.lineTo(PAD.l+W, hy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.amber;
        ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI*2); ctx.fill();
        // Tooltip
        ctx.fillStyle = "#060E1E";
        ctx.fillRect(hx + 6, hy - 22, 110, 18);
        ctx.strokeStyle = "#0E2040"; ctx.lineWidth = 1;
        ctx.strokeRect(hx + 6, hy - 22, 110, 18);
        ctx.fillStyle = COLORS.amber;
        ctx.font = "9px 'Space Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`t=${time[nearIdx].toFixed(3)}  f=${flux[nearIdx].toFixed(5)}`, hx + 10, hy - 9);
      }
    }
  }, [width, height, lcData, hoveredX]);

  return (
    <canvas ref={canvasRef}
      style={{ cursor: "crosshair", display: "block" }}
      onMouseMove={e => setHoveredX(e.clientX - e.currentTarget.getBoundingClientRect().left)}
      onMouseLeave={() => setHoveredX(null)}
    />
  );
}

// ── Canvas: phase-folded ──────────────────────────────────────────────────
function PhaseFoldedCanvas({ width, height, data }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const { phase, flux } = data;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = width  * dpr; canvas.height = height * dpr;
    canvas.style.width = width + "px"; canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const PAD = { l: 8, r: 8, t: 10, b: 20 };
    const W = width - PAD.l - PAD.r, H = height - PAD.t - PAD.b;
    const fArr = flux.filter(isFinite);
    const fMin = Math.min(...fArr) - 0.001;
    const fMax = Math.max(...fArr) + 0.001;
    const toX = p => PAD.l + ((p + 0.5) / 1) * W;
    const toY = f => PAD.t + ((fMax - f) / (fMax - fMin)) * H;

    // Centre dashed line
    ctx.strokeStyle = "rgba(0,212,255,0.3)"; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
    const cx = toX(0);
    ctx.beginPath(); ctx.moveTo(cx, PAD.t); ctx.lineTo(cx, PAD.t+H); ctx.stroke();
    ctx.setLineDash([]);

    const stride = Math.max(1, Math.floor(phase.length / 2000));
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < phase.length; i += stride) {
      if (!isFinite(flux[i])) continue;
      const x = toX(phase[i]), y = toY(flux[i]);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.cyan; ctx.lineWidth = 1;
    ctx.shadowColor = COLORS.cyan; ctx.shadowBlur = 2;
    ctx.stroke(); ctx.shadowBlur = 0;

    ctx.fillStyle = COLORS.slate;
    ctx.font = "9px 'Space Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Phase", PAD.l + W/2, height - 4);
  }, [width, height, data]);
  return <canvas ref={canvasRef} style={{ display: "block" }} />;
}

// ── Canvas: BLS periodogram ───────────────────────────────────────────────
function PeriodogramCanvas({ width, height, data }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const { periods, power, peak_period } = data;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = width  * dpr; canvas.height = height * dpr;
    canvas.style.width = width + "px"; canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const PAD = { l: 8, r: 8, t: 10, b: 20 };
    const W = width - PAD.l - PAD.r, H = height - PAD.t - PAD.b;
    const pMin = Math.min(...periods), pMax = Math.max(...periods);
    const toX = p => PAD.l + ((p - pMin) / (pMax - pMin)) * W;
    const toY = v => PAD.t + (1 - v) * H;

    ctx.beginPath();
    periods.forEach((p, i) => {
      const x = toX(p), y = toY(power[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = COLORS.green; ctx.lineWidth = 1;
    ctx.shadowColor = COLORS.green; ctx.shadowBlur = 2;
    ctx.stroke(); ctx.shadowBlur = 0;

    if (peak_period) {
      const px = toX(peak_period);
      ctx.strokeStyle = "rgba(255,179,71,0.7)"; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(px, PAD.t); ctx.lineTo(px, PAD.t+H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.amber;
      ctx.font = "9px 'Space Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`${peak_period.toFixed(2)}d`, px, PAD.t + 9);
    }

    ctx.fillStyle = COLORS.slate;
    ctx.font = "9px 'Space Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Period [days]", PAD.l + W/2, height - 4);
  }, [width, height, data]);
  return <canvas ref={canvasRef} style={{ display: "block" }} />;
}

// ── Sub-components ────────────────────────────────────────────────────────
function ConfBar({ label, value, color, icon }) {
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
        <span style={{ color:COLORS.slateLight, fontSize:11, fontFamily:"Space Mono, monospace" }}>
          <span style={{ color, marginRight:5 }}>{icon}</span>{label}
        </span>
        <span style={{ color, fontSize:11, fontFamily:"Space Mono, monospace", fontWeight:700 }}>{value.toFixed(1)}%</span>
      </div>
      <div style={{ height:3, background:"#0E2040", borderRadius:2, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${value}%`, background:`linear-gradient(90deg,${color}66,${color})`,
          borderRadius:2, boxShadow:`0 0 5px ${color}`, transition:"width 1s ease" }} />
      </div>
    </div>
  );
}

function ParamCard({ label, value, unit }) {
  return (
    <div style={{ background:"#060E1E", border:"1px solid #0E2040", borderRadius:8, padding:"11px 13px", flex:1, minWidth:100 }}>
      <div style={{ color:COLORS.slate, fontSize:9, fontFamily:"Space Mono, monospace", textTransform:"uppercase", letterSpacing:1, marginBottom:5 }}>{label}</div>
      <div style={{ color:COLORS.cyan, fontSize:17, fontWeight:700, fontFamily:"Space Mono, monospace" }}>{value}</div>
      <div style={{ color:COLORS.slate, fontSize:9, fontFamily:"Space Mono, monospace", marginTop:2 }}>{unit}</div>
    </div>
  );
}

function PipelineStatus({ active, done }) {
  return (
    <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
      <div style={{ fontSize:12, color:COLORS.slate, fontFamily:"Space Mono, monospace", letterSpacing:1, marginBottom:12 }}>PIPELINE STATUS</div>
      {PIPELINE_STEPS.map((s, i) => {
        const status = done ? "done" : active && i <= active ? (i === active ? "active" : "done") : "pending";
        const col = status==="done" ? COLORS.green : status==="active" ? COLORS.cyan : COLORS.slate;
        return (
          <div key={i} style={{ display:"flex", alignItems:"flex-start", gap:10, padding:"8px 0",
            borderBottom:"1px solid #0E2040", opacity:status==="pending"?0.4:1, transition:"opacity 0.3s" }}>
            <div style={{ width:20, height:20, borderRadius:"50%", border:`1.5px solid ${col}`,
              display:"flex", alignItems:"center", justifyContent:"center", color:col, fontSize:10, flexShrink:0,
              boxShadow:status==="active"?`0 0 8px ${col}`:"none",
              animation:status==="active"?"pulse 1.8s ease-in-out infinite":"none" }}>
              {status==="done"?"✓":status==="active"?"▶":"○"}
            </div>
            <div>
              <div style={{ color:COLORS.white, fontSize:12, fontWeight:600, marginBottom:1 }}>{s.label}</div>
              <div style={{ color:COLORS.slate, fontSize:10, fontFamily:"Space Mono, monospace" }}>{s.desc}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [targets, setTargets]       = useState([]);
  const [selected, setSelected]     = useState("L 98-59");
  const [loading, setLoading]       = useState(false);
  const [pipeStep, setPipeStep]     = useState(-1);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [hoveredX, setHoveredX]     = useState(null);
  const [backendOk, setBackendOk]   = useState(null);

  // Track window width for responsive layout
  const [winWidth, setWinWidth] = useState(typeof window !== "undefined" ? window.innerWidth : 1200);
  useEffect(() => {
    const onResize = () => setWinWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const isMobile = winWidth < 500;
  const isTablet = winWidth < 700;

  const containerRef = useRef(null);
  const [canvasWidth, setCanvasWidth] = useState(700);

  // Track phase/periodogram container widths independently
  const smallCanvasRef = useRef(null);
  const [smallCanvasWidth, setSmallCanvasWidth] = useState(300);

  useEffect(() => {
    const obs = new ResizeObserver(([e]) => setCanvasWidth(Math.max(200, e.contentRect.width)));
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const obs = new ResizeObserver(([e]) => setSmallCanvasWidth(Math.max(150, e.contentRect.width)));
    if (smallCanvasRef.current) obs.observe(smallCanvasRef.current);
    return () => obs.disconnect();
  }, [result]);

  // Health check
  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.ok ? setBackendOk(true) : setBackendOk(false))
      .catch(() => setBackendOk(false));

    fetch(`${API}/targets`)
      .then(r => r.json())
      .then(d => setTargets(d.targets || []))
      .catch(() => {});
  }, []);

  const runScan = useCallback(async (target) => {
    setLoading(true);
    setResult(null);
    setError(null);
    setPipeStep(0);

    // Animate pipeline steps
    for (let i = 0; i < PIPELINE_STEPS.length - 1; i++) {
      await new Promise(r => setTimeout(r, 700));
      setPipeStep(i + 1);
    }

    try {
      const res  = await fetch(`${API}/analyse?target=${encodeURIComponent(target)}`);
      const json = await res.json();
      if (json.status === "ok") {
        setResult(json.data);
      } else {
        setError(json.message || "Unknown error");
      }
    } catch (e) {
      setError("Cannot reach backend. Is the Flask server running on port 5000?");
    }

    setLoading(false);
    setPipeStep(-1);
  }, []);

  const cls   = result?.classification;
  const tp    = result?.transit_params;
  const dq    = result?.data_quality;
  const topMeta = cls ? (CLASS_META[cls.top_class] || { icon:"?", color:COLORS.cyan }) : null;

  return (
    <div style={{ minHeight:"100vh", width:"100vw", background:COLORS.bg, color:COLORS.white,
      fontFamily:"Inter, system-ui, sans-serif", paddingBottom:40, overflowX:"hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        html,body{margin:0;padding:0;width:100%;overflow-x:hidden}
        @keyframes pulse{0%,100%{box-shadow:0 0 6px #00D4FF}50%{box-shadow:0 0 18px #00D4FF}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
        ::-webkit-scrollbar{width:6px}
        ::-webkit-scrollbar-track{background:#050B1A}
        ::-webkit-scrollbar-thumb{background:#0E2040;border-radius:3px}
        html{font-size:16px}
        @media(max-width:600px){html{font-size:14px}}
      `}</style>

      {/* Header */}
      <div style={{ borderBottom:"1px solid #0E2040", background:"linear-gradient(180deg,#080F20,#050B1A)", padding: isMobile ? "0 12px" : "0 32px" }}>
        <div style={{ width:"100%", display:"flex", alignItems:"center", justifyContent:"space-between", height: isMobile ? "auto" : 60, flexWrap:"wrap", gap: isMobile ? 8 : 0, padding: isMobile ? "10px 0" : 0 }}>
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <div style={{ width:32, height:32, borderRadius:"50%",
              background:"radial-gradient(circle at 35% 35%, #00D4FF33, #050B1A)",
              border:"1.5px solid #00D4FF55", display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>◎</div>
            <div>
              <div style={{ fontSize:15, fontWeight:700, letterSpacing:0.5 }}>EXODETECT</div>
              <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono, monospace", letterSpacing:1 }}>AI-Enabled Exoplanet Detection</div>
            </div>
          </div>

          <div style={{ display:"flex", gap:8, alignItems:"center" }}>
            {/* Backend status */}
            <div style={{ display:"flex", alignItems:"center", gap:5, padding:"4px 10px",
              background:"#060E1E", border:"1px solid #0E2040", borderRadius:6 }}>
              <div style={{ width:7, height:7, borderRadius:"50%",
                background: backendOk === null ? COLORS.amber : backendOk ? COLORS.green : COLORS.red,
                boxShadow: `0 0 5px ${backendOk === null ? COLORS.amber : backendOk ? COLORS.green : COLORS.red}`,
                animation: "blink 2s ease-in-out infinite" }} />
              <span style={{ fontSize:10, fontFamily:"Space Mono, monospace",
                color: backendOk ? COLORS.green : COLORS.red }}>
                {backendOk === null ? "CONNECTING…" : backendOk ? "BACKEND LIVE" : "BACKEND OFFLINE"}
              </span>
            </div>

            <button onClick={() => runScan(selected)} disabled={loading}
              style={{ background: loading ? "#0E2040" : "linear-gradient(135deg,#00D4FF22,#00D4FF11)",
                border:`1px solid ${loading ? "#0E2040" : COLORS.cyan}`,
                color: loading ? COLORS.slate : COLORS.cyan,
                borderRadius:6, padding:"6px 16px", fontSize:12,
                fontFamily:"Space Mono, monospace", cursor: loading ? "not-allowed" : "pointer",
                fontWeight:700, letterSpacing:1 }}>
              {loading ? "FETCHING…" : "RUN SCAN"}
            </button>
          </div>
        </div>
      </div>

      <div style={{ width:"100%", padding: isMobile ? "12px 12px 0" : "20px 32px 0" }}>

        {/* Target selector */}
        <div style={{ display:"flex", gap:8, marginBottom:18, flexWrap:"wrap", alignItems:"center" }}>
          {targets.length > 0 ? targets.map(t => (
            <button key={t.name} onClick={() => { setSelected(t.name); }}
              title={t.note}
              style={{ background: selected===t.name ? "#00D4FF18" : "transparent",
                border:`1px solid ${selected===t.name ? COLORS.cyan : "#0E2040"}`,
                color: selected===t.name ? COLORS.cyan : COLORS.slate,
                borderRadius:20, padding:"4px 14px", fontSize:11,
                fontFamily:"Space Mono, monospace", cursor:"pointer", transition:"all 0.2s" }}>
              {t.name}
            </button>
          )) : (
            ["L 98-59","TOI-700","WASP-18","TIC 286923464","HD 21749"].map(n => (
              <button key={n} onClick={() => setSelected(n)}
                style={{ background: selected===n?"#00D4FF18":"transparent",
                  border:`1px solid ${selected===n?COLORS.cyan:"#0E2040"}`,
                  color: selected===n?COLORS.cyan:COLORS.slate,
                  borderRadius:20, padding:"4px 14px", fontSize:11,
                  fontFamily:"Space Mono, monospace", cursor:"pointer" }}>
                {n}
              </button>
            ))
          )}
          {result && (
            <span style={{ marginLeft:"auto", color:COLORS.slate, fontSize:11, fontFamily:"Space Mono, monospace" }}>
              Sector {result.sector} · {result.n_cadences.toLocaleString()} cadences · {result.exptime_sec}s exp
            </span>
          )}
        </div>

        {/* Error banner */}
        {error && (
          <div style={{ background:"#FF475722", border:"1px solid #FF475755", borderRadius:8,
            padding:"10px 16px", marginBottom:16, color:COLORS.red, fontSize:12, fontFamily:"Space Mono, monospace" }}>
            ⚠ {error}
          </div>
        )}

        {/* No-data prompt */}
        {!result && !loading && !error && (
          <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12,
            padding:"40px 20px", textAlign:"center", marginBottom:16 }}>
            <div style={{ fontSize:36, marginBottom:12 }}>◎</div>
            <div style={{ color:COLORS.white, fontSize:15, fontWeight:600, marginBottom:6 }}>Ready to scan TESS data</div>
            <div style={{ color:COLORS.slate, fontSize:12, fontFamily:"Space Mono, monospace", marginBottom:20 }}>
              Select a target above, then click RUN SCAN to fetch live data from the MAST archive
            </div>
            <button onClick={() => runScan(selected)}
              style={{ background:"linear-gradient(135deg,#00D4FF22,#00D4FF11)",
                border:`1px solid ${COLORS.cyan}`, color:COLORS.cyan,
                borderRadius:8, padding:"10px 28px", fontSize:13,
                fontFamily:"Space Mono, monospace", cursor:"pointer", fontWeight:700 }}>
              RUN SCAN → {selected}
            </button>
          </div>
        )}

        {/* Main layout */}
        {(result || loading) && (
          <div style={{ display:"grid", gridTemplateColumns: isTablet ? "1fr" : "1fr 290px", gap:16, alignItems:"start" }}>

            {/* Left column */}
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

              {/* Light curve */}
              <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, overflow:"hidden" }}>
                <div style={{ padding:"12px 16px", borderBottom:"1px solid #0E2040",
                  display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                  <div>
                    <div style={{ fontSize:13, fontWeight:600 }}>{result?.target || selected}</div>
                    <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono, monospace", marginTop:1 }}>
                      {result ? `TESS Sector ${result.sector} · PDCSAP Flux` : "Fetching from MAST…"}
                    </div>
                  </div>
                  {loading && <div style={{ color:COLORS.cyan, fontSize:11, fontFamily:"Space Mono, monospace",
                    animation:"blink 1.2s ease-in-out infinite" }}>● DOWNLOADING</div>}
                </div>
                <div style={{ padding:"4px 16px 0", display:"flex", justifyContent:"space-between" }}>
                  <span style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono, monospace" }}>← Normalized Flux</span>
                  <span style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono, monospace" }}>Time [BTJD] →</span>
                </div>
                <div ref={containerRef} style={{ padding:"0 16px 14px" }}>
                  {result?.light_curve ? (
                    <LightCurveCanvas width={canvasWidth-32} height={220}
                      lcData={result.light_curve} hoveredX={hoveredX} setHoveredX={setHoveredX} />
                  ) : (
                    <div style={{ height:220, display:"flex", alignItems:"center", justifyContent:"center",
                      color:COLORS.slate, fontSize:12, fontFamily:"Space Mono, monospace" }}>
                      {loading ? "Fetching light curve…" : "—"}
                    </div>
                  )}
                </div>
                {result && tp && (
                  <div style={{ borderTop:"1px solid #0E2040", padding:"10px 16px",
                    display:"flex", gap:20, background:"#060E1E", flexWrap:"wrap" }}>
                    {[
                      { l:"PERIOD",   v:`${tp.period_days} d` },
                      { l:"DEPTH",    v:`${tp.depth_pct}%` },
                      { l:"DURATION", v:`${tp.duration_hours} hr` },
                      { l:"SNR",      v:`${tp.snr}σ` },
                      { l:"TRANSITS", v:`${tp.n_transits}` },
                    ].map(({l,v}) => (
                      <div key={l}>
                        <div style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono, monospace", letterSpacing:1 }}>{l}</div>
                        <div style={{ fontSize:15, fontWeight:700, color:COLORS.cyan, fontFamily:"Space Mono, monospace", marginTop:2 }}>{v}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Phase-fold + periodogram */}
              <div style={{ display:"grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap:16 }}>
                {["Phase-Folded Transit","BLS Periodogram"].map((title, idx) => (
                  <div key={title} ref={idx === 0 ? smallCanvasRef : undefined} style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, overflow:"hidden" }}>
                    <div style={{ padding:"9px 13px", borderBottom:"1px solid #0E2040",
                      fontSize:10, color:COLORS.slate, fontFamily:"Space Mono, monospace" }}>{title}</div>
                    <div style={{ padding:8 }}>
                      {result ? (
                        idx === 0
                          ? <PhaseFoldedCanvas width={smallCanvasWidth - 16} height={140} data={result.phase_folded} />
                          : <PeriodogramCanvas width={smallCanvasWidth - 16} height={140} data={result.bls} />
                      ) : (
                        <div style={{ height:140, display:"flex", alignItems:"center", justifyContent:"center",
                          color:COLORS.slate, fontSize:11, fontFamily:"Space Mono, monospace" }}>
                          {loading ? "Computing…" : "—"}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Parameter cards */}
              {result && tp && (
                <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                  <div style={{ fontSize:11, color:COLORS.slate, fontFamily:"Space Mono, monospace", letterSpacing:1, marginBottom:12 }}>
                    TRANSIT PARAMETER ESTIMATES · BLS FIT
                  </div>
                  <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
                    <ParamCard label="Orbital Period"  value={`${tp.period_days}`}    unit={`days ±${tp.period_err}`} />
                    <ParamCard label="Transit Depth"   value={`${tp.depth_pct}%`}     unit={`Rp/R★ = ${tp.rp_rs}`} />
                    <ParamCard label="Duration"        value={`${tp.duration_hours}`} unit={`hours ±${tp.duration_err}`} />
                    <ParamCard label="SNR"             value={`${tp.snr}σ`}           unit="significance" />
                    <ParamCard label="N Transits"      value={`${tp.n_transits}`}     unit="observed" />
                    <ParamCard label="Epoch T₀"        value={tp.t0_bjd}              unit="BTJD" />
                  </div>
                </div>
              )}
            </div>

            {/* Right column */}
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
              <PipelineStatus active={pipeStep} done={!!result} />

              {/* Classification */}
              <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                <div style={{ fontSize:12, color:COLORS.slate, fontFamily:"Space Mono, monospace", letterSpacing:1, marginBottom:4 }}>AI CLASSIFICATION</div>
                <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono, monospace", marginBottom:14 }}>Heuristic Ensemble · Real BLS Features</div>

                {cls ? (
                  <>
                    <div style={{ background:`${topMeta.color}10`, border:`1px solid ${topMeta.color}44`,
                      borderRadius:8, padding:"10px 12px", marginBottom:14,
                      display:"flex", gap:10, alignItems:"center" }}>
                      <div style={{ fontSize:22, color:topMeta.color }}>{topMeta.icon}</div>
                      <div>
                        <div style={{ color:topMeta.color, fontSize:13, fontWeight:700 }}>{cls.top_class}</div>
                        <div style={{ color:COLORS.slate, fontSize:10, fontFamily:"Space Mono, monospace" }}>Confidence: {cls.confidence}%</div>
                      </div>
                    </div>
                    {Object.entries(cls.probabilities).map(([label, val]) => {
                      const m = CLASS_META[label] || { icon:"?", color:COLORS.slate };
                      return <ConfBar key={label} label={label} value={val} color={m.color} icon={m.icon} />;
                    })}
                  </>
                ) : (
                  <div style={{ color:COLORS.slate, fontSize:12, fontFamily:"Space Mono, monospace",
                    textAlign:"center", padding:"20px 0" }}>
                    {loading ? "Classifying…" : "Run scan to classify"}
                  </div>
                )}
              </div>

              {/* Data quality */}
              {dq && (
                <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                  <div style={{ fontSize:12, color:COLORS.slate, fontFamily:"Space Mono, monospace", letterSpacing:1, marginBottom:12 }}>DATA QUALITY</div>
                  {[
                    { label:"Completeness",  value:dq.completeness_pct, unit:"%",   color:COLORS.green, max:100 },
                    { label:"CDPP Noise",    value:dq.cdpp_ppm,         unit:" ppm",color:COLORS.cyan,  max:1000 },
                    { label:"Sys. Noise",    value:dq.sys_noise_ppm,    unit:" ppm",color:COLORS.amber, max:200 },
                  ].map(({ label, value, unit, color, max }) => (
                    <div key={label} style={{ marginBottom:10 }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                        <span style={{ color:COLORS.slate, fontSize:11, fontFamily:"Space Mono, monospace" }}>{label}</span>
                        <span style={{ color, fontSize:11, fontFamily:"Space Mono, monospace" }}>{value}{unit}</span>
                      </div>
                      <div style={{ height:3, background:"#0E2040", borderRadius:2, overflow:"hidden" }}>
                        <div style={{ height:"100%", width:`${Math.min(100, (value/max)*100)}%`, background:color, borderRadius:2, opacity:0.8 }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}