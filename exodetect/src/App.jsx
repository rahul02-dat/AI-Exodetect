import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000/api";

const COLORS = {
  bg: "#050B1A", panel: "#0A1628", panelBorder: "#0E2040",
  cyan: "#00D4FF", amber: "#FFB347", green: "#39FF14",
  red: "#FF4757", slate: "#8899BB", slateLight: "#AAB8CC", white: "#E8F0FF",
  purple: "#A78BFA",
};

const CLASS_META = {
  "Exoplanet Transit":   { icon: "◎", color: "#00D4FF" },
  "Eclipsing Binary":    { icon: "⊕", color: "#FFB347" },
  "Stellar Blend":       { icon: "◈", color: "#FF4757" },
  "Starspot":            { icon: "✦", color: "#39FF14" },
  "Stellar Variability": { icon: "∿", color: "#A78BFA" },
};

const MCMC_PARAM_LABELS = {
  "Period (d)":   { unit: "days",   color: "#00D4FF" },
  "T₀ (BTJD)":   { unit: "BTJD",   color: "#00D4FF" },
  "Rp/R★":       { unit: "",        color: "#A78BFA" },
  "a/R★":        { unit: "",        color: "#A78BFA" },
  "Inc (°)":     { unit: "degrees", color: "#FFB347" },
  "u₁":          { unit: "",        color: "#8899BB" },
  "u₂":          { unit: "",        color: "#8899BB" },
  "Depth (ppm)": { unit: "ppm",     color: "#39FF14" },
  "Rp (R⊕)":    { unit: "R⊕",      color: "#39FF14" },
};

// ── Canvas: light curve with optional model overlay ────────────────────────
function LightCurveCanvas({ width, height, lcData, modelFlux, hoveredX, setHoveredX }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !lcData) return;
    const { time, flux } = lcData;
    if (!time || time.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr;
    canvas.style.width = width + "px"; canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    const PAD = { l: 54, r: 14, t: 20, b: 24 };
    const W = width - PAD.l - PAD.r, H = height - PAD.t - PAD.b;
    const tMin = Math.min(...time), tMax = Math.max(...time);
    const fArr = flux.filter(isFinite);
    const fMin = Math.min(...fArr) - 0.002, fMax = Math.max(...fArr) + 0.002;
    const toX = t => PAD.l + ((t - tMin) / (tMax - tMin)) * W;
    const toY = f => PAD.t + ((fMax - f) / (fMax - fMin)) * H;

    // Grid
    ctx.strokeStyle = "#0E2040"; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = PAD.t + (i / 5) * H;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + W, y); ctx.stroke();
      ctx.fillStyle = COLORS.slate; ctx.font = "9px 'Space Mono',monospace";
      ctx.textAlign = "right";
      ctx.fillText((fMax - (i / 5) * (fMax - fMin)).toFixed(4), PAD.l - 4, y + 3);
    }
    for (let i = 0; i <= 6; i++) {
      const x = PAD.l + (i / 6) * W;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + H); ctx.stroke();
      ctx.fillStyle = COLORS.slate; ctx.textAlign = "center";
      ctx.fillText((tMin + (i / 6) * (tMax - tMin)).toFixed(1), x, PAD.t + H + 14);
    }

    // Raw flux
    const stride = Math.max(1, Math.floor(time.length / 3000));
    ctx.beginPath(); let started = false;
    for (let i = 0; i < time.length; i += stride) {
      if (!isFinite(flux[i])) continue;
      const x = toX(time[i]), y = toY(flux[i]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.cyan; ctx.lineWidth = 0.8;
    ctx.shadowColor = COLORS.cyan; ctx.shadowBlur = 2;
    ctx.stroke(); ctx.shadowBlur = 0;

    // MCMC model overlay
    if (modelFlux && modelFlux.length === time.length) {
      ctx.beginPath(); started = false;
      for (let i = 0; i < time.length; i += stride) {
        const x = toX(time[i]), y = toY(modelFlux[i]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = COLORS.amber; ctx.lineWidth = 2.0;
      ctx.shadowColor = COLORS.amber; ctx.shadowBlur = 6;
      ctx.stroke(); ctx.shadowBlur = 0;
    }

    // Hover
    if (hoveredX !== null) {
      const t = tMin + ((hoveredX - PAD.l) / W) * (tMax - tMin);
      if (t >= tMin && t <= tMax) {
        let ni = 0, nd = Infinity;
        for (let i = 0; i < time.length; i++) {
          const d = Math.abs(time[i] - t); if (d < nd) { nd = d; ni = i; }
        }
        const hx = toX(time[ni]), hy = toY(flux[ni]);
        ctx.strokeStyle = "rgba(255,179,71,0.5)"; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(hx, PAD.t); ctx.lineTo(hx, PAD.t+H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PAD.l, hy); ctx.lineTo(PAD.l+W, hy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.amber;
        ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI*2); ctx.fill();
      }
    }
  }, [width, height, lcData, modelFlux, hoveredX]);

  return (
    <canvas ref={canvasRef}
      style={{ cursor:"crosshair", display:"block" }}
      onMouseMove={e => setHoveredX(e.clientX - e.currentTarget.getBoundingClientRect().left)}
      onMouseLeave={() => setHoveredX(null)} />
  );
}

function PhaseFoldedCanvas({ width, height, data }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !data) return;
    const { phase, flux } = data;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width*dpr; canvas.height = height*dpr;
    canvas.style.width=width+"px"; canvas.style.height=height+"px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0,0,width,height);
    const PAD={l:8,r:8,t:10,b:20}, W=width-PAD.l-PAD.r, H=height-PAD.t-PAD.b;
    const fArr=flux.filter(isFinite), fMin=Math.min(...fArr)-0.001, fMax=Math.max(...fArr)+0.001;
    const toX=p=>PAD.l+((p+0.5)/1)*W, toY=f=>PAD.t+((fMax-f)/(fMax-fMin))*H;
    const cx=toX(0);
    ctx.strokeStyle="rgba(0,212,255,0.3)"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(cx,PAD.t); ctx.lineTo(cx,PAD.t+H); ctx.stroke();
    ctx.setLineDash([]);
    const stride=Math.max(1,Math.floor(phase.length/2000));
    ctx.beginPath(); let st=false;
    for(let i=0;i<phase.length;i+=stride){
      if(!isFinite(flux[i]))continue;
      const x=toX(phase[i]),y=toY(flux[i]);
      if(!st){ctx.moveTo(x,y);st=true;}else ctx.lineTo(x,y);
    }
    ctx.strokeStyle=COLORS.cyan; ctx.lineWidth=1;
    ctx.shadowColor=COLORS.cyan; ctx.shadowBlur=2; ctx.stroke(); ctx.shadowBlur=0;
    ctx.fillStyle=COLORS.slate; ctx.font="9px 'Space Mono',monospace";
    ctx.textAlign="center"; ctx.fillText("Phase",PAD.l+W/2,height-4);
  },[width,height,data]);
  return <canvas ref={ref} style={{display:"block"}} />;
}

function PeriodogramCanvas({ width, height, data }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !data) return;
    const { periods, power, peak_period } = data;
    const dpr = window.devicePixelRatio || 1;
    canvas.width=width*dpr; canvas.height=height*dpr;
    canvas.style.width=width+"px"; canvas.style.height=height+"px";
    const ctx=canvas.getContext("2d");
    ctx.scale(dpr,dpr);
    ctx.clearRect(0,0,width,height);
    const PAD={l:8,r:8,t:10,b:20}, W=width-PAD.l-PAD.r, H=height-PAD.t-PAD.b;
    const pMin=Math.min(...periods), pMax=Math.max(...periods);
    const toX=p=>PAD.l+((p-pMin)/(pMax-pMin))*W, toY=v=>PAD.t+(1-v)*H;
    ctx.beginPath();
    periods.forEach((p,i)=>{
      const x=toX(p),y=toY(power[i]);
      if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.strokeStyle=COLORS.green; ctx.lineWidth=1;
    ctx.shadowColor=COLORS.green; ctx.shadowBlur=2; ctx.stroke(); ctx.shadowBlur=0;
    if(peak_period){
      const px=toX(peak_period);
      ctx.strokeStyle="rgba(255,179,71,0.7)"; ctx.lineWidth=1.5; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(px,PAD.t); ctx.lineTo(px,PAD.t+H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle=COLORS.amber; ctx.font="9px 'Space Mono',monospace";
      ctx.textAlign="center"; ctx.fillText(`${peak_period.toFixed(2)}d`,px,PAD.t+9);
    }
    ctx.fillStyle=COLORS.slate; ctx.font="9px 'Space Mono',monospace";
    ctx.textAlign="center"; ctx.fillText("Period [days]",PAD.l+W/2,height-4);
  },[width,height,data]);
  return <canvas ref={ref} style={{display:"block"}} />;
}

// ── Sub-components ─────────────────────────────────────────────────────────
function ConfBar({ label, value, color, icon }) {
  return (
    <div style={{ marginBottom:9 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
        <span style={{ color:COLORS.slateLight, fontSize:11, fontFamily:"Space Mono,monospace" }}>
          <span style={{ color, marginRight:5 }}>{icon}</span>{label}
        </span>
        <span style={{ color, fontSize:11, fontFamily:"Space Mono,monospace", fontWeight:700 }}>{value.toFixed(1)}%</span>
      </div>
      <div style={{ height:3, background:"#0E2040", borderRadius:2, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${value}%`,
          background:`linear-gradient(90deg,${color}66,${color})`,
          borderRadius:2, boxShadow:`0 0 5px ${color}`, transition:"width 1s ease" }} />
      </div>
    </div>
  );
}

function ParamCard({ label, value, unit, highlight }) {
  return (
    <div style={{ background:"#060E1E", border:`1px solid ${highlight?"#00D4FF44":"#0E2040"}`,
      borderRadius:8, padding:"11px 13px", flex:1, minWidth:100,
      boxShadow: highlight ? "0 0 12px #00D4FF22" : "none" }}>
      <div style={{ color:COLORS.slate, fontSize:9, fontFamily:"Space Mono,monospace",
        textTransform:"uppercase", letterSpacing:1, marginBottom:5 }}>{label}</div>
      <div style={{ color: highlight ? COLORS.cyan : COLORS.white,
        fontSize:17, fontWeight:700, fontFamily:"Space Mono,monospace" }}>{value}</div>
      <div style={{ color:COLORS.slate, fontSize:9, fontFamily:"Space Mono,monospace", marginTop:2 }}>{unit}</div>
    </div>
  );
}

function MCMCParamRow({ name, data }) {
  const meta  = MCMC_PARAM_LABELS[name] || { unit:"", color:COLORS.cyan };
  const { median, err_low, err_high } = data;
  return (
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
      padding:"8px 0", borderBottom:"1px solid #0E2040" }}>
      <span style={{ color:COLORS.slate, fontSize:11, fontFamily:"Space Mono,monospace", minWidth:120 }}>{name}</span>
      <div style={{ textAlign:"right" }}>
        <span style={{ color:meta.color, fontSize:13, fontWeight:700, fontFamily:"Space Mono,monospace" }}>
          {median.toFixed(4)}
        </span>
        <span style={{ color:COLORS.slate, fontSize:10, fontFamily:"Space Mono,monospace", marginLeft:6 }}>
          +{err_high.toFixed(4)} / -{err_low.toFixed(4)}
        </span>
        {meta.unit && (
          <span style={{ color:"#4A5568", fontSize:9, fontFamily:"Space Mono,monospace", marginLeft:4 }}>
            {meta.unit}
          </span>
        )}
      </div>
    </div>
  );
}

// ── MCMC progress bar ──────────────────────────────────────────────────────
function MCMCProgress({ step, pct }) {
  return (
    <div>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
        <span style={{ color:COLORS.slateLight, fontSize:11, fontFamily:"Space Mono,monospace" }}>{step}</span>
        <span style={{ color:COLORS.purple, fontSize:11, fontFamily:"Space Mono,monospace", fontWeight:700 }}>{pct}%</span>
      </div>
      <div style={{ height:4, background:"#0E2040", borderRadius:2, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${pct}%`,
          background:`linear-gradient(90deg,${COLORS.purple}66,${COLORS.purple})`,
          borderRadius:2, boxShadow:`0 0 8px ${COLORS.purple}`,
          transition:"width 0.4s ease" }} />
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [targets, setTargets]     = useState([]);
  const [selected, setSelected]   = useState("L 98-59");
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [hoveredX, setHoveredX]   = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const [activeTab, setActiveTab] = useState("lightcurve"); // lightcurve | mcmc

  // MCMC state
  const [mcmcJobId, setMcmcJobId]   = useState(null);
  const [mcmcState, setMcmcState]   = useState(null);  // PENDING|PROGRESS|SUCCESS|FAILURE
  const [mcmcStep, setMcmcStep]     = useState("");
  const [mcmcPct, setMcmcPct]       = useState(0);
  const [mcmcResult, setMcmcResult] = useState(null);
  const [mcmcLoading, setMcmcLoading] = useState(false);
  const pollRef = useRef(null);

  const containerRef = useRef(null);
  const [canvasWidth, setCanvasWidth] = useState(700);

  useEffect(() => {
    const obs = new ResizeObserver(([e]) => setCanvasWidth(Math.max(300, e.contentRect.width)));
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.ok ? setBackendOk(true) : setBackendOk(false)).catch(() => setBackendOk(false));
    fetch(`${API}/targets`).then(r => r.json()).then(d => setTargets(d.targets || [])).catch(() => {});
  }, []);

  // Poll MCMC status
  const pollMCMC = useCallback((jobId) => {
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/mcmc/status?job_id=${jobId}`);
        const d = await r.json();
        setMcmcState(d.state);
        setMcmcStep(d.step || "");
        setMcmcPct(d.pct || 0);

        if (d.state === "SUCCESS") {
          clearInterval(pollRef.current);
          // Fetch full result
          const r2  = await fetch(`${API}/mcmc/result?job_id=${jobId}`);
          const d2  = await r2.json();
          if (d2.status === "ok") {
            setMcmcResult(d2.data);
          }
          setMcmcLoading(false);
        }
        if (d.state === "FAILURE") {
          clearInterval(pollRef.current);
          setMcmcLoading(false);
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setMcmcLoading(false);
      }
    }, 1500);
  }, []);

  const runScan = useCallback(async (target) => {
    setLoading(true); setResult(null); setError(null);
    setMcmcResult(null); setMcmcJobId(null); setMcmcState(null);
    try {
      const res  = await fetch(`${API}/analyse?target=${encodeURIComponent(target)}`);
      const json = await res.json();
      if (json.status === "ok") setResult(json.data);
      else setError(json.message || "Unknown error");
    } catch {
      setError("Cannot reach backend on port 5000");
    }
    setLoading(false);
  }, []);

  const startMCMC = useCallback(async () => {
    if (!selected) return;
    setMcmcLoading(true); setMcmcResult(null);
    setMcmcState("PENDING"); setMcmcStep("Queuing job…"); setMcmcPct(0);
    setActiveTab("mcmc");
    try {
      const r = await fetch(`${API}/mcmc/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: selected, n_walkers: 32, n_steps: 2000, n_burn: 500 }),
      });
      const d = await r.json();
      setMcmcJobId(d.job_id);
      pollMCMC(d.job_id);
    } catch {
      setMcmcLoading(false);
      setMcmcState("FAILURE");
    }
  }, [selected, pollMCMC]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const cls     = result?.classification;
  const tp      = result?.transit_params;
  const dq      = result?.data_quality;
  const topMeta = cls ? (CLASS_META[cls.top_class] || { icon:"?", color:COLORS.cyan }) : null;

  const TABS = [
    { id:"lightcurve", label:"Light Curve" },
    { id:"mcmc",       label:"MCMC Fit" + (mcmcState === "SUCCESS" ? " ✓" : "") },
  ];

  return (
    <div style={{ minHeight:"100vh", background:COLORS.bg, color:COLORS.white,
      fontFamily:"Inter,system-ui,sans-serif", paddingBottom:40 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        @keyframes pulse{0%,100%{box-shadow:0 0 6px #00D4FF}50%{box-shadow:0 0 18px #00D4FF}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
        ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#050B1A}
        ::-webkit-scrollbar-thumb{background:#0E2040;border-radius:3px}
      `}</style>

      {/* Header */}
      <div style={{ borderBottom:"1px solid #0E2040", background:"linear-gradient(180deg,#080F20,#050B1A)", padding:"0 24px" }}>
        <div style={{ maxWidth:1200, margin:"0 auto", display:"flex", alignItems:"center", justifyContent:"space-between", height:60 }}>
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <div style={{ width:32, height:32, borderRadius:"50%",
              background:"radial-gradient(circle at 35% 35%,#00D4FF33,#050B1A)",
              border:"1.5px solid #00D4FF55", display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>◎</div>
            <div>
              <div style={{ fontSize:15, fontWeight:700, letterSpacing:0.5 }}>EXODETECT</div>
              <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1 }}>
                PHASE 2 · CNN + MCMC · APPLE SILICON
              </div>
            </div>
          </div>
          <div style={{ display:"flex", gap:8, alignItems:"center" }}>
            <div style={{ display:"flex", alignItems:"center", gap:5, padding:"4px 10px",
              background:"#060E1E", border:"1px solid #0E2040", borderRadius:6 }}>
              <div style={{ width:7, height:7, borderRadius:"50%",
                background:backendOk===null?COLORS.amber:backendOk?COLORS.green:COLORS.red,
                animation:"blink 2s ease-in-out infinite" }} />
              <span style={{ fontSize:10, fontFamily:"Space Mono,monospace",
                color:backendOk?COLORS.green:COLORS.red }}>
                {backendOk===null?"CONNECTING…":backendOk?"BACKEND LIVE":"OFFLINE"}
              </span>
            </div>
            <button onClick={() => runScan(selected)} disabled={loading}
              style={{ background:loading?"#0E2040":"linear-gradient(135deg,#00D4FF22,#00D4FF11)",
                border:`1px solid ${loading?"#0E2040":COLORS.cyan}`,
                color:loading?COLORS.slate:COLORS.cyan, borderRadius:6, padding:"6px 16px",
                fontSize:12, fontFamily:"Space Mono,monospace", cursor:loading?"not-allowed":"pointer",
                fontWeight:700, letterSpacing:1 }}>
              {loading?"FETCHING…":"RUN SCAN"}
            </button>
            {result && (
              <button onClick={startMCMC} disabled={mcmcLoading}
                style={{ background:mcmcLoading?"#0E2040":"linear-gradient(135deg,#A78BFA22,#A78BFA11)",
                  border:`1px solid ${mcmcLoading?"#0E2040":COLORS.purple}`,
                  color:mcmcLoading?COLORS.slate:COLORS.purple, borderRadius:6, padding:"6px 16px",
                  fontSize:12, fontFamily:"Space Mono,monospace", cursor:mcmcLoading?"not-allowed":"pointer",
                  fontWeight:700, letterSpacing:1 }}>
                {mcmcLoading?"MCMC RUNNING…":"RUN MCMC FIT"}
              </button>
            )}
          </div>
        </div>
      </div>

      <div style={{ maxWidth:1200, margin:"0 auto", padding:"20px 24px 0" }}>

        {/* Target selector */}
        <div style={{ display:"flex", gap:8, marginBottom:18, flexWrap:"wrap", alignItems:"center" }}>
          {(targets.length>0 ? targets.map(t=>t.name) : ["L 98-59","TOI-700","WASP-18","TIC 286923464","HD 21749"]).map(n => (
            <button key={n} onClick={() => setSelected(n)}
              style={{ background:selected===n?"#00D4FF18":"transparent",
                border:`1px solid ${selected===n?COLORS.cyan:"#0E2040"}`,
                color:selected===n?COLORS.cyan:COLORS.slate,
                borderRadius:20, padding:"4px 14px", fontSize:11,
                fontFamily:"Space Mono,monospace", cursor:"pointer", transition:"all 0.2s" }}>
              {n}
            </button>
          ))}
          {result && (
            <span style={{ marginLeft:"auto", color:COLORS.slate, fontSize:11, fontFamily:"Space Mono,monospace" }}>
              Sector {result.sector} · {result.n_cadences?.toLocaleString()} cadences
            </span>
          )}
        </div>

        {error && (
          <div style={{ background:"#FF475722", border:"1px solid #FF475755", borderRadius:8,
            padding:"10px 16px", marginBottom:16, color:COLORS.red, fontSize:12, fontFamily:"Space Mono,monospace" }}>
            ⚠ {error}
          </div>
        )}

        {!result && !loading && !error && (
          <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12,
            padding:"40px 20px", textAlign:"center", marginBottom:16 }}>
            <div style={{ fontSize:36, marginBottom:12 }}>◎</div>
            <div style={{ color:COLORS.white, fontSize:15, fontWeight:600, marginBottom:6 }}>ExoDetect Phase 2</div>
            <div style={{ color:COLORS.slate, fontSize:12, fontFamily:"Space Mono,monospace", marginBottom:20 }}>
              CNN classifier + MCMC Bayesian transit fitting · batman + emcee
            </div>
            <button onClick={() => runScan(selected)}
              style={{ background:"linear-gradient(135deg,#00D4FF22,#00D4FF11)",
                border:`1px solid ${COLORS.cyan}`, color:COLORS.cyan,
                borderRadius:8, padding:"10px 28px", fontSize:13,
                fontFamily:"Space Mono,monospace", cursor:"pointer", fontWeight:700 }}>
              RUN SCAN → {selected}
            </button>
          </div>
        )}

        {(result || loading) && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 290px", gap:16, alignItems:"start" }}>

            {/* Left column */}
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

              {/* Tab bar */}
              <div style={{ display:"flex", gap:2, background:"#060E1E",
                border:"1px solid #0E2040", borderRadius:10, padding:4, width:"fit-content" }}>
                {TABS.map(tab => (
                  <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                    style={{ background:activeTab===tab.id?"#0A1628":"transparent",
                      border:activeTab===tab.id?"1px solid #0E2040":"1px solid transparent",
                      color:activeTab===tab.id?COLORS.white:COLORS.slate,
                      borderRadius:7, padding:"6px 16px", fontSize:12, fontFamily:"Space Mono,monospace",
                      cursor:"pointer", transition:"all 0.2s", fontWeight:activeTab===tab.id?600:400 }}>
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* ── TAB: Light Curve ── */}
              {activeTab === "lightcurve" && (
                <>
                  <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, overflow:"hidden" }}>
                    <div style={{ padding:"12px 16px", borderBottom:"1px solid #0E2040",
                      display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                      <div>
                        <div style={{ fontSize:13, fontWeight:600 }}>{result?.target || selected}</div>
                        <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono,monospace", marginTop:1 }}>
                          {result ? `TESS Sector ${result.sector} · PDCSAP Flux` : "Fetching…"}
                        </div>
                      </div>
                      {mcmcResult && (
                        <div style={{ display:"flex", alignItems:"center", gap:6, fontSize:10,
                          color:COLORS.amber, fontFamily:"Space Mono,monospace" }}>
                          <div style={{ width:10, height:2, background:COLORS.amber, borderRadius:1 }} />
                          MCMC model overlay active
                        </div>
                      )}
                    </div>
                    <div style={{ padding:"4px 16px 0", display:"flex", justifyContent:"space-between" }}>
                      <span style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono,monospace" }}>← Normalized Flux</span>
                      <span style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono,monospace" }}>Time [BTJD] →</span>
                    </div>
                    <div ref={containerRef} style={{ padding:"0 16px 14px" }}>
                      {result?.light_curve ? (
                        <LightCurveCanvas
                          width={canvasWidth - 32} height={220}
                          lcData={result.light_curve}
                          modelFlux={mcmcResult?.model_flux || null}
                          hoveredX={hoveredX} setHoveredX={setHoveredX} />
                      ) : (
                        <div style={{ height:220, display:"flex", alignItems:"center", justifyContent:"center",
                          color:COLORS.slate, fontSize:12, fontFamily:"Space Mono,monospace" }}>
                          {loading?"Downloading…":"—"}
                        </div>
                      )}
                    </div>
                    {result && tp && (
                      <div style={{ borderTop:"1px solid #0E2040", padding:"10px 16px",
                        display:"flex", gap:20, background:"#060E1E", flexWrap:"wrap" }}>
                        {[
                          {l:"PERIOD",   v:`${tp.period_days} d`},
                          {l:"DEPTH",    v:`${tp.depth_pct}%`},
                          {l:"DURATION", v:`${tp.duration_hours} hr`},
                          {l:"SNR",      v:`${tp.snr}σ`},
                          {l:"TRANSITS", v:`${tp.n_transits}`},
                        ].map(({l,v}) => (
                          <div key={l}>
                            <div style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1 }}>{l}</div>
                            <div style={{ fontSize:15, fontWeight:700, color:COLORS.cyan, fontFamily:"Space Mono,monospace", marginTop:2 }}>{v}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Phase-fold + periodogram */}
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                    {["Phase-Folded Transit","BLS Periodogram"].map((title,idx) => (
                      <div key={title} style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, overflow:"hidden" }}>
                        <div style={{ padding:"9px 13px", borderBottom:"1px solid #0E2040",
                          fontSize:10, color:COLORS.slate, fontFamily:"Space Mono,monospace" }}>{title}</div>
                        <div style={{ padding:8 }}>
                          {result
                            ? idx===0
                              ? <PhaseFoldedCanvas width={canvasWidth/2-40} height={140} data={result.phase_folded} />
                              : <PeriodogramCanvas width={canvasWidth/2-40} height={140} data={result.bls} />
                            : <div style={{ height:140, display:"flex", alignItems:"center", justifyContent:"center",
                                color:COLORS.slate, fontSize:11, fontFamily:"Space Mono,monospace" }}>
                                {loading?"Computing…":"—"}
                              </div>
                          }
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* BLS param cards */}
                  {result && tp && (
                    <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                      <div style={{ fontSize:11, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1, marginBottom:12 }}>
                        BLS PARAMETER ESTIMATES
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
                </>
              )}

              {/* ── TAB: MCMC ── */}
              {activeTab === "mcmc" && (
                <>
                  {/* MCMC status */}
                  {mcmcLoading && (
                    <div style={{ background:COLORS.panel, border:"1px solid #A78BFA44",
                      borderRadius:12, padding:20 }}>
                      <div style={{ fontSize:12, color:COLORS.purple, fontFamily:"Space Mono,monospace",
                        letterSpacing:1, marginBottom:14 }}>MCMC SAMPLING IN PROGRESS</div>
                      <MCMCProgress step={mcmcStep} pct={mcmcPct} />
                      <div style={{ marginTop:12, fontSize:10, color:COLORS.slate, fontFamily:"Space Mono,monospace" }}>
                        emcee · 32 walkers · 2000 steps · batman transit model
                      </div>
                    </div>
                  )}

                  {/* MCMC not started */}
                  {!mcmcLoading && !mcmcResult && (
                    <div style={{ background:COLORS.panel, border:"1px solid #0E2040",
                      borderRadius:12, padding:"40px 20px", textAlign:"center" }}>
                      <div style={{ fontSize:28, marginBottom:10, color:COLORS.purple }}>⟨ψ⟩</div>
                      <div style={{ color:COLORS.white, fontSize:14, fontWeight:600, marginBottom:8 }}>
                        Bayesian MCMC Transit Fitting
                      </div>
                      <div style={{ color:COLORS.slate, fontSize:11, fontFamily:"Space Mono,monospace", marginBottom:6 }}>
                        batman physical model · emcee affine-invariant sampler
                      </div>
                      <div style={{ color:COLORS.slate, fontSize:11, fontFamily:"Space Mono,monospace", marginBottom:24 }}>
                        Fits: Period · T₀ · Rp/R★ · a/R★ · inc · limb darkening
                      </div>
                      {result ? (
                        <button onClick={startMCMC}
                          style={{ background:"linear-gradient(135deg,#A78BFA22,#A78BFA11)",
                            border:`1px solid ${COLORS.purple}`, color:COLORS.purple,
                            borderRadius:8, padding:"10px 28px", fontSize:13,
                            fontFamily:"Space Mono,monospace", cursor:"pointer", fontWeight:700 }}>
                          START MCMC FIT → {selected}
                        </button>
                      ) : (
                        <div style={{ color:COLORS.slate, fontSize:12, fontFamily:"Space Mono,monospace" }}>
                          Run a scan first to load light curve data
                        </div>
                      )}
                    </div>
                  )}

                  {/* MCMC results */}
                  {mcmcResult && (
                    <>
                      {/* Summary bar */}
                      <div style={{ background:COLORS.panel, border:"1px solid #A78BFA44",
                        borderRadius:12, padding:"12px 16px",
                        display:"flex", gap:20, flexWrap:"wrap", alignItems:"center" }}>
                        {[
                          {l:"STATUS",     v:"CONVERGED",                            c:COLORS.green},
                          {l:"SAMPLES",    v:mcmcResult.n_samples?.toLocaleString(), c:COLORS.purple},
                          {l:"ACCEPTANCE", v:`${(mcmcResult.acceptance_frac*100).toFixed(1)}%`, c:COLORS.cyan},
                          {l:"ELAPSED",    v:`${mcmcResult.elapsed_sec}s`,           c:COLORS.amber},
                        ].map(({l,v,c}) => (
                          <div key={l}>
                            <div style={{ fontSize:9, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1 }}>{l}</div>
                            <div style={{ fontSize:14, fontWeight:700, color:c, fontFamily:"Space Mono,monospace", marginTop:2 }}>{v}</div>
                          </div>
                        ))}
                      </div>

                      {/* Posterior parameter table */}
                      <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                        <div style={{ fontSize:11, color:COLORS.slate, fontFamily:"Space Mono,monospace",
                          letterSpacing:1, marginBottom:4 }}>MCMC POSTERIOR ESTIMATES</div>
                        <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono,monospace", marginBottom:12 }}>
                          median  +1σ / -1σ  (16th–84th percentile)
                        </div>
                        {Object.entries(mcmcResult.percentiles || {}).map(([name, data]) => (
                          <MCMCParamRow key={name} name={name} data={data} />
                        ))}
                      </div>

                      {/* Key params as cards */}
                      <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                        <div style={{ fontSize:11, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1, marginBottom:12 }}>
                          MCMC BEST-FIT PARAMETERS
                        </div>
                        <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
                          {mcmcResult.percentiles && (() => {
                            const p = mcmcResult.percentiles;
                            return [
                              { label:"Period",     value:`${p["Period (d)"]?.median.toFixed(4)}`, unit:`days ±${p["Period (d)"]?.err_high.toFixed(5)}`, highlight:true },
                              { label:"Rp/R★",      value:`${p["Rp/R★"]?.median.toFixed(4)}`,     unit:`±${p["Rp/R★"]?.err_high.toFixed(4)}`, highlight:false },
                              { label:"Depth",      value:`${p["Depth (ppm)"]?.median.toFixed(0)}`, unit:"ppm", highlight:false },
                              { label:"Inclination",value:`${p["Inc (°)"]?.median.toFixed(2)}°`,  unit:`±${p["Inc (°)"]?.err_high.toFixed(3)}°`, highlight:false },
                              { label:"Rp",         value:`${p["Rp (R⊕)"]?.median.toFixed(2)}`,  unit:"R⊕ (estimated)", highlight:true },
                              { label:"a/R★",       value:`${p["a/R★"]?.median.toFixed(2)}`,      unit:`±${p["a/R★"]?.err_high.toFixed(3)}`, highlight:false },
                            ].map(c => <ParamCard key={c.label} {...c} />);
                          })()}
                        </div>
                      </div>

                      {/* Posterior plot */}
                      {mcmcResult.plots?.posterior && (
                        <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, overflow:"hidden" }}>
                          <div style={{ padding:"10px 16px", borderBottom:"1px solid #0E2040",
                            fontSize:11, color:COLORS.slate, fontFamily:"Space Mono,monospace" }}>
                            PHASE-FOLDED TRANSIT + POSTERIOR MODEL
                          </div>
                          <img src={`data:image/png;base64,${mcmcResult.plots.posterior}`}
                            style={{ width:"100%", display:"block" }} alt="Posterior transit model" />
                        </div>
                      )}

                      {/* Corner plot */}
                      {mcmcResult.plots?.corner && (
                        <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, overflow:"hidden" }}>
                          <div style={{ padding:"10px 16px", borderBottom:"1px solid #0E2040",
                            fontSize:11, color:COLORS.slate, fontFamily:"Space Mono,monospace" }}>
                            POSTERIOR CORNER PLOT · Period · T₀ · Rp/R★ · a/R★ · Inc
                          </div>
                          <img src={`data:image/png;base64,${mcmcResult.plots.corner}`}
                            style={{ width:"100%", display:"block" }} alt="Corner plot" />
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </div>

            {/* Right column */}
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

              {/* Classification */}
              <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                <div style={{ fontSize:12, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1, marginBottom:4 }}>
                  AI CLASSIFICATION
                </div>
                <div style={{ fontSize:10, color:COLORS.slate, fontFamily:"Space Mono,monospace", marginBottom:14 }}>
                  {result?.classifier || "CNN Ensemble"}
                </div>
                {cls ? (
                  <>
                    <div style={{ background:`${topMeta.color}10`, border:`1px solid ${topMeta.color}44`,
                      borderRadius:8, padding:"10px 12px", marginBottom:14,
                      display:"flex", gap:10, alignItems:"center" }}>
                      <div style={{ fontSize:22, color:topMeta.color }}>{topMeta.icon}</div>
                      <div>
                        <div style={{ color:topMeta.color, fontSize:13, fontWeight:700 }}>{cls.top_class}</div>
                        <div style={{ color:COLORS.slate, fontSize:10, fontFamily:"Space Mono,monospace" }}>
                          Confidence: {cls.confidence}%
                        </div>
                      </div>
                    </div>
                    {Object.entries(cls.probabilities).map(([label, val]) => {
                      const m = CLASS_META[label] || { icon:"?", color:COLORS.slate };
                      return <ConfBar key={label} label={label} value={val} color={m.color} icon={m.icon} />;
                    })}
                  </>
                ) : (
                  <div style={{ color:COLORS.slate, fontSize:12, fontFamily:"Space Mono,monospace",
                    textAlign:"center", padding:"20px 0" }}>
                    {loading?"Classifying…":"Run scan to classify"}
                  </div>
                )}
              </div>

              {/* MCMC progress sidebar */}
              {(mcmcLoading || mcmcResult) && (
                <div style={{ background:COLORS.panel, border:`1px solid ${mcmcResult?"#A78BFA44":"#0E2040"}`,
                  borderRadius:12, padding:16 }}>
                  <div style={{ fontSize:12, color:COLORS.purple, fontFamily:"Space Mono,monospace", letterSpacing:1, marginBottom:12 }}>
                    MCMC STATUS
                  </div>
                  {mcmcLoading && <MCMCProgress step={mcmcStep} pct={mcmcPct} />}
                  {mcmcResult && (
                    <div>
                      <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:10 }}>
                        <div style={{ width:8, height:8, borderRadius:"50%", background:COLORS.green,
                          boxShadow:`0 0 6px ${COLORS.green}` }} />
                        <span style={{ color:COLORS.green, fontSize:11, fontFamily:"Space Mono,monospace" }}>CONVERGED</span>
                      </div>
                      {[
                        {l:"Job ID",     v:mcmcJobId?.slice(0,8)+"…"},
                        {l:"Samples",    v:mcmcResult.n_samples?.toLocaleString()},
                        {l:"Acceptance", v:`${(mcmcResult.acceptance_frac*100).toFixed(1)}%`},
                        {l:"Runtime",    v:`${mcmcResult.elapsed_sec}s`},
                      ].map(({l,v}) => (
                        <div key={l} style={{ display:"flex", justifyContent:"space-between",
                          padding:"5px 0", borderBottom:"1px solid #0E2040" }}>
                          <span style={{ color:COLORS.slate, fontSize:10, fontFamily:"Space Mono,monospace" }}>{l}</span>
                          <span style={{ color:COLORS.white, fontSize:10, fontFamily:"Space Mono,monospace" }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Data quality */}
              {dq && (
                <div style={{ background:COLORS.panel, border:"1px solid #0E2040", borderRadius:12, padding:16 }}>
                  <div style={{ fontSize:12, color:COLORS.slate, fontFamily:"Space Mono,monospace", letterSpacing:1, marginBottom:12 }}>
                    DATA QUALITY
                  </div>
                  {[
                    {label:"Completeness", value:dq.completeness_pct, unit:"%",   color:COLORS.green, max:100},
                    {label:"CDPP Noise",   value:dq.cdpp_ppm,         unit:" ppm",color:COLORS.cyan,  max:1000},
                    {label:"Sys. Noise",   value:dq.sys_noise_ppm,    unit:" ppm",color:COLORS.amber, max:200},
                  ].map(({label,value,unit,color,max}) => (
                    <div key={label} style={{ marginBottom:10 }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                        <span style={{ color:COLORS.slate, fontSize:11, fontFamily:"Space Mono,monospace" }}>{label}</span>
                        <span style={{ color, fontSize:11, fontFamily:"Space Mono,monospace" }}>{value}{unit}</span>
                      </div>
                      <div style={{ height:3, background:"#0E2040", borderRadius:2, overflow:"hidden" }}>
                        <div style={{ height:"100%", width:`${Math.min(100,(value/max)*100)}%`,
                          background:color, borderRadius:2, opacity:0.8 }} />
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