import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000/api";
const COLORS = {
  bg:"#050B1A",panel:"#0A1628",border:"#0E2040",
  cyan:"#00D4FF",amber:"#FFB347",green:"#39FF14",
  red:"#FF4757",slate:"#8899BB",light:"#AAB8CC",white:"#E8F0FF",purple:"#A78BFA",
};
const CLASS_META = {
  "Exoplanet Transit":{icon:"◎",color:"#00D4FF"},
  "Eclipsing Binary":{icon:"⊕",color:"#FFB347"},
  "Stellar Blend":{icon:"◈",color:"#FF4757"},
  "Starspot":{icon:"✦",color:"#39FF14"},
};

// ── Reusable canvas components ─────────────────────────────────────────────
function LCCanvas({width,height,lcData,modelFlux,hoveredX,setHoveredX}){
  const ref=useRef(null);
  useEffect(()=>{
    const c=ref.current; if(!c||!lcData)return;
    const {time,flux}=lcData; if(!time||!time.length)return;
    const dpr=window.devicePixelRatio||1;
    c.width=width*dpr;c.height=height*dpr;
    c.style.width=width+"px";c.style.height=height+"px";
    const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
    const P={l:50,r:12,t:18,b:22};
    const W=width-P.l-P.r,H=height-P.t-P.b;
    const tMin=Math.min(...time),tMax=Math.max(...time);
    const fArr=flux.filter(isFinite);
    const fMin=Math.min(...fArr)-0.002,fMax=Math.max(...fArr)+0.002;
    const toX=t=>P.l+((t-tMin)/(tMax-tMin))*W;
    const toY=f=>P.t+((fMax-f)/(fMax-fMin))*H;
    ctx.strokeStyle="#0E2040";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=P.t+(i/4)*H;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(P.l+W,y);ctx.stroke();}
    const stride=Math.max(1,Math.floor(time.length/3000));
    ctx.beginPath();let st=false;
    for(let i=0;i<time.length;i+=stride){
      if(!isFinite(flux[i]))continue;
      const x=toX(time[i]),y=toY(flux[i]);
      if(!st){ctx.moveTo(x,y);st=true;}else ctx.lineTo(x,y);
    }
    ctx.strokeStyle=COLORS.cyan;ctx.lineWidth=0.8;ctx.shadowColor=COLORS.cyan;ctx.shadowBlur=2;ctx.stroke();ctx.shadowBlur=0;
    if(modelFlux&&modelFlux.length===time.length){
      ctx.beginPath();st=false;
      for(let i=0;i<time.length;i+=stride){
        const x=toX(time[i]),y=toY(modelFlux[i]);
        if(!st){ctx.moveTo(x,y);st=true;}else ctx.lineTo(x,y);
      }
      ctx.strokeStyle=COLORS.amber;ctx.lineWidth=2;ctx.shadowColor=COLORS.amber;ctx.shadowBlur=5;ctx.stroke();ctx.shadowBlur=0;
    }
    if(hoveredX!==null){
      const t=tMin+((hoveredX-P.l)/W)*(tMax-tMin);
      if(t>=tMin&&t<=tMax){
        let ni=0,nd=Infinity;
        for(let i=0;i<time.length;i++){const d=Math.abs(time[i]-t);if(d<nd){nd=d;ni=i;}}
        const hx=toX(time[ni]),hy=toY(flux[ni]);
        ctx.strokeStyle="rgba(255,179,71,0.5)";ctx.lineWidth=1;ctx.setLineDash([4,4]);
        ctx.beginPath();ctx.moveTo(hx,P.t);ctx.lineTo(hx,P.t+H);ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle=COLORS.amber;ctx.beginPath();ctx.arc(hx,hy,4,0,Math.PI*2);ctx.fill();
      }
    }
  },[width,height,lcData,modelFlux,hoveredX]);
  return <canvas ref={ref} style={{cursor:"crosshair",display:"block"}}
    onMouseMove={e=>setHoveredX(e.clientX-e.currentTarget.getBoundingClientRect().left)}
    onMouseLeave={()=>setHoveredX(null)}/>;
}

function AttentionCanvas({width,height,weights}){
  const ref=useRef(null);
  useEffect(()=>{
    const c=ref.current; if(!c||!weights||!weights.length)return;
    const dpr=window.devicePixelRatio||1;
    c.width=width*dpr;c.height=height*dpr;
    c.style.width=width+"px";c.style.height=height+"px";
    const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
    const P={l:10,r:10,t:8,b:18};
    const W=width-P.l-P.r,H=height-P.t-P.b;
    const n=weights.length;
    const maxW=Math.max(...weights)||1;
    const barW=W/n;
    weights.forEach((w,i)=>{
      const norm=w/maxW;
      const x=P.l+i*barW;
      const bh=norm*H;
      const r=Math.round(167*norm+0*(1-norm));
      const g=Math.round(139*norm);
      const b=Math.round(250*norm+14*(1-norm));
      ctx.fillStyle=`rgba(${r},${g},${b},${0.4+norm*0.6})`;
      ctx.fillRect(x,P.t+H-bh,barW-1,bh);
    });
    // Transit centre line
    const cx=P.l+W/2;
    ctx.strokeStyle="rgba(255,179,71,0.7)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(cx,P.t);ctx.lineTo(cx,P.t+H);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=COLORS.slate;ctx.font="9px 'Space Mono',monospace";
    ctx.textAlign="left"; ctx.fillText("-0.5",P.l,height-3);
    ctx.textAlign="center";ctx.fillText("0 (transit)",cx,height-3);
    ctx.textAlign="right"; ctx.fillText("+0.5",P.l+W,height-3);
  },[width,height,weights]);
  return <canvas ref={ref} style={{display:"block"}}/>;
}

function PhaseFoldCanvas({width,height,data}){
  const ref=useRef(null);
  useEffect(()=>{
    const c=ref.current; if(!c||!data)return;
    const {phase,flux}=data;
    const dpr=window.devicePixelRatio||1;
    c.width=width*dpr;c.height=height*dpr;
    c.style.width=width+"px";c.style.height=height+"px";
    const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
    const P={l:8,r:8,t:10,b:18};const W=width-P.l-P.r,H=height-P.t-P.b;
    const fArr=flux.filter(isFinite),fMin=Math.min(...fArr)-0.001,fMax=Math.max(...fArr)+0.001;
    const toX=p=>P.l+((p+0.5)/1)*W,toY=f=>P.t+((fMax-f)/(fMax-fMin))*H;
    ctx.strokeStyle="rgba(0,212,255,0.3)";ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(toX(0),P.t);ctx.lineTo(toX(0),P.t+H);ctx.stroke();ctx.setLineDash([]);
    const stride=Math.max(1,Math.floor(phase.length/2000));
    ctx.beginPath();let st=false;
    for(let i=0;i<phase.length;i+=stride){
      if(!isFinite(flux[i]))continue;
      const x=toX(phase[i]),y=toY(flux[i]);
      if(!st){ctx.moveTo(x,y);st=true;}else ctx.lineTo(x,y);
    }
    ctx.strokeStyle=COLORS.cyan;ctx.lineWidth=1;ctx.shadowColor=COLORS.cyan;ctx.shadowBlur=2;ctx.stroke();ctx.shadowBlur=0;
  },[width,height,data]);
  return <canvas ref={ref} style={{display:"block"}}/>;
}

function PeriodogramCanvas({width,height,data}){
  const ref=useRef(null);
  useEffect(()=>{
    const c=ref.current; if(!c||!data)return;
    const {periods,power,peak_period}=data;
    const dpr=window.devicePixelRatio||1;
    c.width=width*dpr;c.height=height*dpr;c.style.width=width+"px";c.style.height=height+"px";
    const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
    const P={l:8,r:8,t:10,b:18};const W=width-P.l-P.r,H=height-P.t-P.b;
    const pMin=Math.min(...periods),pMax=Math.max(...periods);
    const toX=p=>P.l+((p-pMin)/(pMax-pMin))*W,toY=v=>P.t+(1-v)*H;
    ctx.beginPath();
    periods.forEach((p,i)=>{const x=toX(p),y=toY(power[i]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
    ctx.strokeStyle=COLORS.green;ctx.lineWidth=1;ctx.shadowColor=COLORS.green;ctx.shadowBlur=2;ctx.stroke();ctx.shadowBlur=0;
    if(peak_period){
      const px=toX(peak_period);
      ctx.strokeStyle="rgba(255,179,71,0.7)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
      ctx.beginPath();ctx.moveTo(px,P.t);ctx.lineTo(px,P.t+H);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=COLORS.amber;ctx.font="9px 'Space Mono',monospace";ctx.textAlign="center";
      ctx.fillText(`${peak_period.toFixed(2)}d`,px,P.t+9);
    }
  },[width,height,data]);
  return <canvas ref={ref} style={{display:"block"}}/>;
}

// ── Small reusables ────────────────────────────────────────────────────────
const ss = (style) => ({ fontFamily:"Space Mono,monospace", ...style });
function ConfBar({label,value,color,icon}){
  return(
    <div style={{marginBottom:8}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
        <span style={ss({color:COLORS.light,fontSize:11})}><span style={{color,marginRight:5}}>{icon}</span>{label}</span>
        <span style={ss({color,fontSize:11,fontWeight:700})}>{value.toFixed(1)}%</span>
      </div>
      <div style={{height:3,background:"#0E2040",borderRadius:2,overflow:"hidden"}}>
        <div style={{height:"100%",width:`${value}%`,background:`linear-gradient(90deg,${color}66,${color})`,
          borderRadius:2,boxShadow:`0 0 5px ${color}`,transition:"width 1s ease"}}/>
      </div>
    </div>
  );
}
function ParamCard({label,value,unit}){
  return(
    <div style={{background:"#060E1E",border:"1px solid #0E2040",borderRadius:8,padding:"10px 12px",flex:1,minWidth:90}}>
      <div style={ss({color:COLORS.slate,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:4})}>{label}</div>
      <div style={ss({color:COLORS.cyan,fontSize:16,fontWeight:700})}>{value}</div>
      <div style={ss({color:COLORS.slate,fontSize:9,marginTop:2})}>{unit}</div>
    </div>
  );
}
function Badge({text,color}){
  return <span style={{background:`${color}22`,border:`1px solid ${color}66`,color,
    borderRadius:12,padding:"2px 10px",fontSize:10,fontFamily:"Space Mono,monospace",fontWeight:700}}>{text}</span>;
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App(){
  const [targets,setTargets]=useState([]);
  const [selected,setSelected]=useState("L 98-59");
  const [loading,setLoading]=useState(false);
  const [result,setResult]=useState(null);
  const [error,setError]=useState(null);
  const [hoveredX,setHoveredX]=useState(null);
  const [backendOk,setBackendOk]=useState(null);
  const [modelInfo,setModelInfo]=useState(null);
  const [activeTab,setActiveTab]=useState("lightcurve");

  // MCMC
  const [mcmcJobId,setMcmcJobId]=useState(null);
  const [mcmcState,setMcmcState]=useState(null);
  const [mcmcStep,setMcmcStep]=useState("");
  const [mcmcPct,setMcmcPct]=useState(0);
  const [mcmcResult,setMcmcResult]=useState(null);
  const [mcmcLoading,setMcmcLoading]=useState(false);
  const mcmcPoll=useRef(null);

  // Batch
  const [batchJobId,setBatchJobId]=useState(null);
  const [batchState,setBatchState]=useState(null);
  const [batchStep,setBatchStep]=useState("");
  const [batchPct,setBatchPct]=useState(0);
  const [batchResult,setBatchResult]=useState(null);
  const [batchLoading,setBatchLoading]=useState(false);
  const [batchSector,setBatchSector]=useState(14);
  const [batchMax,setBatchMax]=useState(10);
  const batchPoll=useRef(null);

  const containerRef=useRef(null);
  const [cw,setCw]=useState(700);
  useEffect(()=>{
    const obs=new ResizeObserver(([e])=>setCw(Math.max(300,e.contentRect.width)));
    if(containerRef.current)obs.observe(containerRef.current);
    return()=>obs.disconnect();
  },[]);

  useEffect(()=>{
    fetch(`${API}/health`).then(r=>r.ok?setBackendOk(true):setBackendOk(false)).catch(()=>setBackendOk(false));
    fetch(`${API}/targets`).then(r=>r.json()).then(d=>setTargets(d.targets||[])).catch(()=>{});
    fetch(`${API}/model-info`).then(r=>r.json()).then(setModelInfo).catch(()=>{});
  },[]);

  const pollMCMC=useCallback((jobId)=>{
    mcmcPoll.current=setInterval(async()=>{
      try{
        const r=await fetch(`${API}/mcmc/status?job_id=${jobId}`);
        const d=await r.json();
        setMcmcState(d.state);setMcmcStep(d.step||"");setMcmcPct(d.pct||0);
        if(d.state==="SUCCESS"){
          clearInterval(mcmcPoll.current);
          const r2=await fetch(`${API}/mcmc/result?job_id=${jobId}`);
          const d2=await r2.json();
          if(d2.status==="ok")setMcmcResult(d2.data);
          setMcmcLoading(false);
        }
        if(d.state==="FAILURE"){clearInterval(mcmcPoll.current);setMcmcLoading(false);}
      }catch{clearInterval(mcmcPoll.current);setMcmcLoading(false);}
    },1500);
  },[]);

  const pollBatch=useCallback((jobId)=>{
    batchPoll.current=setInterval(async()=>{
      try{
        const r=await fetch(`${API}/batch/status?job_id=${jobId}`);
        const d=await r.json();
        setBatchState(d.state);setBatchStep(d.step||"");setBatchPct(d.pct||0);
        if(d.state==="SUCCESS"){
          clearInterval(batchPoll.current);
          const r2=await fetch(`${API}/batch/results?job_id=${jobId}`);
          const d2=await r2.json();
          if(d2.status==="ok")setBatchResult(d2.data);
          setBatchLoading(false);
        }
        if(d.state==="FAILURE"){clearInterval(batchPoll.current);setBatchLoading(false);}
      }catch{clearInterval(batchPoll.current);setBatchLoading(false);}
    },2000);
  },[]);

  const runScan=useCallback(async(target)=>{
    setLoading(true);setResult(null);setError(null);setMcmcResult(null);setMcmcJobId(null);setMcmcState(null);
    try{
      const res=await fetch(`${API}/analyse?target=${encodeURIComponent(target)}`);
      const json=await res.json();
      if(json.status==="ok")setResult(json.data);
      else setError(json.message||"Unknown error");
    }catch{setError("Cannot reach backend on port 5000");}
    setLoading(false);
  },[]);

  const startMCMC=useCallback(async()=>{
    setMcmcLoading(true);setMcmcResult(null);setMcmcState("PENDING");setMcmcStep("Queuing…");setMcmcPct(0);
    setActiveTab("mcmc");
    const r=await fetch(`${API}/mcmc/start`,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({target:selected,n_walkers:32,n_steps:2000,n_burn:500})});
    const d=await r.json();
    setMcmcJobId(d.job_id);pollMCMC(d.job_id);
  },[selected,pollMCMC]);

  const startBatch=useCallback(async()=>{
    setBatchLoading(true);setBatchResult(null);setBatchState("PENDING");setBatchStep("Queuing…");setBatchPct(0);
    setActiveTab("batch");
    const r=await fetch(`${API}/batch/start`,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({sector:batchSector,max_targets:batchMax})});
    const d=await r.json();
    setBatchJobId(d.job_id);pollBatch(d.job_id);
  },[batchSector,batchMax,pollBatch]);

  const downloadPDF=useCallback(async(type)=>{
    let url;
    if(type==="candidate"){
      url=`${API}/report/candidate?target=${encodeURIComponent(selected)}`;
      if(mcmcJobId)url+=`&mcmc_job_id=${mcmcJobId}`;
    }else{
      url=`${API}/report/sector?job_id=${batchJobId}&sector=${batchSector}`;
    }
    const a=document.createElement("a");a.href=url;a.download="ExoDetect_Report.pdf";a.click();
  },[selected,mcmcJobId,batchJobId,batchSector]);

  useEffect(()=>()=>{
    if(mcmcPoll.current)clearInterval(mcmcPoll.current);
    if(batchPoll.current)clearInterval(batchPoll.current);
  },[]);

  const cls=result?.classification;
  const tp=result?.transit_params;
  const dq=result?.data_quality;
  const topMeta=cls?(CLASS_META[cls.top_class]||{icon:"?",color:COLORS.cyan}):null;

  const TABS=[
    {id:"lightcurve",label:"Light Curve"},
    {id:"xai",       label:"XAI Heatmap"+(result?.attention_weights?" ✓":"")},
    {id:"mcmc",      label:"MCMC Fit"+(mcmcState==="SUCCESS"?" ✓":"")},
    {id:"batch",     label:"Batch Scan"+(batchState==="SUCCESS"?" ✓":"")},
  ];

  return(
    <div style={{minHeight:"100vh",background:COLORS.bg,color:COLORS.white,fontFamily:"Inter,system-ui,sans-serif",paddingBottom:40}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
        @keyframes pulse{0%,100%{box-shadow:0 0 6px #00D4FF}50%{box-shadow:0 0 18px #00D4FF}}
        ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#050B1A}::-webkit-scrollbar-thumb{background:#0E2040;border-radius:3px}
      `}</style>

      {/* Header */}
      <div style={{borderBottom:"1px solid #0E2040",background:"linear-gradient(180deg,#080F20,#050B1A)",padding:"0 24px"}}>
        <div style={{maxWidth:1200,margin:"0 auto",display:"flex",alignItems:"center",justifyContent:"space-between",height:60}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div style={{width:32,height:32,borderRadius:"50%",background:"radial-gradient(circle at 35% 35%,#00D4FF33,#050B1A)",border:"1.5px solid #00D4FF55",display:"flex",alignItems:"center",justifyContent:"center",fontSize:16}}>◎</div>
            <div>
              <div style={{fontSize:15,fontWeight:700,letterSpacing:0.5}}>EXODETECT</div>
              <div style={ss({fontSize:10,color:COLORS.slate,letterSpacing:1})}>
                PHASE 3 · {modelInfo?.ensemble?"ENSEMBLE (CNN+TF)":modelInfo?.cnn?.loaded?"CNN ONLY":"HEURISTIC"} · APPLE SILICON
              </div>
            </div>
          </div>
          <div style={{display:"flex",gap:6,alignItems:"center",flexWrap:"wrap"}}>
            {/* Backend status */}
            <div style={{display:"flex",alignItems:"center",gap:5,padding:"4px 10px",background:"#060E1E",border:"1px solid #0E2040",borderRadius:6}}>
              <div style={{width:7,height:7,borderRadius:"50%",background:backendOk===null?COLORS.amber:backendOk?COLORS.green:COLORS.red,animation:"blink 2s ease-in-out infinite"}}/>
              <span style={ss({fontSize:10,color:backendOk?COLORS.green:COLORS.red})}>
                {backendOk===null?"CONNECTING":backendOk?"LIVE":"OFFLINE"}
              </span>
            </div>
            <button onClick={()=>runScan(selected)} disabled={loading}
              style={{background:loading?"#0E2040":"linear-gradient(135deg,#00D4FF22,#00D4FF11)",border:`1px solid ${loading?"#0E2040":COLORS.cyan}`,color:loading?COLORS.slate:COLORS.cyan,borderRadius:6,padding:"6px 14px",fontSize:11,fontFamily:"Space Mono,monospace",cursor:loading?"not-allowed":"pointer",fontWeight:700}}>
              {loading?"FETCHING…":"RUN SCAN"}
            </button>
            {result&&<button onClick={startMCMC} disabled={mcmcLoading}
              style={{background:"linear-gradient(135deg,#A78BFA22,#A78BFA11)",border:`1px solid ${mcmcLoading?"#0E2040":COLORS.purple}`,color:mcmcLoading?COLORS.slate:COLORS.purple,borderRadius:6,padding:"6px 14px",fontSize:11,fontFamily:"Space Mono,monospace",cursor:mcmcLoading?"not-allowed":"pointer",fontWeight:700}}>
              {mcmcLoading?"MCMC…":"MCMC FIT"}
            </button>}
            <button onClick={startBatch} disabled={batchLoading}
              style={{background:"linear-gradient(135deg,#FFB34722,#FFB34711)",border:`1px solid ${batchLoading?"#0E2040":COLORS.amber}`,color:batchLoading?COLORS.slate:COLORS.amber,borderRadius:6,padding:"6px 14px",fontSize:11,fontFamily:"Space Mono,monospace",cursor:batchLoading?"not-allowed":"pointer",fontWeight:700}}>
              {batchLoading?"SCANNING…":"BATCH SCAN"}
            </button>
            {result&&<button onClick={()=>downloadPDF("candidate")}
              style={{background:"linear-gradient(135deg,#39FF1422,#39FF1411)",border:`1px solid ${COLORS.green}`,color:COLORS.green,borderRadius:6,padding:"6px 14px",fontSize:11,fontFamily:"Space Mono,monospace",cursor:"pointer",fontWeight:700}}>
              ↓ PDF
            </button>}
          </div>
        </div>
      </div>

      <div style={{maxWidth:1200,margin:"0 auto",padding:"18px 24px 0"}}>

        {/* Model info bar */}
        {modelInfo&&(
          <div style={{display:"flex",gap:8,marginBottom:14,flexWrap:"wrap",alignItems:"center"}}>
            <Badge text={modelInfo.ensemble?"Ensemble: CNN+TF":modelInfo.cnn?.loaded?"CNN only":"Heuristic"}
              color={modelInfo.ensemble?COLORS.purple:modelInfo.cnn?.loaded?COLORS.cyan:COLORS.amber}/>
            {modelInfo.cnn?.loaded&&<Badge text={`CNN AUC ${modelInfo.cnn.metrics?.test_auc?.toFixed(3)||"?"}`} color={COLORS.cyan}/>}
            {modelInfo.transformer?.loaded&&<Badge text={`TF AUC ${modelInfo.transformer.metrics?.test_auc?.toFixed(3)||"?"}`} color={COLORS.purple}/>}
            {modelInfo.ensemble_weights&&<Badge text={`Weights CNN:${(modelInfo.ensemble_weights.cnn*100).toFixed(0)}% TF:${(modelInfo.ensemble_weights.transformer*100).toFixed(0)}%`} color={COLORS.slate}/>}
            <Badge text={`Device: ${modelInfo.device}`} color={COLORS.green}/>
          </div>
        )}

        {/* Target row */}
        <div style={{display:"flex",gap:8,marginBottom:16,flexWrap:"wrap",alignItems:"center"}}>
          {(targets.length>0?targets.map(t=>t.name):["L 98-59","TOI-700","WASP-18","TIC 286923464","HD 21749"]).map(n=>(
            <button key={n} onClick={()=>setSelected(n)}
              style={{background:selected===n?"#00D4FF18":"transparent",border:`1px solid ${selected===n?COLORS.cyan:"#0E2040"}`,color:selected===n?COLORS.cyan:COLORS.slate,borderRadius:20,padding:"4px 14px",fontSize:11,fontFamily:"Space Mono,monospace",cursor:"pointer",transition:"all 0.2s"}}>
              {n}
            </button>
          ))}
          {result&&<span style={ss({marginLeft:"auto",color:COLORS.slate,fontSize:11})}>Sector {result.sector} · {result.n_cadences?.toLocaleString()} cadences</span>}
        </div>

        {error&&<div style={{background:"#FF475722",border:"1px solid #FF475755",borderRadius:8,padding:"10px 16px",marginBottom:14,color:COLORS.red,fontSize:12,fontFamily:"Space Mono,monospace"}}>⚠ {error}</div>}

        {!result&&!loading&&!error&&(
          <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:"40px 20px",textAlign:"center",marginBottom:16}}>
            <div style={{fontSize:36,marginBottom:10}}>◎</div>
            <div style={{color:COLORS.white,fontSize:15,fontWeight:600,marginBottom:6}}>ExoDetect Phase 3</div>
            <div style={ss({color:COLORS.slate,fontSize:12,marginBottom:6})}>Transformer + CNN Ensemble · Batch Sector Processing · XAI Heatmaps · PDF Reports</div>
            <div style={{display:"flex",gap:10,justifyContent:"center",marginTop:20,flexWrap:"wrap"}}>
              <button onClick={()=>runScan(selected)} style={{background:"linear-gradient(135deg,#00D4FF22,#00D4FF11)",border:`1px solid ${COLORS.cyan}`,color:COLORS.cyan,borderRadius:8,padding:"10px 24px",fontSize:12,fontFamily:"Space Mono,monospace",cursor:"pointer",fontWeight:700}}>
                SCAN → {selected}
              </button>
              <button onClick={startBatch} style={{background:"linear-gradient(135deg,#FFB34722,#FFB34711)",border:`1px solid ${COLORS.amber}`,color:COLORS.amber,borderRadius:8,padding:"10px 24px",fontSize:12,fontFamily:"Space Mono,monospace",cursor:"pointer",fontWeight:700}}>
                BATCH SCAN SECTOR {batchSector}
              </button>
            </div>
          </div>
        )}

        {(result||loading||batchLoading||batchResult)&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 280px",gap:16,alignItems:"start"}}>
            {/* Left */}
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {/* Tab bar */}
              <div style={{display:"flex",gap:2,background:"#060E1E",border:"1px solid #0E2040",borderRadius:10,padding:3,width:"fit-content"}}>
                {TABS.map(tab=>(
                  <button key={tab.id} onClick={()=>setActiveTab(tab.id)}
                    style={{background:activeTab===tab.id?"#0A1628":"transparent",border:activeTab===tab.id?"1px solid #0E2040":"1px solid transparent",color:activeTab===tab.id?COLORS.white:COLORS.slate,borderRadius:7,padding:"6px 14px",fontSize:11,fontFamily:"Space Mono,monospace",cursor:"pointer",transition:"all 0.2s",fontWeight:activeTab===tab.id?600:400}}>
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* ── LIGHT CURVE TAB ── */}
              {activeTab==="lightcurve"&&(
                <>
                  <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,overflow:"hidden"}}>
                    <div style={{padding:"11px 15px",borderBottom:"1px solid #0E2040",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                      <div>
                        <div style={{fontSize:13,fontWeight:600}}>{result?.target||selected}</div>
                        <div style={ss({fontSize:10,color:COLORS.slate,marginTop:1})}>TESS Sector {result?.sector} · PDCSAP · {result?.classifier}</div>
                      </div>
                      {mcmcResult&&<Badge text="MCMC overlay active" color={COLORS.amber}/>}
                    </div>
                    <div ref={containerRef} style={{padding:"6px 15px 14px"}}>
                      {result?.light_curve
                        ?<LCCanvas width={cw-30} height={210} lcData={result.light_curve}
                            modelFlux={mcmcResult?.model_flux||null} hoveredX={hoveredX} setHoveredX={setHoveredX}/>
                        :<div style={{height:210,display:"flex",alignItems:"center",justifyContent:"center",color:COLORS.slate,fontSize:12,fontFamily:"Space Mono,monospace"}}>{loading?"Downloading…":"—"}</div>
                      }
                    </div>
                    {result&&tp&&(
                      <div style={{borderTop:"1px solid #0E2040",padding:"9px 15px",display:"flex",gap:18,background:"#060E1E",flexWrap:"wrap"}}>
                        {[{l:"PERIOD",v:`${tp.period_days}d`},{l:"DEPTH",v:`${tp.depth_pct}%`},{l:"DURATION",v:`${tp.duration_hours}hr`},{l:"SNR",v:`${tp.snr}σ`},{l:"TRANSITS",v:`${tp.n_transits}`}].map(({l,v})=>(
                          <div key={l}>
                            <div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>{l}</div>
                            <div style={ss({fontSize:14,fontWeight:700,color:COLORS.cyan,marginTop:2})}>{v}</div>
                          </div>
                        ))}
                        {result.p_cnn&&<div><div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>P(PLANET) CNN</div><div style={ss({fontSize:14,fontWeight:700,color:COLORS.cyan,marginTop:2})}>{result.p_cnn}%</div></div>}
                        {result.p_transformer&&<div><div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>P(PLANET) TF</div><div style={ss({fontSize:14,fontWeight:700,color:COLORS.purple,marginTop:2})}>{result.p_transformer}%</div></div>}
                      </div>
                    )}
                  </div>
                  {/* Phase-fold + periodogram */}
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                    {["Phase-Folded","BLS Periodogram"].map((t,i)=>(
                      <div key={t} style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,overflow:"hidden"}}>
                        <div style={{padding:"8px 12px",borderBottom:"1px solid #0E2040",fontSize:10,color:COLORS.slate,fontFamily:"Space Mono,monospace"}}>{t}</div>
                        <div style={{padding:7}}>
                          {result?i===0
                            ?<PhaseFoldCanvas width={cw/2-36} height={130} data={result.phase_folded}/>
                            :<PeriodogramCanvas width={cw/2-36} height={130} data={result.bls}/>
                            :<div style={{height:130,display:"flex",alignItems:"center",justifyContent:"center",color:COLORS.slate,fontSize:11,fontFamily:"Space Mono,monospace"}}>{loading?"Computing…":"—"}</div>
                          }
                        </div>
                      </div>
                    ))}
                  </div>
                  {result&&tp&&(
                    <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:14}}>
                      <div style={ss({fontSize:11,color:COLORS.slate,letterSpacing:1,marginBottom:10})}>BLS PARAMETER ESTIMATES</div>
                      <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                        <ParamCard label="Period"   value={`${tp.period_days}`}    unit={`days ±${tp.period_err}`}/>
                        <ParamCard label="Depth"    value={`${tp.depth_pct}%`}     unit={`Rp/R★=${tp.rp_rs}`}/>
                        <ParamCard label="Duration" value={`${tp.duration_hours}`} unit={`hr ±${tp.duration_err}`}/>
                        <ParamCard label="SNR"      value={`${tp.snr}σ`}           unit="significance"/>
                        <ParamCard label="N Transit" value={`${tp.n_transits}`}    unit="observed"/>
                        <ParamCard label="T₀"       value={tp.t0_bjd}             unit="BTJD"/>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* ── XAI TAB ── */}
              {activeTab==="xai"&&(
                <>
                  <div style={{background:COLORS.panel,border:"1px solid #A78BFA44",borderRadius:12,padding:16}}>
                    <div style={ss({fontSize:12,color:COLORS.purple,letterSpacing:1,marginBottom:6})}>TRANSITFORMER ATTENTION HEATMAP</div>
                    <div style={ss({fontSize:11,color:COLORS.slate,marginBottom:14})}>
                      Multi-head self-attention weights from the last Transformer encoder layer.<br/>
                      High values near phase=0 indicate the model detected a real transit signal.
                    </div>
                    {result?.attention_weights?(
                      <>
                        <AttentionCanvas width={cw-52} height={120} weights={result.attention_weights}/>
                        <div style={{display:"flex",gap:16,marginTop:12,flexWrap:"wrap"}}>
                          <div>
                            <div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>PEAK ATTENTION PATCH</div>
                            <div style={ss({fontSize:14,fontWeight:700,color:COLORS.purple,marginTop:2})}>
                              {(()=>{const mi=result.attention_weights.indexOf(Math.max(...result.attention_weights));return((mi/result.attention_weights.length)-0.5).toFixed(3);})()}
                            </div>
                          </div>
                          <div>
                            <div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>N PATCHES</div>
                            <div style={ss({fontSize:14,fontWeight:700,color:COLORS.purple,marginTop:2})}>{result.attention_weights.length}</div>
                          </div>
                          <div>
                            <div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>TRANSIT ATTENTION RATIO</div>
                            <div style={ss({fontSize:14,fontWeight:700,color:COLORS.purple,marginTop:2})}>
                              {(()=>{
                                const w=result.attention_weights;
                                const n=w.length;
                                const centre=Math.floor(n/2);
                                const width=Math.floor(n*0.1);
                                const transitSum=w.slice(centre-width,centre+width).reduce((a,b)=>a+b,0);
                                const totalSum=w.reduce((a,b)=>a+b,0);
                                return (transitSum/totalSum*100).toFixed(1)+"%";
                              })()}
                            </div>
                          </div>
                        </div>
                      </>
                    ):(
                      <div style={{height:120,display:"flex",alignItems:"center",justifyContent:"center",color:COLORS.slate,fontSize:12,fontFamily:"Space Mono,monospace",flexDirection:"column",gap:8}}>
                        <div>No attention weights available</div>
                        {!modelInfo?.transformer?.loaded&&<div style={ss({fontSize:10,color:COLORS.red})}>TransitFormer not loaded — run train_transformer.py first</div>}
                        {modelInfo?.transformer?.loaded&&!result&&<div style={ss({fontSize:10})}>Run a scan to generate attention weights</div>}
                      </div>
                    )}
                  </div>
                  {result&&(
                    <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:14}}>
                      <div style={ss({fontSize:11,color:COLORS.slate,letterSpacing:1,marginBottom:10})}>MODEL CONFIDENCE BREAKDOWN</div>
                      <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                        {result.p_cnn&&<ParamCard label="CNN P(planet)"         value={`${result.p_cnn}%`}         unit="Phase 1 model"/>}
                        {result.p_transformer&&<ParamCard label="TF P(planet)"  value={`${result.p_transformer}%`} unit="Phase 3 model"/>}
                        {result.p_cnn&&result.p_transformer&&(
                          <ParamCard label="Ensemble"
                            value={`${(0.45*result.p_cnn+0.55*result.p_transformer).toFixed(1)}%`}
                            unit="CNN×0.45 + TF×0.55"/>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* ── MCMC TAB ── */}
              {activeTab==="mcmc"&&(
                <>
                  {mcmcLoading&&(
                    <div style={{background:COLORS.panel,border:"1px solid #A78BFA44",borderRadius:12,padding:18}}>
                      <div style={ss({fontSize:12,color:COLORS.purple,letterSpacing:1,marginBottom:12})}>MCMC IN PROGRESS</div>
                      <div style={{marginBottom:6,display:"flex",justifyContent:"space-between"}}>
                        <span style={ss({color:COLORS.light,fontSize:11})}>{mcmcStep}</span>
                        <span style={ss({color:COLORS.purple,fontSize:11,fontWeight:700})}>{mcmcPct}%</span>
                      </div>
                      <div style={{height:4,background:"#0E2040",borderRadius:2,overflow:"hidden"}}>
                        <div style={{height:"100%",width:`${mcmcPct}%`,background:`linear-gradient(90deg,${COLORS.purple}66,${COLORS.purple})`,borderRadius:2,boxShadow:`0 0 8px ${COLORS.purple}`,transition:"width 0.4s ease"}}/>
                      </div>
                    </div>
                  )}
                  {!mcmcLoading&&!mcmcResult&&(
                    <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:"36px 20px",textAlign:"center"}}>
                      <div style={{fontSize:26,color:COLORS.purple,marginBottom:10}}>⟨ψ⟩</div>
                      <div style={{color:COLORS.white,fontSize:14,fontWeight:600,marginBottom:6}}>MCMC Bayesian Transit Fitting</div>
                      <div style={ss({color:COLORS.slate,fontSize:11,marginBottom:20})}>batman + emcee · Fits P · T₀ · Rp/R★ · a/R★ · inc · limb darkening</div>
                      {result?<button onClick={startMCMC} style={{background:"linear-gradient(135deg,#A78BFA22,#A78BFA11)",border:`1px solid ${COLORS.purple}`,color:COLORS.purple,borderRadius:8,padding:"10px 24px",fontSize:12,fontFamily:"Space Mono,monospace",cursor:"pointer",fontWeight:700}}>START MCMC → {selected}</button>
                        :<div style={ss({color:COLORS.slate,fontSize:12})}>Run a scan first</div>}
                    </div>
                  )}
                  {mcmcResult&&(
                    <>
                      <div style={{background:COLORS.panel,border:"1px solid #A78BFA44",borderRadius:12,padding:"11px 15px",display:"flex",gap:18,flexWrap:"wrap"}}>
                        {[{l:"STATUS",v:"CONVERGED",c:COLORS.green},{l:"SAMPLES",v:mcmcResult.n_samples?.toLocaleString(),c:COLORS.purple},{l:"ACCEPTANCE",v:`${(mcmcResult.acceptance_frac*100).toFixed(1)}%`,c:COLORS.cyan},{l:"ELAPSED",v:`${mcmcResult.elapsed_sec}s`,c:COLORS.amber}].map(({l,v,c})=>(
                          <div key={l}><div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1})}>{l}</div><div style={ss({fontSize:13,fontWeight:700,color:c,marginTop:2})}>{v}</div></div>
                        ))}
                      </div>
                      <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:14}}>
                        <div style={ss({fontSize:11,color:COLORS.slate,letterSpacing:1,marginBottom:4})}>POSTERIOR ESTIMATES · median +1σ/-1σ</div>
                        {Object.entries(mcmcResult.percentiles||{}).map(([name,d])=>(
                          <div key={name} style={{display:"flex",justifyContent:"space-between",padding:"6px 0",borderBottom:"1px solid #0E2040"}}>
                            <span style={ss({color:COLORS.slate,fontSize:11,minWidth:120})}>{name}</span>
                            <div style={{textAlign:"right"}}>
                              <span style={ss({color:COLORS.cyan,fontSize:12,fontWeight:700})}>{d.median.toFixed(4)}</span>
                              <span style={ss({color:COLORS.slate,fontSize:10,marginLeft:6})}>+{d.err_high.toFixed(4)} / -{d.err_low.toFixed(4)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                      {mcmcResult.plots?.posterior&&<div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,overflow:"hidden"}}><div style={{padding:"9px 14px",borderBottom:"1px solid #0E2040",fontSize:10,color:COLORS.slate,fontFamily:"Space Mono,monospace"}}>POSTERIOR TRANSIT MODEL</div><img src={`data:image/png;base64,${mcmcResult.plots.posterior}`} style={{width:"100%",display:"block"}} alt="posterior"/></div>}
                      {mcmcResult.plots?.corner&&<div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,overflow:"hidden"}}><div style={{padding:"9px 14px",borderBottom:"1px solid #0E2040",fontSize:10,color:COLORS.slate,fontFamily:"Space Mono,monospace"}}>CORNER PLOT</div><img src={`data:image/png;base64,${mcmcResult.plots.corner}`} style={{width:"100%",display:"block"}} alt="corner"/></div>}
                    </>
                  )}
                </>
              )}

              {/* ── BATCH TAB ── */}
              {activeTab==="batch"&&(
                <>
                  {/* Batch controls */}
                  <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:16}}>
                    <div style={ss({fontSize:12,color:COLORS.slate,letterSpacing:1,marginBottom:12})}>BATCH SECTOR SCAN SETTINGS</div>
                    <div style={{display:"flex",gap:12,alignItems:"center",flexWrap:"wrap"}}>
                      <div>
                        <div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1,marginBottom:4})}>TESS SECTOR</div>
                        <select value={batchSector} onChange={e=>setBatchSector(Number(e.target.value))}
                          style={{background:"#060E1E",border:"1px solid #0E2040",color:COLORS.cyan,borderRadius:6,padding:"6px 12px",fontSize:12,fontFamily:"Space Mono,monospace",outline:"none"}}>
                          {[13,14,15,27,40,56,70].map(s=><option key={s} value={s} style={{background:"#060E1E"}}>Sector {s}</option>)}
                        </select>
                      </div>
                      <div>
                        <div style={ss({fontSize:9,color:COLORS.slate,letterSpacing:1,marginBottom:4})}>MAX TARGETS</div>
                        <select value={batchMax} onChange={e=>setBatchMax(Number(e.target.value))}
                          style={{background:"#060E1E",border:"1px solid #0E2040",color:COLORS.amber,borderRadius:6,padding:"6px 12px",fontSize:12,fontFamily:"Space Mono,monospace",outline:"none"}}>
                          {[5,10,20,50].map(n=><option key={n} value={n} style={{background:"#060E1E"}}>{n} targets</option>)}
                        </select>
                      </div>
                      <button onClick={startBatch} disabled={batchLoading}
                        style={{background:batchLoading?"#0E2040":"linear-gradient(135deg,#FFB34722,#FFB34711)",border:`1px solid ${batchLoading?"#0E2040":COLORS.amber}`,color:batchLoading?COLORS.slate:COLORS.amber,borderRadius:8,padding:"9px 20px",fontSize:12,fontFamily:"Space Mono,monospace",cursor:batchLoading?"not-allowed":"pointer",fontWeight:700,marginTop:13}}>
                        {batchLoading?"SCANNING…":"START BATCH"}
                      </button>
                      {batchResult&&<button onClick={()=>downloadPDF("sector")}
                        style={{background:"linear-gradient(135deg,#39FF1422,#39FF1411)",border:`1px solid ${COLORS.green}`,color:COLORS.green,borderRadius:8,padding:"9px 20px",fontSize:12,fontFamily:"Space Mono,monospace",cursor:"pointer",fontWeight:700,marginTop:13}}>
                        ↓ SECTOR PDF
                      </button>}
                    </div>
                  </div>

                  {/* Progress */}
                  {batchLoading&&(
                    <div style={{background:COLORS.panel,border:"1px solid #FFB34744",borderRadius:12,padding:16}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                        <span style={ss({color:COLORS.light,fontSize:11})}>{batchStep}</span>
                        <span style={ss({color:COLORS.amber,fontSize:11,fontWeight:700})}>{batchPct}%</span>
                      </div>
                      <div style={{height:4,background:"#0E2040",borderRadius:2,overflow:"hidden"}}>
                        <div style={{height:"100%",width:`${batchPct}%`,background:`linear-gradient(90deg,${COLORS.amber}66,${COLORS.amber})`,borderRadius:2,boxShadow:`0 0 8px ${COLORS.amber}`,transition:"width 0.6s ease"}}/>
                      </div>
                    </div>
                  )}

                  {/* Results table */}
                  {batchResult&&(
                    <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,overflow:"hidden"}}>
                      <div style={{padding:"11px 15px",borderBottom:"1px solid #0E2040",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                        <div style={ss({fontSize:11,color:COLORS.slate,letterSpacing:1})}>
                          SECTOR {batchSector} · {batchResult.total_processed} processed · {batchResult.candidates} candidates
                        </div>
                      </div>
                      <div style={{overflowX:"auto"}}>
                        <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"Space Mono,monospace"}}>
                          <thead>
                            <tr style={{background:"#060E1E"}}>
                              {["TIC ID","P(planet)","Period","Depth","SNR","CNN","TF","Class","Flag"].map(h=>(
                                <th key={h} style={{padding:"8px 10px",color:COLORS.slate,fontWeight:700,textAlign:"left",borderBottom:"1px solid #0E2040",whiteSpace:"nowrap"}}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {(batchResult.top_candidates||[]).map((r,i)=>(
                              <tr key={i} style={{borderBottom:"1px solid #0E2040",background:i%2===0?"transparent":"#060E1E",opacity:r.status==="ok"?1:0.4}}>
                                <td style={{padding:"7px 10px",color:COLORS.white}}>{r.tic_id}</td>
                                <td style={{padding:"7px 10px",color:r.p_planet>0.7?COLORS.green:r.p_planet>0.4?COLORS.amber:COLORS.red,fontWeight:700}}>{r.p_planet?(r.p_planet*100).toFixed(1)+"%":"—"}</td>
                                <td style={{padding:"7px 10px",color:COLORS.cyan}}>{r.period_days?.toFixed(3)}d</td>
                                <td style={{padding:"7px 10px",color:COLORS.slate}}>{r.depth_pct?.toFixed(3)}%</td>
                                <td style={{padding:"7px 10px",color:COLORS.slate}}>{r.snr?.toFixed(1)}σ</td>
                                <td style={{padding:"7px 10px",color:COLORS.cyan}}>{r.p_cnn?(r.p_cnn*100).toFixed(0)+"%":"—"}</td>
                                <td style={{padding:"7px 10px",color:COLORS.purple}}>{r.p_transformer?(r.p_transformer*100).toFixed(0)+"%":"—"}</td>
                                <td style={{padding:"7px 10px",color:COLORS.slate,whiteSpace:"nowrap"}}>{r.top_class||"—"}</td>
                                <td style={{padding:"7px 10px"}}>{r.flag_mcmc&&<Badge text="★ MCMC" color={COLORS.purple}/>}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Right sidebar */}
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {/* Classification */}
              <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:14}}>
                <div style={ss({fontSize:11,color:COLORS.slate,letterSpacing:1,marginBottom:4})}>AI CLASSIFICATION</div>
                <div style={ss({fontSize:10,color:COLORS.slate,marginBottom:12})}>{result?.classifier||"Waiting…"}</div>
                {cls?(
                  <>
                    <div style={{background:`${topMeta.color}10`,border:`1px solid ${topMeta.color}44`,borderRadius:8,padding:"9px 11px",marginBottom:12,display:"flex",gap:10,alignItems:"center"}}>
                      <div style={{fontSize:20,color:topMeta.color}}>{topMeta.icon}</div>
                      <div>
                        <div style={{color:topMeta.color,fontSize:12,fontWeight:700}}>{cls.top_class}</div>
                        <div style={ss({color:COLORS.slate,fontSize:10})}>Confidence: {cls.confidence}%</div>
                      </div>
                    </div>
                    {Object.entries(cls.probabilities).map(([label,val])=>{
                      const m=CLASS_META[label]||{icon:"?",color:COLORS.slate};
                      return <ConfBar key={label} label={label} value={val} color={m.color} icon={m.icon}/>;
                    })}
                  </>
                ):(
                  <div style={{color:COLORS.slate,fontSize:11,fontFamily:"Space Mono,monospace",textAlign:"center",padding:"16px 0"}}>
                    {loading?"Classifying…":"Run scan"}
                  </div>
                )}
              </div>

              {/* MCMC sidebar status */}
              {(mcmcLoading||mcmcResult)&&(
                <div style={{background:COLORS.panel,border:`1px solid ${mcmcResult?"#A78BFA44":"#0E2040"}`,borderRadius:12,padding:14}}>
                  <div style={ss({fontSize:11,color:COLORS.purple,letterSpacing:1,marginBottom:10})}>MCMC STATUS</div>
                  {mcmcLoading&&(
                    <div>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                        <span style={ss({color:COLORS.light,fontSize:10})}>{mcmcStep}</span>
                        <span style={ss({color:COLORS.purple,fontSize:10,fontWeight:700})}>{mcmcPct}%</span>
                      </div>
                      <div style={{height:3,background:"#0E2040",borderRadius:2,overflow:"hidden"}}>
                        <div style={{height:"100%",width:`${mcmcPct}%`,background:COLORS.purple,borderRadius:2,transition:"width 0.4s ease"}}/>
                      </div>
                    </div>
                  )}
                  {mcmcResult&&[{l:"Samples",v:mcmcResult.n_samples?.toLocaleString()},{l:"Acceptance",v:`${(mcmcResult.acceptance_frac*100).toFixed(1)}%`},{l:"Runtime",v:`${mcmcResult.elapsed_sec}s`}].map(({l,v})=>(
                    <div key={l} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:"1px solid #0E2040"}}>
                      <span style={ss({color:COLORS.slate,fontSize:10})}>{l}</span>
                      <span style={ss({color:COLORS.white,fontSize:10})}>{v}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Data quality */}
              {dq&&(
                <div style={{background:COLORS.panel,border:"1px solid #0E2040",borderRadius:12,padding:14}}>
                  <div style={ss({fontSize:11,color:COLORS.slate,letterSpacing:1,marginBottom:10})}>DATA QUALITY</div>
                  {[{label:"Completeness",value:dq.completeness_pct,unit:"%",color:COLORS.green,max:100},{label:"CDPP",value:dq.cdpp_ppm,unit:" ppm",color:COLORS.cyan,max:1000},{label:"Sys. Noise",value:dq.sys_noise_ppm,unit:" ppm",color:COLORS.amber,max:200}].map(({label,value,unit,color,max})=>(
                    <div key={label} style={{marginBottom:8}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                        <span style={ss({color:COLORS.slate,fontSize:10})}>{label}</span>
                        <span style={ss({color,fontSize:10})}>{value}{unit}</span>
                      </div>
                      <div style={{height:2,background:"#0E2040",borderRadius:1,overflow:"hidden"}}>
                        <div style={{height:"100%",width:`${Math.min(100,(value/max)*100)}%`,background:color,opacity:0.8}}/>
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