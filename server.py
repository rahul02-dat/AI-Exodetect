"""
server_phase3.py — ExoDetect Backend v4

New in Phase 3:
  GET  /api/classify         → CNN + TransitFormer ensemble + attention weights
  POST /api/batch/start      → kick off full sector batch scan
  GET  /api/batch/status     → poll sector scan progress
  GET  /api/batch/results    → ranked candidate list
  GET  /api/report/candidate → download PDF for one candidate
  GET  /api/report/sector    → download PDF for full sector
  GET  /api/model-info       → CNN + Transformer metadata

Phase 4 update:
  Stellar feature construction (period/duration/depth/SNR + optional
  Teff/radius/log g/Tmag) now goes through the shared stellar_features
  module instead of being hand-rolled here, and each model's tensor is
  sized to that model's own n_stellar — see stellar_features.py and the
  ExoEnsemble.predict() changes in transit_transformer.py for why. This
  also adds a live TIC catalog lookup (fetch_stellar_params) so real
  Teff/radius/log g/Tmag values are used when a model checkpoint expects
  the expanded 8-feature schema, rather than falling back to imputed
  defaults for every request.

Drop this alongside server_phase2.py (or replace it).
"""

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
import torch, torch.nn as nn
import redis, json, io, traceback, warnings, os
warnings.filterwarnings("ignore")

# Phase 2 imports
from tasks import celery_app as phase2_celery, run_mcmc_task
from mcmc_fitter import run_mcmc

# Phase 3 imports
from transit_transformer import (
    TransitFormer, ExoEnsemble, build_transformer, load_transformer, get_device
)
from batch_pipeline import celery_app as batch_celery, run_sector
from report_generator import generate_candidate_report, generate_sector_report

# Phase 4 import — shared stellar feature engineering (see module docstring)
from stellar_features import build_stellar_features, fetch_stellar_params, extract_tic_id

app   = Flask(__name__)
CORS(app)
REDIS = redis.Redis(host="localhost", port=6379, db=0)
DEVICE = get_device()
print(f"[INFO] Device: {DEVICE}")

# ── Model loading ──────────────────────────────────────────────────────────

BACKEND_DIR      = os.path.dirname(__file__)
CNN_PATH         = os.path.join(BACKEND_DIR, "exodetect_cnn.pt")
TRANSFORMER_PATH = os.path.join(BACKEND_DIR, "exodetect_transformer.pt")

CNN_MODEL = None
TF_MODEL  = None
ENSEMBLE  = None
CNN_META  = {}
TF_META   = {}


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k=5, p=2):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(oc)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(p)
        
        self.shortcut = nn.Sequential()
        if ic != oc:
            self.shortcut = nn.Sequential(
                nn.Conv1d(ic, oc, 1),
                nn.BatchNorm1d(oc)
            )

    def forward(self, x):
        out = self.bn(self.conv(x))
        out += self.shortcut(x)
        out = self.relu(out)
        return self.pool(out)


class SafeAdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
    def forward(self, x):
        if x.device.type == "mps":
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)

class ExoDetectCNN(nn.Module):
    def __init__(self, global_len=201, local_len=81, n_stellar=4):
        super().__init__()
        self.n_stellar = n_stellar   # Phase 4: recorded so ExoEnsemble.predict()
                                      # can size this model's stellar tensor correctly.
        self.global_branch = nn.Sequential(
            ConvBlock(1,16,5,2), ConvBlock(16,32,5,2),
            ConvBlock(32,64,5,2), ConvBlock(64,128,3,2),
            SafeAdaptiveAvgPool1d(8), nn.Flatten())
        self.local_branch = nn.Sequential(
            ConvBlock(1,16,5,2), ConvBlock(16,32,5,2),
            ConvBlock(32,64,3,2), SafeAdaptiveAvgPool1d(4), nn.Flatten())
        fused = (128*8)+(64*4)+n_stellar
        self.head = nn.Sequential(
            nn.Linear(fused,512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512,256),   nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256,2))
    def forward(self, gv, lv, sf):
        return self.head(torch.cat([self.global_branch(gv), self.local_branch(lv), sf], 1))


def load_models():
    global CNN_MODEL, TF_MODEL, ENSEMBLE, CNN_META, TF_META
    if os.path.exists(CNN_PATH):
        ckpt = torch.load(CNN_PATH, map_location="cpu")
        CNN_MODEL = ExoDetectCNN(**ckpt["model_config"]).to(DEVICE)
        CNN_MODEL.load_state_dict(ckpt["model_state_dict"])
        CNN_MODEL.eval()
        CNN_META = ckpt.get("metrics", {})
        print(f"[INFO] CNN loaded — Val AUC {CNN_META.get('best_val_auc','?')} "
              f"(n_stellar={CNN_MODEL.n_stellar})")
    else:
        print(f"[WARN] No CNN model at {CNN_PATH}")

    if os.path.exists(TRANSFORMER_PATH):
        TF_MODEL, TF_META = load_transformer(TRANSFORMER_PATH, device=DEVICE)
        print(f"[INFO] TransitFormer loaded — Val AUC {TF_META.get('best_val_auc','?')} "
              f"(n_stellar={TF_MODEL.n_stellar})")
    else:
        print(f"[WARN] No TransitFormer at {TRANSFORMER_PATH}")
        print("       Run: python train_transformer.py")

    if CNN_MODEL and TF_MODEL:
        ENSEMBLE = ExoEnsemble(CNN_MODEL, TF_MODEL, device=DEVICE)
        print("[INFO] Ensemble ready (CNN + TransitFormer)")
        if CNN_MODEL.n_stellar != TF_MODEL.n_stellar:
            print(f"[INFO] Note: CNN (n_stellar={CNN_MODEL.n_stellar}) and TransitFormer "
                  f"(n_stellar={TF_MODEL.n_stellar}) are on different stellar-feature "
                  f"schema versions — this is fine, each is fed its own correctly-sized "
                  f"feature vector, but consider retraining both to the same schema "
                  f"once convenient.")
    elif CNN_MODEL:
        print("[INFO] Running CNN-only mode (no TransitFormer yet)")


# ── Preprocessing ──────────────────────────────────────────────────────────

GLOBAL_LEN = 201
LOCAL_LEN  = 81

def _smooth(flux, w=51):
    return flux / median_filter(flux.astype(float), size=w, mode="reflect")

def _fold_bin(time, flux, period, t0, n_bins, local=False, lw=0.2):
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)
    if local:
        m = np.abs(phase) < lw; phase, flux = phase[m], flux[m]
        if len(phase) < 5: return None
        p0, p1 = -lw, lw
    else:
        p0, p1 = -0.5, 0.5
    bins = np.linspace(p0, p1, n_bins + 1)
    idx  = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    view = np.array([np.median(flux[idx==b]) if np.any(idx==b) else 1.0
                     for b in range(n_bins)])
    view = np.where(np.isfinite(view), view, 1.0)
    oot  = np.abs(np.linspace(p0, p1, n_bins)) > 0.05
    if oot.sum() > 3:
        mu, sig = np.median(view[oot]), np.std(view[oot]) + 1e-8
        view = (view - mu) / sig
    return view.astype(np.float32)

def _view_tensors(gv, lv):
    """Global/local view tensors — these don't depend on which model consumes
    them, unlike the stellar feature tensor (see _stellar_tensor_for)."""
    gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0).to(DEVICE)
    lv_t = torch.tensor(lv).unsqueeze(0).unsqueeze(0).to(DEVICE) if lv is not None else None
    return gv_t, lv_t

def _stellar_tensor_for(model, period, dur_hr, depth_ppm, snr, stellar_params):
    """Builds a stellar feature tensor sized to whatever a specific model
    checkpoint expects (model.n_stellar) — see stellar_features.py."""
    sf = build_stellar_features(
        model.n_stellar, period, dur_hr, depth_ppm, snr,
        stellar_params.get("teff"), stellar_params.get("rad"),
        stellar_params.get("logg"), stellar_params.get("tmag"),
    )
    return torch.tensor([sf]).to(DEVICE)

def _heuristic(period, depth_pct, dur_hr, snr):
    if depth_pct > 5: return {"Exoplanet Transit": 8.0, "Eclipsing Binary": 81.0, "Stellar Blend": 7.0, "Starspot": 4.0}
    if snr < 7:       return {"Exoplanet Transit": 30.0,"Eclipsing Binary": 20.0, "Stellar Blend": 25.0,"Starspot": 25.0}
    return             {"Exoplanet Transit": 91.0,"Eclipsing Binary": 5.0,  "Stellar Blend": 3.0, "Starspot": 1.0}


def classify_with_ensemble(gv, lv, period, dur_hr, depth_ppm, snr, stellar_params=None):
    """Returns probs dict, p_cnn, p_tf, attn_weights.

    stellar_params: optional dict with teff/rad/logg/tmag (see
    fetch_stellar_params in stellar_features.py). Safe to omit or leave
    fields as None — build_stellar_features() imputes neutral defaults."""
    stellar_params = stellar_params or {}

    if ENSEMBLE and gv is not None and lv is not None:
        gv_t, lv_t = _view_tensors(gv, lv)
        p, p_cnn, p_tf, attn = ENSEMBLE.predict(
            gv_t, lv_t, period, dur_hr, depth_ppm, snr,
            teff=stellar_params.get("teff"), rad=stellar_params.get("rad"),
            logg=stellar_params.get("logg"), tmag=stellar_params.get("tmag"),
        )
        probs = ENSEMBLE.classify(p, depth_ppm, period, dur_hr)
        return probs, p_cnn, p_tf, attn.tolist() if attn is not None else None
    elif CNN_MODEL and gv is not None and lv is not None:
        gv_t, lv_t = _view_tensors(gv, lv)
        sf_t = _stellar_tensor_for(CNN_MODEL, period, dur_hr, depth_ppm, snr, stellar_params)
        with torch.no_grad():
            logits = CNN_MODEL(gv_t, lv_t, sf_t)
            p_cnn  = float(torch.softmax(logits, dim=1)[:, 1].item())
        p_fp = 1 - p_cnn
        depth_f = depth_ppm / 1e4
        if depth_f > 5: eb,bl,sp = 0.65,0.25,0.10
        elif dur_hr/(period*24+1e-6) > 0.15: eb,bl,sp = 0.20,0.65,0.15
        else: eb,bl,sp = 0.35,0.40,0.25
        probs = {
            "Exoplanet Transit": round(p_cnn*100,1),
            "Eclipsing Binary":  round(p_fp*eb*100,1),
            "Stellar Blend":     round(p_fp*bl*100,1),
            "Starspot":          round(p_fp*sp*100,1),
        }
        return probs, p_cnn, None, None
    else:
        probs = _heuristic(period, depth_ppm/1e4, dur_hr, snr)
        return probs, None, None, None


KNOWN_TARGETS = {
    "L 98-59":       {"tic":"TIC 307210830","note":"3 terrestrial planets"},
    "TOI-700":       {"tic":"TIC 150428135","note":"Habitable-zone Earth-size"},
    "WASP-18":       {"tic":"TIC 100100827","note":"Hot Jupiter, ~1 day period"},
    "TIC 286923464": {"tic":"TIC 286923464","note":"HD 118203 b — eccentric"},
    "HD 21749":      {"tic":"TIC 279741379","note":"Sub-Neptune + super-Earth"},
}


def fetch_and_analyse(target_name):
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0: search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0: raise ValueError(f"No TESS data for '{target_name}'")

    # Phase 4: resolve TIC ID + live stellar params up front, once, so both
    # the CNN and TransitFormer (and the PDF report) can use real Teff/
    # radius/log g/Tmag when available instead of imputed neutral defaults.
    tic_id = extract_tic_id(search[0])
    stellar_params = fetch_stellar_params(tic_id) if tic_id is not None else {}

    lc = search[0].download(flux_column="pdcsap_flux")
    lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

    time  = np.array(lc.time.value)
    flux  = np.array(lc.flux.value)
    ferr  = np.array(lc.flux_err.value) if lc.flux_err is not None else np.full_like(flux, 5e-4)
    sector_val = search[0].sequence_number if hasattr(search[0],"sequence_number") else None
    sector = int(np.atleast_1d(sector_val)[0]) if sector_val is not None else "?"
    exp_val = search[0].exptime.value if hasattr(search[0],"exptime") else None
    exptime = float(np.atleast_1d(exp_val)[0]) if exp_val is not None else 120.0

    bls     = BoxLeastSquares(time, flux, ferr)
    periods = np.linspace(0.51, 27.0, 5000)
    power   = bls.power(periods, np.linspace(0.01, 0.4, 20))
    bi      = np.argmax(power.power)
    period  = float(power.period[bi]); dur = float(power.duration[bi])
    t0      = float(power.transit_time[bi]); depth = float(power.depth[bi])

    stride = max(1, len(periods)//500)
    bls_out = {
        "periods": power.period[::stride].tolist(),
        "power":   (power.power[::stride]/np.max(power.power)).tolist(),
        "peak_period": period,
    }

    in_tr  = np.abs(((time-t0)%period)-period/2) < dur/2
    noise  = np.std(flux[~in_tr]) if (~in_tr).sum()>10 else 1e-4
    sig    = abs(np.mean(flux[in_tr])-np.mean(flux[~in_tr])) if in_tr.sum()>0 else 0
    snr    = float(sig/noise*np.sqrt(max(in_tr.sum(),1)))
    dur_hr = dur*24; depth_ppm = depth*1e6

    phase   = ((time-t0+period/2)%period)/period-0.5
    srt     = np.argsort(phase)

    sf      = _smooth(flux)
    gv      = _fold_bin(time, sf, period, t0, GLOBAL_LEN)
    lv      = _fold_bin(time, sf, period, t0, LOCAL_LEN, local=True)

    probs, p_cnn, p_tf, attn = classify_with_ensemble(
        gv, lv, period, dur_hr, depth_ppm, snr, stellar_params
    )
    top_class = max(probs, key=probs.get)
    cdpp      = float(lc.estimate_cdpp().value) if hasattr(lc,"estimate_cdpp") else 200.0
    comp      = float(100*(1-np.isnan(lc.flux.value).sum()/len(lc.flux.value)))

    return {
        "target":       target_name,
        "tic_id":       tic_id,
        "stellar_params": stellar_params,
        "sector":       sector,
        "exptime_sec":  exptime,
        "n_cadences":   len(time),
        "classifier":   "Ensemble (CNN+TF)" if ENSEMBLE else ("CNN" if CNN_MODEL else "Heuristic"),
        "model_auc":    CNN_META.get("test_auc"),
        "light_curve":  {"time":time.tolist(),"flux":flux.tolist(),"flux_err":ferr.tolist()},
        "bls":          bls_out,
        "phase_folded": {"phase":phase[srt].tolist(),"flux":flux[srt].tolist()},
        "attention_weights": attn,
        "p_cnn":        round(float(p_cnn)*100,1) if p_cnn else None,
        "p_transformer":round(float(p_tf)*100,1)  if p_tf  else None,
        "transit_params": {
            "period_days":    round(period,4),
            "period_err":     round(period*0.001,5),
            "depth_pct":      round(depth*100,4),
            "duration_hours": round(dur_hr,3),
            "duration_err":   round(dur_hr*0.05,4),
            "t0_bjd":         round(t0,5),
            "n_transits":     max(1,int((time[-1]-time[0])/period)),
            "snr":            round(snr,2),
            "rp_rs":          round(np.sqrt(max(depth,0)),4),
        },
        "classification": {
            "top_class":     top_class,
            "confidence":    round(probs[top_class],1),
            "probabilities": probs,
        },
        "data_quality": {
            "completeness_pct": round(comp,1),
            "cdpp_ppm":         round(cdpp,1),
            "sys_noise_ppm":    round(cdpp*0.15,1),
        },
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/api/targets")
def get_targets():
    return jsonify({"targets":[{"name":k,**v} for k,v in KNOWN_TARGETS.items()]})

@app.route("/api/analyse")
def analyse():
    target = request.args.get("target","L 98-59")
    try:
        return jsonify({"status":"ok","data":fetch_and_analyse(target)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

@app.route("/api/health")
def health():
    return jsonify({
        "status":"ok","device":str(DEVICE),
        "cnn":CNN_MODEL is not None,
        "transformer":TF_MODEL is not None,
        "ensemble":ENSEMBLE is not None,
    })

@app.route("/api/model-info")
def model_info():
    return jsonify({
        "phase":3,
        "device":str(DEVICE),
        "cnn":{"loaded":CNN_MODEL is not None,"metrics":CNN_META,
               "n_stellar": CNN_MODEL.n_stellar if CNN_MODEL else None},
        "transformer":{"loaded":TF_MODEL is not None,"metrics":TF_META,
                        "n_stellar": TF_MODEL.n_stellar if TF_MODEL else None},
        "ensemble_weights":{"cnn":0.45,"transformer":0.55} if ENSEMBLE else None,
        "mcmc_backend":"emcee 3.1 + batman 2.4",
    })

# Phase 2 MCMC routes (unchanged)
@app.route("/api/mcmc/start", methods=["POST"])
def mcmc_start():
    body   = request.get_json(force=True,silent=True) or {}
    task   = run_mcmc_task.apply_async(args=[
        body.get("target","L 98-59"),
        int(body.get("n_walkers",32)),
        int(body.get("n_steps",2000)),
        int(body.get("n_burn",500)),
    ])
    return jsonify({"job_id":task.id,"status":"queued"})

@app.route("/api/mcmc/status")
def mcmc_status():
    job_id = request.args.get("job_id")
    task   = phase2_celery.AsyncResult(job_id)
    if task.state == "PENDING":   return jsonify({"state":"PENDING","step":"Queued…","pct":0})
    if task.state == "PROGRESS":
        m = task.info or {}
        return jsonify({"state":"PROGRESS","step":m.get("step",""),"pct":m.get("pct",0)})
    if task.state == "SUCCESS":
        r = task.result
        if r.get("status") == "error": return jsonify({"state":"FAILURE","message":r.get("message")})
        return jsonify({"state":"SUCCESS","pct":100,"step":"Done",
                        "elapsed_sec":r.get("elapsed_sec"),"acceptance":r.get("acceptance_frac")})
    return jsonify({"state":"FAILURE","message":str(task.info)}),500

@app.route("/api/mcmc/result")
def mcmc_result():
    job_id = request.args.get("job_id")
    task   = phase2_celery.AsyncResult(job_id)
    if task.state != "SUCCESS": return jsonify({"error":"Not done"}),400
    r = task.result
    if r.get("status") == "error": return jsonify({"status":"error","message":r.get("message")}),500
    return jsonify({"status":"ok","data":r})

# Phase 3 Batch routes
@app.route("/api/batch/start", methods=["POST"])
def batch_start():
    body    = request.get_json(force=True, silent=True) or {}
    sector  = int(body.get("sector", 14))
    max_t   = int(body.get("max_targets", 10))
    print(f"[API] Batch start → sector={sector}, max_targets={max_t}")
    task    = run_sector.apply_async(args=[sector, max_t])
    print(f"[API] Batch job queued → {task.id}")
    return jsonify({"job_id": task.id, "sector": sector, "status": "queued"})

@app.route("/api/batch/status")
def batch_status():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    task = batch_celery.AsyncResult(job_id)

    if task.state == "PENDING":
        return jsonify({"state": "PENDING", "pct": 0,
                        "step": "Job queued — waiting for worker…",
                        "candidates_found": 0})

    if task.state == "PROGRESS":
        m = task.info or {}
        return jsonify({"state": "PROGRESS",
                        "pct":  m.get("pct", 0),
                        "step": m.get("step", "Processing…"),
                        "candidates_found": m.get("candidates_found", 0)})

    if task.state == "SUCCESS":
        r = task.result or {}
        if r.get("status") == "error":
            return jsonify({"state": "FAILURE", "message": r.get("message", "Unknown error")})
        return jsonify({"state": "SUCCESS", "pct": 100,
                        "step": "Done",
                        "candidates_found": r.get("candidates", 0),
                        "total_processed":  r.get("total_processed", 0)})

    # FAILURE / REVOKED
    return jsonify({"state": "FAILURE",
                    "message": str(task.info) if task.info else "Task failed"}), 500

@app.route("/api/batch/results")
def batch_results():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    task = batch_celery.AsyncResult(job_id)
    if task.state != "SUCCESS":
        return jsonify({"error": f"Job not complete yet (state={task.state})"}), 400
    result = task.result
    if not result:
        return jsonify({"error": "Empty result from worker"}), 500
    if result.get("status") == "error":
        return jsonify({"status": "error", "message": result.get("message")}), 500
    return jsonify({"status": "ok", "data": result})


@app.route("/api/debug/batch")
def debug_batch():
    """
    Runs _process_one synchronously — no Celery needed.
    GET /api/debug/batch?target=L+98-59
    """
    from batch_pipeline import _process_one, _get_worker_models
    target = request.args.get("target", "L 98-59")
    try:
        models = _get_worker_models()
        result = _process_one(target, None, models)
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/batch/test")
def batch_test():
    """
    Returns a mock batch result instantly — no MAST download needed.
    Use this to verify the frontend table renders correctly before
    running a real batch job.
    GET /api/batch/test
    """
    mock = {
        "sector": 14,
        "total_processed": 5,
        "successful": 5,
        "candidates": 2,
        "errors": 0,
        "status": "done",
        "top_candidates": [
            {"tic_id": "L 98-59", "status": "ok", "period_days": 3.6906,
             "depth_pct": 0.358, "duration_hr": 1.21, "snr": 22.4,
             "p_planet": 0.9421, "p_cnn": 0.93, "p_transformer": 0.95,
             "top_class": "Exoplanet Transit", "confidence": 94.2,
             "probabilities": {"Exoplanet Transit": 94.2, "Eclipsing Binary": 3.1,
                               "Stellar Blend": 1.9, "Starspot": 0.8},
             "flag_mcmc": True, "n_cadences": 18241, "note": "3 confirmed planets"},
            {"tic_id": "TOI-700", "status": "ok", "period_days": 37.426,
             "depth_pct": 0.194, "duration_hr": 2.08, "snr": 14.1,
             "p_planet": 0.8833, "p_cnn": 0.87, "p_transformer": 0.90,
             "top_class": "Exoplanet Transit", "confidence": 88.3,
             "probabilities": {"Exoplanet Transit": 88.3, "Eclipsing Binary": 6.2,
                               "Stellar Blend": 4.1, "Starspot": 1.4},
             "flag_mcmc": True, "n_cadences": 21034, "note": "Habitable-zone Earth"},
            {"tic_id": "WASP-18", "status": "ok", "period_days": 0.9415,
             "depth_pct": 1.12, "duration_hr": 2.14, "snr": 38.7,
             "p_planet": 0.6120, "p_cnn": 0.62, "p_transformer": None,
             "top_class": "Exoplanet Transit", "confidence": 61.2,
             "probabilities": {"Exoplanet Transit": 61.2, "Eclipsing Binary": 25.1,
                               "Stellar Blend": 9.3, "Starspot": 4.4},
             "flag_mcmc": False, "n_cadences": 14882, "note": "Hot Jupiter 0.94d"},
            {"tic_id": "HD 21749", "status": "ok", "period_days": 35.614,
             "depth_pct": 0.451, "duration_hr": 3.11, "snr": 9.2,
             "p_planet": 0.4421, "p_cnn": 0.44, "p_transformer": None,
             "top_class": "Stellar Blend", "confidence": 42.1,
             "probabilities": {"Exoplanet Transit": 44.2, "Eclipsing Binary": 12.0,
                               "Stellar Blend": 37.4, "Starspot": 6.4},
             "flag_mcmc": False, "n_cadences": 19872, "note": "Multi-planet system"},
            {"tic_id": "TIC 286923464", "status": "ok", "period_days": 6.1354,
             "depth_pct": 0.827, "duration_hr": 4.22, "snr": 7.1,
             "p_planet": 0.3214, "p_cnn": 0.32, "p_transformer": None,
             "top_class": "Eclipsing Binary", "confidence": 48.3,
             "probabilities": {"Exoplanet Transit": 32.1, "Eclipsing Binary": 48.3,
                               "Stellar Blend": 15.2, "Starspot": 4.4},
             "flag_mcmc": False, "n_cadences": 16341, "note": "HD 118203 b eccentric"},
        ]
    }
    return jsonify({"status": "ok", "data": mock})

# Phase 3 Report routes
@app.route("/api/report/candidate")
def report_candidate():
    """Generate and stream PDF for one candidate."""
    target = request.args.get("target","L 98-59")
    job_id = request.args.get("mcmc_job_id")   # optional

    try:
        scan   = fetch_and_analyse(target)
        mcmc   = None
        if job_id:
            task = phase2_celery.AsyncResult(job_id)
            if task.state == "SUCCESS":
                mcmc = task.result

        pdf_bytes = generate_candidate_report(scan, mcmc)
        fname     = f"ExoDetect_{target.replace(' ','_')}.pdf"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=fname,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/api/report/sector")
def report_sector():
    """Generate sector summary PDF from a completed batch job."""
    job_id = request.args.get("job_id")
    sector = request.args.get("sector",14)
    if not job_id: return jsonify({"error":"job_id required"}),400
    task = batch_celery.AsyncResult(job_id)
    if task.state != "SUCCESS": return jsonify({"error":"Batch not done"}),400

    result    = task.result
    pdf_bytes = generate_sector_report(sector, result.get("top_candidates",[]))
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"ExoDetect_Sector_{sector}.pdf",
    )


if __name__ == "__main__":
    load_models()
    print("="*60)
    print("  ExoDetect Backend v4 — Phase 3 (CNN + TF + Batch + PDF)")
    print(f"  Device : {DEVICE}")
    print("  http://localhost:8000")
    print("="*60)
    app.run(debug=True, port=8000)