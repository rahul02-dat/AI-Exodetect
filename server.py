# """
# server_phase2.py — ExoDetect Backend v3

# New in Phase 2:
#   POST /api/mcmc/start   → kick off async MCMC job, returns job_id
#   GET  /api/mcmc/status  → poll job progress (step, pct)
#   GET  /api/mcmc/result  → fetch completed results (plots, posteriors)
#   GET  /api/model-info   → CNN + MCMC metadata

# Apple Silicon note:
#   PyTorch uses MPS automatically when available.
#   All MCMC runs on CPU (numpy/scipy) — MPS not needed for emcee.
# """

# from flask import Flask, jsonify, request, Response
# from flask_cors import CORS
# import numpy as np
# import lightkurve as lk
# from astropy.timeseries import BoxLeastSquares
# from scipy.ndimage import median_filter
# import torch, torch.nn as nn
# import traceback, warnings, os, json
# warnings.filterwarnings("ignore")

# from tasks import celery_app, run_mcmc_task

# app = Flask(__name__)
# CORS(app)

# # ── Device — MPS has operator gaps (adaptive_avg_pool1d, float64), use CPU ──
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")
#     print("[INFO] Using CPU for inference (MPS skipped – operator gaps)")


# # ── CNN model (identical to Phase 1) ─────────────────────────────────────

# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel=5, pool=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
#             nn.BatchNorm1d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(pool),
#         )
#     def forward(self, x): return self.net(x)


# class ExoDetectCNN(nn.Module):
#     def __init__(self, global_len=201, local_len=81, n_stellar=4):
#         super().__init__()
#         self.global_branch = nn.Sequential(
#             ConvBlock(1,  16, 5, 2), ConvBlock(16, 32, 5, 2),
#             ConvBlock(32, 64, 5, 2), ConvBlock(64, 128, 3, 2),
#             nn.AdaptiveAvgPool1d(8), nn.Flatten(),
#         )
#         self.local_branch = nn.Sequential(
#             ConvBlock(1, 16, 5, 2), ConvBlock(16, 32, 5, 2),
#             ConvBlock(32, 64, 3, 2), nn.AdaptiveAvgPool1d(4), nn.Flatten(),
#         )
#         fused = (128 * 8) + (64 * 4) + n_stellar
#         self.head = nn.Sequential(
#             nn.Linear(fused, 512), nn.ReLU(True), nn.Dropout(0.5),
#             nn.Linear(512, 256),   nn.ReLU(True), nn.Dropout(0.3),
#             nn.Linear(256, 2),
#         )

#     def forward(self, gv, lv, sf):
#         return self.head(torch.cat([self.global_branch(gv),
#                                     self.local_branch(lv), sf], dim=1))

#     def predict_proba(self, gv, lv, sf):
#         with torch.no_grad():
#             return torch.softmax(self.forward(gv, lv, sf), dim=1)[:, 1].item()


# MODEL, MODEL_META = None, {}

# def load_model(path=None):
#     global MODEL, MODEL_META
#     path = path or os.path.join(os.path.dirname(__file__), "exodetect_cnn.pt")
#     if not os.path.exists(path):
#         print(f"[WARN] No model at {path} — heuristic fallback active")
#         return
#     ckpt       = torch.load(path, map_location="cpu")
#     cfg        = ckpt["model_config"]
#     MODEL      = ExoDetectCNN(**cfg).to(DEVICE)
#     MODEL.load_state_dict(ckpt["model_state_dict"])
#     MODEL.eval()
#     MODEL_META = ckpt.get("metrics", {})
#     print(f"[INFO] CNN loaded on {DEVICE} — Val AUC {MODEL_META.get('best_val_auc','?')}")


# # ── Preprocessing (must match notebook exactly) ───────────────────────────

# GLOBAL_VIEW_LEN = 201
# LOCAL_VIEW_LEN  = 81

# def _smooth(flux, w=51):
#     return flux / median_filter(flux.astype(float), size=w, mode="reflect")

# def _fold_bin(time, flux, period, t0, n_bins, local=False, lw=0.2):
#     phase = ((time - t0) % period) / period
#     phase = np.where(phase > 0.5, phase - 1, phase)
#     if local:
#         m = np.abs(phase) < lw
#         phase, flux = phase[m], flux[m]
#         if len(phase) < 5: return None
#         p0, p1 = -lw, lw
#     else:
#         p0, p1 = -0.5, 0.5
#     bins = np.linspace(p0, p1, n_bins + 1)
#     idx  = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
#     view = np.array([np.median(flux[idx == b]) if np.any(idx == b) else 1.0
#                      for b in range(n_bins)])
#     view = np.where(np.isfinite(view), view, 1.0)
#     oot  = np.abs(np.linspace(p0, p1, n_bins)) > 0.05
#     if oot.sum() > 3:
#         mu, sig = np.median(view[oot]), np.std(view[oot]) + 1e-8
#         view = (view - mu) / sig
#     return view.astype(np.float32)

# def cnn_classify(period, dur_hr, depth_ppm, snr, time, flux):
#     if MODEL is None:
#         return _heuristic(period, depth_ppm / 1e4, dur_hr, snr)
#     sf = _smooth(flux)
#     t0 = time[np.argmin(flux)]
#     gv = _fold_bin(time, sf, period, t0, GLOBAL_VIEW_LEN)
#     lv = _fold_bin(time, sf, period, t0, LOCAL_VIEW_LEN, local=True)
#     if gv is None or lv is None:
#         return _heuristic(period, depth_ppm / 1e4, dur_hr, snr)
#     gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
#     lv_t = torch.tensor(lv).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
#     sf_t = torch.tensor([[np.log1p(period),
#                           np.log1p(max(dur_hr, 0)),
#                           np.log1p(max(depth_ppm, 0)) / 10,
#                           np.clip(snr, 0, 100) / 100]]).float().to(DEVICE)
#     p_planet = MODEL.predict_proba(gv_t, lv_t, sf_t)
#     p_fp     = 1.0 - p_planet
#     if depth_ppm > 50000:
#         eb, bl, sp = 0.65, 0.25, 0.10
#     elif dur_hr / (period * 24 + 1e-6) > 0.15:
#         eb, bl, sp = 0.20, 0.65, 0.15
#     else:
#         eb, bl, sp = 0.35, 0.40, 0.25
#     return {
#         "Exoplanet Transit": round(p_planet * 100, 1),
#         "Eclipsing Binary":  round(p_fp * eb * 100, 1),
#         "Stellar Blend":     round(p_fp * bl * 100, 1),
#         "Starspot":          round(p_fp * sp * 100, 1),
#     }

# def _heuristic(period, depth_pct, dur_hr, snr):
#     if depth_pct > 5:
#         return {"Exoplanet Transit": 8.0, "Eclipsing Binary": 81.0,
#                 "Stellar Blend": 7.0, "Starspot": 4.0}
#     if snr < 7:
#         return {"Exoplanet Transit": 30.0, "Eclipsing Binary": 20.0,
#                 "Stellar Blend": 25.0, "Starspot": 25.0}
#     return {"Exoplanet Transit": 91.0, "Eclipsing Binary": 5.0,
#             "Stellar Blend": 3.0, "Starspot": 1.0}


# # ── Core BLS pipeline (quick scan) ───────────────────────────────────────

# KNOWN_TARGETS = {
#     "L 98-59":       {"tic": "TIC 307210830", "note": "3 terrestrial planets"},
#     "TOI-700":       {"tic": "TIC 150428135", "note": "Habitable-zone Earth-size"},
#     "WASP-18":       {"tic": "TIC 100100827", "note": "Hot Jupiter, ~1 day period"},
#     "TIC 286923464": {"tic": "TIC 286923464", "note": "HD 118203 b — eccentric"},
#     "HD 21749":      {"tic": "TIC 279741379", "note": "Sub-Neptune + super-Earth"},
# }

# def fetch_and_analyse(target_name):
#     search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
#     if len(search) == 0:
#         search = lk.search_lightcurve(target_name, mission="TESS")
#     if len(search) == 0:
#         raise ValueError(f"No TESS data for '{target_name}'")

#     lc = search[0].download(flux_column="pdcsap_flux")
#     lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

#     time  = np.array(lc.time.value)
#     flux  = np.array(lc.flux.value)
#     ferr  = np.array(lc.flux_err.value) if lc.flux_err is not None \
#             else np.full_like(flux, 5e-4)

#     sector  = int(search[0].sequence_number) if hasattr(search[0], "sequence_number") else "?"
#     exptime = float(np.atleast_1d(search[0].exptime.value)[0]) if hasattr(search[0], "exptime") else 120.0

#     bls      = BoxLeastSquares(time, flux, ferr)
#     periods  = np.linspace(0.5, 27.0, 5000)
#     # max duration must be strictly < min period (0.5 d); cap at 0.4 d
#     durations = np.linspace(0.01, 0.4, 20)
#     power    = bls.power(periods, durations)
#     bi       = np.argmax(power.power)

#     period   = float(power.period[bi])
#     dur      = float(power.duration[bi])
#     t0       = float(power.transit_time[bi])
#     depth    = float(power.depth[bi])

#     stride = max(1, len(periods) // 500)
#     bls_out = {
#         "periods":     power.period[::stride].tolist(),
#         "power":       (power.power[::stride] / np.max(power.power)).tolist(),
#         "peak_period": period,
#     }

#     in_tr  = np.abs(((time - t0) % period) - period / 2) < dur / 2
#     noise  = np.std(flux[~in_tr]) if (~in_tr).sum() > 10 else 1e-4
#     sig    = abs(np.mean(flux[in_tr]) - np.mean(flux[~in_tr])) if in_tr.sum() > 0 else 0
#     snr    = float(sig / noise * np.sqrt(max(in_tr.sum(), 1)))

#     phase   = ((time - t0 + period / 2) % period) / period - 0.5
#     srt     = np.argsort(phase)
#     phase_folded = {"phase": phase[srt].tolist(), "flux": flux[srt].tolist()}

#     probs     = cnn_classify(period, dur * 24, depth * 1e6, snr, time, flux)
#     top_class = max(probs, key=probs.get)

#     cdpp         = float(lc.estimate_cdpp().value) if hasattr(lc, "estimate_cdpp") else 200.0
#     completeness = 100 * (1 - np.isnan(lc.flux.value).sum() / len(lc.flux.value))

#     return {
#         "target":      target_name,
#         "sector":      sector,
#         "exptime_sec": exptime,
#         "n_cadences":  len(time),
#         "classifier":  "CNN (ExoDetectCNN)" if MODEL else "Heuristic",
#         "model_auc":   MODEL_META.get("test_auc"),
#         "light_curve": {"time": time.tolist(), "flux": flux.tolist(),
#                         "flux_err": ferr.tolist()},
#         "bls":         bls_out,
#         "phase_folded": phase_folded,
#         "transit_params": {
#             "period_days":    round(period, 4),
#             "period_err":     round(period * 0.001, 5),
#             "depth_pct":      round(depth * 100, 4),
#             "duration_hours": round(dur * 24, 3),
#             "duration_err":   round(dur * 24 * 0.05, 4),
#             "t0_bjd":         round(t0, 5),
#             "n_transits":     max(1, int((time[-1] - time[0]) / period)),
#             "snr":            round(snr, 2),
#             "rp_rs":          round(np.sqrt(max(depth, 0)), 4),
#         },
#         "classification": {
#             "top_class":     top_class,
#             "confidence":    round(probs[top_class], 1),
#             "probabilities": probs,
#         },
#         "data_quality": {
#             "completeness_pct": round(float(completeness), 1),
#             "cdpp_ppm":         round(float(cdpp), 1),
#             "sys_noise_ppm":    round(float(cdpp) * 0.15, 1),
#         },
#     }


# # ── Routes — existing ─────────────────────────────────────────────────────

# @app.route("/api/targets")
# def get_targets():
#     return jsonify({"targets": [
#         {"name": k, **v} for k, v in KNOWN_TARGETS.items()
#     ]})

# @app.route("/api/analyse")
# def analyse():
#     target = request.args.get("target", "L 98-59")
#     try:
#         return jsonify({"status": "ok", "data": fetch_and_analyse(target)})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route("/api/health")
# def health():
#     return jsonify({
#         "status":     "ok",
#         "device":     str(DEVICE),
#         "classifier": "CNN" if MODEL else "heuristic",
#     })

# @app.route("/api/model-info")
# def model_info():
#     return jsonify({
#         "model_loaded": MODEL is not None,
#         "device":       str(DEVICE),
#         "metrics":      MODEL_META,
#         "phase":        2,
#         "mcmc_backend": "emcee 3.1 + batman 2.4",
#     })


# # ── Routes — Phase 2 MCMC ─────────────────────────────────────────────────

# @app.route("/api/mcmc/start", methods=["POST"])
# def mcmc_start():
#     """
#     Kick off an async MCMC job.

#     Body (JSON):
#       { "target": "L 98-59", "n_walkers": 32, "n_steps": 2000, "n_burn": 500 }

#     Returns:
#       { "job_id": "...", "status": "queued" }
#     """
#     body      = request.get_json(force=True, silent=True) or {}
#     target    = body.get("target", "L 98-59")
#     n_walkers = int(body.get("n_walkers", 32))
#     n_steps   = int(body.get("n_steps",  2000))
#     n_burn    = int(body.get("n_burn",   500))

#     task = run_mcmc_task.apply_async(
#         args=[target, n_walkers, n_steps, n_burn]
#     )
#     return jsonify({"job_id": task.id, "status": "queued",
#                     "target": target})


# @app.route("/api/mcmc/status")
# def mcmc_status():
#     """
#     Poll job progress.
#     GET /api/mcmc/status?job_id=<id>

#     Returns:
#       { "state": "PROGRESS"|"SUCCESS"|"FAILURE",
#         "step": "…", "pct": 42 }
#     """
#     job_id = request.args.get("job_id")
#     if not job_id:
#         return jsonify({"error": "job_id required"}), 400

#     task = celery_app.AsyncResult(job_id)

#     if task.state == "PENDING":
#         return jsonify({"state": "PENDING", "step": "Queued…", "pct": 0})

#     if task.state == "PROGRESS":
#         meta = task.info or {}
#         return jsonify({"state": "PROGRESS",
#                         "step": meta.get("step", "Running…"),
#                         "pct":  meta.get("pct",  0)})

#     if task.state == "SUCCESS":
#         result = task.result
#         if result.get("status") == "error":
#             return jsonify({"state": "FAILURE",
#                             "message": result.get("message", "Unknown error")})
#         return jsonify({"state": "SUCCESS", "pct": 100,
#                         "step": "Done",
#                         "elapsed_sec": result.get("elapsed_sec"),
#                         "acceptance":  result.get("acceptance_frac")})

#     # FAILURE
#     return jsonify({"state": "FAILURE",
#                     "message": str(task.info)}), 500


# @app.route("/api/mcmc/result")
# def mcmc_result():
#     """
#     Fetch full MCMC result (only after state == SUCCESS).
#     GET /api/mcmc/result?job_id=<id>

#     Returns the full result dict including:
#       - percentiles (median, err_low, err_high per parameter)
#       - model_flux  (best-fit transit model over original time array)
#       - plots.corner    (base64 PNG)
#       - plots.posterior (base64 PNG)
#       - light_curve (time + flux arrays)
#       - bls_init
#     """
#     job_id = request.args.get("job_id")
#     if not job_id:
#         return jsonify({"error": "job_id required"}), 400

#     task = celery_app.AsyncResult(job_id)
#     if task.state != "SUCCESS":
#         return jsonify({"error": f"Job not done yet (state={task.state})"}), 400

#     result = task.result
#     if result.get("status") == "error":
#         return jsonify({"status": "error",
#                         "message": result.get("message")}), 500

#     return jsonify({"status": "ok", "data": result})


# if __name__ == "__main__":
#     load_model()
#     print("=" * 60)
#     print("  ExoDetect Backend v3 — Phase 2 (CNN + MCMC)")
#     print(f"  Device : {DEVICE}")
#     print("  http://localhost:8000")
#     print("=" * 60)
#     app.run(debug=True, port=8000)


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

app   = Flask(__name__)
CORS(app)
REDIS = redis.Redis(host="localhost", port=6379, db=1)
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
        self.net = nn.Sequential(
            nn.Conv1d(ic,oc,k,padding=k//2), nn.BatchNorm1d(oc),
            nn.ReLU(True), nn.MaxPool1d(p))
    def forward(self, x): return self.net(x)


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
        print(f"[INFO] CNN loaded — Val AUC {CNN_META.get('best_val_auc','?')}")
    else:
        print(f"[WARN] No CNN model at {CNN_PATH}")

    if os.path.exists(TRANSFORMER_PATH):
        TF_MODEL, TF_META = load_transformer(TRANSFORMER_PATH, device=DEVICE)
        print(f"[INFO] TransitFormer loaded — Val AUC {TF_META.get('best_val_auc','?')}")
    else:
        print(f"[WARN] No TransitFormer at {TRANSFORMER_PATH}")
        print("       Run: python train_transformer.py")

    if CNN_MODEL and TF_MODEL:
        ENSEMBLE = ExoEnsemble(CNN_MODEL, TF_MODEL, device=DEVICE)
        print("[INFO] Ensemble ready (CNN + TransitFormer)")
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

def _tensors(gv, lv, period, dur_hr, depth_ppm, snr):
    gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0).to(DEVICE)
    lv_t = torch.tensor(lv).unsqueeze(0).unsqueeze(0).to(DEVICE) if lv is not None else None
    sf_t = torch.tensor([[
        float(np.log1p(period)),
        float(np.log1p(max(dur_hr, 0))),
        float(np.log1p(max(depth_ppm, 0))) / 10,
        float(np.clip(snr, 0, 100)) / 100,
    ]]).to(DEVICE)
    return gv_t, lv_t, sf_t

def _heuristic(period, depth_pct, dur_hr, snr):
    if depth_pct > 5: return {"Exoplanet Transit": 8.0, "Eclipsing Binary": 81.0, "Stellar Blend": 7.0, "Starspot": 4.0}
    if snr < 7:       return {"Exoplanet Transit": 30.0,"Eclipsing Binary": 20.0, "Stellar Blend": 25.0,"Starspot": 25.0}
    return             {"Exoplanet Transit": 91.0,"Eclipsing Binary": 5.0,  "Stellar Blend": 3.0, "Starspot": 1.0}


def classify_with_ensemble(gv, lv, period, dur_hr, depth_ppm, snr):
    """Returns probs dict, p_cnn, p_tf, attn_weights."""
    if ENSEMBLE and gv is not None and lv is not None:
        gv_t, lv_t, sf_t = _tensors(gv, lv, period, dur_hr, depth_ppm, snr)
        p, p_cnn, p_tf, attn = ENSEMBLE.predict(gv_t, lv_t, sf_t)
        probs = ENSEMBLE.classify(p, depth_ppm, period, dur_hr)
        return probs, p_cnn, p_tf, attn.tolist() if attn is not None else None
    elif CNN_MODEL and gv is not None and lv is not None:
        gv_t, lv_t, sf_t = _tensors(gv, lv, period, dur_hr, depth_ppm, snr)
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

    probs, p_cnn, p_tf, attn = classify_with_ensemble(gv, lv, period, dur_hr, depth_ppm, snr)
    top_class = max(probs, key=probs.get)
    cdpp      = float(lc.estimate_cdpp().value) if hasattr(lc,"estimate_cdpp") else 200.0
    comp      = float(100*(1-np.isnan(lc.flux.value).sum()/len(lc.flux.value)))

    return {
        "target":       target_name,
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
        "cnn":{"loaded":CNN_MODEL is not None,"metrics":CNN_META},
        "transformer":{"loaded":TF_MODEL is not None,"metrics":TF_META},
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
    body   = request.get_json(force=True,silent=True) or {}
    sector = int(body.get("sector",14))
    max_t  = int(body.get("max_targets",10))
    task   = run_sector.apply_async(args=[sector, max_t])
    return jsonify({"job_id":task.id,"sector":sector,"status":"queued"})

@app.route("/api/batch/status")
def batch_status():
    job_id = request.args.get("job_id")
    task   = batch_celery.AsyncResult(job_id)
    if task.state == "PENDING":  return jsonify({"state":"PENDING","pct":0,"step":"Queued…"})
    if task.state == "PROGRESS":
        m = task.info or {}
        return jsonify({"state":"PROGRESS","pct":m.get("pct",0),
                        "step":m.get("step",""),"candidates":m.get("candidates_found",0)})
    if task.state == "SUCCESS":
        r = task.result
        return jsonify({"state":"SUCCESS","pct":100,
                        "candidates":r.get("candidates",0),
                        "total":r.get("total_processed",0)})
    return jsonify({"state":"FAILURE","message":str(task.info)}),500

@app.route("/api/batch/results")
def batch_results():
    job_id = request.args.get("job_id")
    task   = batch_celery.AsyncResult(job_id)
    if task.state != "SUCCESS": return jsonify({"error":"Not done"}),400
    return jsonify({"status":"ok","data":task.result})

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