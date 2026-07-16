"""
batch_pipeline.py — Phase 3 FIXED

Root causes fixed:
  1. process_target.apply() inside run_sector caused a deadlock
     (worker calling itself synchronously).
     Fix: inline the processing logic directly in run_sector.

  2. Results were stored under wrong Redis key format;
     server_phase3.py was reading task.result directly
     but the key lookup failed silently.
     Fix: return results directly from the Celery task,
     server reads task.result (standard Celery pattern).

  3. TransitFormer not loaded → silent fallback to None
     caused p_transformer to be None in results.
     Fix: explicit CNN-only path with clear logging.

Phase 4 update:
  _classify() now builds each submodel's stellar feature tensor via the
  shared stellar_features module, sized to that submodel's own
  n_stellar — see stellar_features.py and the matching change in
  server.py for the reasoning (avoids the CNN/TF stellar-feature formula
  silently drifting between files, and lets CNN/TF be on different
  schema versions during a gradual rollout). _process_one() now also
  does a live TIC catalog lookup per target so real Teff/radius/log g/
  Tmag values feed the expanded schema when a model expects it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
import torch
import torch.nn as nn
import warnings, traceback, os, time
from pathlib import Path
warnings.filterwarnings("ignore")

# Phase 4 import — shared stellar feature engineering (see module docstring)
from stellar_features import build_stellar_features, extract_tic_id, fetch_stellar_params

# ── Celery app ─────────────────────────────────────────────────────────────
celery_app = Celery(
    "exodetect_batch",
    broker="redis://localhost:6379/0",    # DB 0 — matches tasks.py
    backend="redis://localhost:6379/0",   # DB 0 — matches tasks.py
)
celery_app.conf.update(
    task_default_queue="batch",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=7200,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_acks_late=True,
    worker_max_tasks_per_child=20,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR      = Path(__file__).parent
CNN_PATH         = BACKEND_DIR / "exodetect_cnn.pt"
TRANSFORMER_PATH = BACKEND_DIR / "exodetect_transformer.pt"
GLOBAL_LEN       = 201
LOCAL_LEN        = 81
MCMC_SNR_THRESHOLD    = 12.0
PLANET_PROB_THRESHOLD = 0.65


# ── Model loader — cached per worker process ───────────────────────────────
_worker_models = {}

def _get_worker_models():
    global _worker_models
    if _worker_models:
        return _worker_models

    device = torch.device("mps") if torch.backends.mps.is_available() \
             else torch.device("cpu")

    # ── CNN ────────────────────────────────────────────────────────────────
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
            self.n_stellar = n_stellar   # Phase 4: so _classify() can size this
                                          # model's stellar tensor correctly.
            self.gb = nn.Sequential(
                ConvBlock(1,16,5,2), ConvBlock(16,32,5,2),
                ConvBlock(32,64,5,2), ConvBlock(64,128,3,2),
                SafeAdaptiveAvgPool1d(8), nn.Flatten())
            self.lb = nn.Sequential(
                ConvBlock(1,16,5,2), ConvBlock(16,32,5,2),
                ConvBlock(32,64,3,2), SafeAdaptiveAvgPool1d(4), nn.Flatten())
            fused = (128*8) + (64*4) + n_stellar
            self.head = nn.Sequential(
                nn.Linear(fused,512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512,256),   nn.ReLU(True), nn.Dropout(0.3),
                nn.Linear(256,2))
        def forward(self, gv, lv, sf):
            return self.head(torch.cat([self.gb(gv), self.lb(lv), sf], 1))

    cnn = None
    if CNN_PATH.exists():
        try:
            ckpt = torch.load(CNN_PATH, map_location="cpu")
            cnn  = ExoDetectCNN(**ckpt["model_config"]).to(device)
            cnn.load_state_dict(ckpt["model_state_dict"])
            cnn.eval()
            print(f"[Worker] CNN loaded on {device} (n_stellar={cnn.n_stellar})")
        except Exception as e:
            print(f"[Worker] CNN load failed: {e}")
    else:
        print(f"[Worker] No CNN at {CNN_PATH} — heuristic mode")

    # ── TransitFormer (optional) ───────────────────────────────────────────
    tf_model = None
    if TRANSFORMER_PATH.exists():
        try:
            from transit_transformer import load_transformer
            tf_model, _ = load_transformer(TRANSFORMER_PATH, device=device)
            print(f"[Worker] TransitFormer loaded on {device} (n_stellar={tf_model.n_stellar})")
        except Exception as e:
            print(f"[Worker] TransitFormer load failed: {e}")
    else:
        print(f"[Worker] No TransitFormer at {TRANSFORMER_PATH} — CNN only")

    _worker_models = {
        "cnn":         cnn,
        "transformer": tf_model,
        "device":      device,
    }
    return _worker_models


# ── Preprocessing ──────────────────────────────────────────────────────────

def _smooth(flux, w=51):
    from scipy.ndimage import median_filter
    return flux / median_filter(flux.astype(float), size=w, mode="reflect")

def _fold_bin(time, flux, period, t0, n_bins, local=False, lw=0.2):
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)
    if local:
        m = np.abs(phase) < lw
        phase, flux = phase[m], flux[m]
        if len(phase) < 5:
            return None
        p0, p1 = -lw, lw
    else:
        p0, p1 = -0.5, 0.5
    bins = np.linspace(p0, p1, n_bins + 1)
    idx  = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    view = np.array([
        np.median(flux[idx == b]) if np.any(idx == b) else 1.0
        for b in range(n_bins)
    ])
    view = np.where(np.isfinite(view), view, 1.0)
    oot  = np.abs(np.linspace(p0, p1, n_bins)) > 0.05
    if oot.sum() > 3:
        mu, sig = np.median(view[oot]), np.std(view[oot]) + 1e-8
        view = (view - mu) / sig
    return view.astype(np.float32)


def _classify(gv, lv, period, dur_hr, depth_ppm, snr, models, stellar_params=None):
    """
    Run CNN + optional TransitFormer, return p_planet, p_cnn, p_tf, probs, attn.
    Falls back gracefully at every stage. Each submodel's stellar feature
    vector is sized to that model's own n_stellar (see stellar_features.py),
    so CNN and TransitFormer checkpoints can be on different schema
    versions during a gradual rollout.
    """
    stellar_params = stellar_params or {}
    device = models["device"]
    cnn    = models["cnn"]
    tf     = models["transformer"]

    p_cnn, p_tf, attn = None, None, None

    # CNN
    if cnn is not None and gv is not None and lv is not None:
        try:
            gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0).to(device)
            lv_t = torch.tensor(lv).unsqueeze(0).unsqueeze(0).to(device)
            sf = build_stellar_features(
                cnn.n_stellar, period, dur_hr, depth_ppm, snr,
                stellar_params.get("teff"), stellar_params.get("rad"),
                stellar_params.get("logg"), stellar_params.get("tmag"),
            )
            sf_t = torch.tensor([sf]).to(device)
            with torch.no_grad():
                logits = cnn(gv_t, lv_t, sf_t)
                p_cnn  = float(torch.softmax(logits, dim=1)[0, 1].item())
        except Exception as e:
            print(f"[Classify] CNN error: {e}")

    # TransitFormer
    if tf is not None and gv is not None:
        try:
            gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0).to(device)
            sf = build_stellar_features(
                tf.n_stellar, period, dur_hr, depth_ppm, snr,
                stellar_params.get("teff"), stellar_params.get("rad"),
                stellar_params.get("logg"), stellar_params.get("tmag"),
            )
            sf_t = torch.tensor([sf]).to(device)
            p_tf_arr, attn_arr = tf.predict_proba(gv_t, sf_t)
            p_tf  = float(p_tf_arr[0])
            attn  = attn_arr[0].tolist() if attn_arr is not None else None
        except Exception as e:
            print(f"[Classify] TF error: {e}")

    # Ensemble or fallback
    if p_cnn is not None and p_tf is not None:
        p_planet = 0.45 * p_cnn + 0.55 * p_tf
    elif p_cnn is not None:
        p_planet = p_cnn
    elif p_tf is not None:
        p_planet = p_tf
    else:
        # Pure heuristic
        p_planet = 0.80 if snr > 10 and depth_ppm < 50000 else 0.20

    p_fp = 1.0 - p_planet
    if depth_ppm > 50000:
        eb, bl, sp = 0.65, 0.25, 0.10
    elif dur_hr / (period * 24 + 1e-6) > 0.15:
        eb, bl, sp = 0.20, 0.65, 0.15
    else:
        eb, bl, sp = 0.35, 0.40, 0.25

    probs = {
        "Exoplanet Transit": round(p_planet * 100, 1),
        "Eclipsing Binary":  round(p_fp * eb * 100, 1),
        "Stellar Blend":     round(p_fp * bl * 100, 1),
        "Starspot":          round(p_fp * sp * 100, 1),
    }
    return p_planet, p_cnn, p_tf, probs, attn


def _process_one(target_str, sector, models):
    """
    Download + BLS + classify one target.
    Returns a result dict. Never raises — returns error dict on failure.
    """
    try:
        search_kw = {"mission": "TESS", "author": "SPOC"}
        if sector:
            search_kw["sector"] = sector

        search = lk.search_lightcurve(target_str, **search_kw)
        if len(search) == 0:
            search = lk.search_lightcurve(target_str, mission="TESS")
        if len(search) == 0:
            return {"tic_id": target_str, "status": "no_data"}

        # Phase 4: resolve TIC ID + live stellar params once per target, so
        # both submodels can use real Teff/radius/log g/Tmag when a model
        # expects the expanded schema. Never blocks the scan if it fails.
        tic_id_num = extract_tic_id(search[0])
        stellar_params = fetch_stellar_params(tic_id_num) if tic_id_num is not None else {}

        lc = search[0].download(flux_column="pdcsap_flux")
        lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

        time_arr = np.array(lc.time.value)
        flux_arr = np.array(lc.flux.value)
        ferr_arr = np.array(lc.flux_err.value) \
                   if lc.flux_err is not None else np.full_like(flux_arr, 5e-4)

        if len(time_arr) < 100:
            return {"tic_id": target_str, "status": "too_short"}

        # BLS
        bls     = BoxLeastSquares(time_arr, flux_arr, ferr_arr)
        periods = np.linspace(0.5, 27.0, 3000)
        power   = bls.power(periods, np.linspace(0.05, 0.5, 15))
        bi      = np.argmax(power.power)
        period  = float(power.period[bi])
        dur     = float(power.duration[bi])
        t0      = float(power.transit_time[bi])
        depth   = float(power.depth[bi])

        in_tr  = np.abs(((time_arr - t0) % period) - period / 2) < dur / 2
        noise  = np.std(flux_arr[~in_tr]) if (~in_tr).sum() > 10 else 1e-4
        sig    = abs(np.mean(flux_arr[in_tr]) - np.mean(flux_arr[~in_tr])) \
                 if in_tr.sum() > 0 else 0.0
        snr      = float(sig / noise * np.sqrt(max(in_tr.sum(), 1)))
        dur_hr   = dur * 24
        depth_ppm = depth * 1e6

        # Views
        smoothed = _smooth(flux_arr)
        gv = _fold_bin(time_arr, smoothed, period, t0, GLOBAL_LEN)
        lv = _fold_bin(time_arr, smoothed, period, t0, LOCAL_LEN, local=True)

        # Classify
        p_planet, p_cnn, p_tf, probs, attn = _classify(
            gv, lv, period, dur_hr, depth_ppm, snr, models, stellar_params
        )
        top_class = max(probs, key=probs.get)

        return {
            "tic_id":           target_str,
            "tic_id_num":       tic_id_num,
            "stellar_params":   stellar_params,
            "status":           "ok",
            "period_days":      round(period, 4),
            "depth_pct":        round(depth * 100, 4),
            "duration_hr":      round(dur_hr, 3),
            "snr":              round(snr, 2),
            "p_planet":         round(float(p_planet), 4),
            "p_cnn":            round(float(p_cnn), 4) if p_cnn is not None else None,
            "p_transformer":    round(float(p_tf),  4) if p_tf  is not None else None,
            "top_class":        top_class,
            "confidence":       round(probs[top_class], 1),
            "probabilities":    probs,
            "attention_weights":attn,
            "flag_mcmc":        snr > MCMC_SNR_THRESHOLD and p_planet > PLANET_PROB_THRESHOLD,
            "n_cadences":       len(time_arr),
        }

    except Exception as e:
        traceback.print_exc()
        return {"tic_id": target_str, "status": "error", "message": str(e)}


# ── Demo target list ────────────────────────────────────────────────────────
# In full production, query MAST TIC catalog for the sector.
# For local testing these are confirmed targets with TESS data.
DEMO_TARGETS = [
    ("L 98-59",       "3 confirmed planets"),
    ("TOI-700",       "Habitable-zone Earth"),
    ("WASP-18",       "Hot Jupiter 0.94d"),
    ("HD 21749",      "Multi-planet system"),
    ("TIC 286923464", "HD 118203 b eccentric"),
    ("TIC 260647166", "TOI-125 multi-planet"),
    ("TIC 55652896",  "TOI-402"),
    ("TIC 144065872", "TOI-134"),
    ("TIC 279741379", "HD 21749 b"),
    ("TIC 150428135", "TOI-700 d"),
]


# ── Main batch task ────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="batch.run_sector")
def run_sector(self, sector_number, max_targets=10):
    """
    Process multiple TESS targets inline (no sub-task calls).
    Returns full results dict directly — readable via task.result.
    """
    try:
        # Load models once for this worker
        models = _get_worker_models()

        targets = DEMO_TARGETS[:min(max_targets, len(DEMO_TARGETS))]
        total   = len(targets)

        self.update_state(state="PROGRESS",
                          meta={"step": f"Starting — {total} targets queued",
                                "pct": 2, "candidates_found": 0})

        results    = []
        candidates = []
        errors     = 0

        for i, (target_str, note) in enumerate(targets):
            print(f"[Sector {sector_number}] {i+1}/{total} → {target_str}")

            self.update_state(state="PROGRESS",
                              meta={
                                  "step": f"Processing {target_str} ({i+1}/{total})",
                                  "pct":  2 + int(90 * i / total),
                                  "candidates_found": len(candidates),
                              })

            res = _process_one(target_str, sector_number, models)
            res["note"] = note
            results.append(res)

            if res.get("status") == "ok" and res.get("flag_mcmc"):
                candidates.append(res)
            if res.get("status") == "error":
                errors += 1

            time.sleep(0.3)   # polite to MAST

        # Sort by P(planet) descending
        ok_results = [r for r in results if r.get("status") == "ok"]
        ok_results.sort(key=lambda x: x.get("p_planet", 0), reverse=True)

        self.update_state(state="PROGRESS",
                          meta={"step": "Finalising results…", "pct": 97,
                                "candidates_found": len(candidates)})

        summary = {
            "sector":           sector_number,
            "total_processed":  total,
            "successful":       len(ok_results),
            "candidates":       len(candidates),
            "errors":           errors,
            "top_candidates":   ok_results,       # ALL ok results, sorted
            "status":           "done",
        }

        print(f"[Sector {sector_number}] Complete — "
              f"{len(ok_results)} OK, {len(candidates)} candidates, {errors} errors")
        return summary

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e),
                "top_candidates": [], "candidates": 0, "total_processed": 0}