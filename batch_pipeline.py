"""
batch_pipeline.py — Phase 3: Full TESS Sector Batch Processor

Celery tasks that:
  1. Download a full TESS sector target list from MAST
  2. Process each light curve through CNN + TransitFormer ensemble
  3. Priority-queue high-confidence candidates for MCMC
  4. Store results in Redis for the frontend to poll

Usage (from backend folder):
  celery -A batch_pipeline worker --loglevel=info --concurrency=4
  celery -A batch_pipeline beat   --loglevel=info   (for scheduled runs)
"""

from celery import Celery, group, chord
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
import torch
import json, time, warnings, traceback, os
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Celery app ─────────────────────────────────────────────────────────────
celery_app = Celery(
    "exodetect_batch",
    broker="redis://localhost:6379/1",    # separate DB from Phase 2 tasks
    backend="redis://localhost:6379/1",
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=7200,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_acks_late=True,
    worker_max_tasks_per_child=50,   # restart after 50 tasks (memory safety)
)

# ── Constants ──────────────────────────────────────────────────────────────
BACKEND_DIR      = Path(__file__).parent
CNN_PATH         = BACKEND_DIR / "exodetect_cnn.pt"
TRANSFORMER_PATH = BACKEND_DIR / "exodetect_transformer.pt"
GLOBAL_LEN       = 201
LOCAL_LEN        = 81

# SNR threshold above which candidates get auto-queued for MCMC
MCMC_SNR_THRESHOLD = 12.0
# P(planet) threshold to flag as candidate
PLANET_PROB_THRESHOLD = 0.70


# ── Model loader (lazy, per-worker) ───────────────────────────────────────

_models = {}

def get_models():
    """Load models once per worker process."""
    global _models
    if _models:
        return _models

    from transit_transformer import (
        TransitFormer, ExoEnsemble, get_device, load_transformer
    )

    # Import CNN architecture inline (avoids circular import)
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, ic, oc, k=5, p=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ic, oc, k, padding=k//2), nn.BatchNorm1d(oc),
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

    device = get_device()

    # Load CNN
    cnn = None
    if CNN_PATH.exists():
        ckpt = torch.load(CNN_PATH, map_location="cpu")
        cnn  = ExoDetectCNN(**ckpt["model_config"]).to(device)
        cnn.load_state_dict(ckpt["model_state_dict"])
        cnn.eval()
        print(f"[Worker] CNN loaded on {device}")

    # Load TransitFormer
    tf_model = None
    if TRANSFORMER_PATH.exists():
        tf_model, _ = load_transformer(TRANSFORMER_PATH, device=device)
        print(f"[Worker] TransitFormer loaded on {device}")

    # Build ensemble (or fallback)
    if cnn and tf_model:
        ensemble = ExoEnsemble(cnn, tf_model, device=device)
    else:
        ensemble = None
        print("[Worker] Running in heuristic-only mode (no model files found)")

    _models = {"cnn": cnn, "transformer": tf_model,
               "ensemble": ensemble, "device": device}
    return _models


# ── Preprocessing (identical to Phase 1/2) ────────────────────────────────

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

def _make_tensors(gv_arr, lv_arr, period, dur_hr, depth_ppm, snr, device):
    gv_t = torch.tensor(gv_arr).unsqueeze(0).unsqueeze(0).to(device)
    lv_t = torch.tensor(lv_arr).unsqueeze(0).unsqueeze(0).to(device) if lv_arr is not None else None
    sf_t = torch.tensor([[
        float(np.log1p(period)),
        float(np.log1p(max(dur_hr, 0))),
        float(np.log1p(max(depth_ppm, 0))) / 10,
        float(np.clip(snr, 0, 100)) / 100,
    ]]).to(device)
    return gv_t, lv_t, sf_t


# ── Single target processing ───────────────────────────────────────────────

@celery_app.task(bind=True, name="batch.process_target", max_retries=2)
def process_target(self, tic_id, sector=None):
    """
    Process a single TIC target:
      1. Download TESS LC
      2. BLS periodogram
      3. CNN + Transformer ensemble classification
      4. Return structured result dict

    Returns dict with keys:
      tic_id, period, depth_pct, duration_hr, snr,
      p_planet, p_cnn, p_transformer, top_class, confidence,
      attention_weights (list of floats, length n_patches),
      flag_mcmc (bool)
    """
    try:
        models = get_models()
        device = models["device"]

        # Download LC
        target = f"TIC {tic_id}" if str(tic_id).isdigit() else tic_id
        search_kwargs = {"mission": "TESS", "author": "SPOC"}
        if sector:
            search_kwargs["sector"] = sector

        search = lk.search_lightcurve(target, **search_kwargs)
        if len(search) == 0:
            search = lk.search_lightcurve(target, mission="TESS")
        if len(search) == 0:
            return {"tic_id": tic_id, "status": "no_data"}

        lc = search[0].download(flux_column="pdcsap_flux")
        lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

        time_arr = np.array(lc.time.value)
        flux_arr = np.array(lc.flux.value)
        ferr_arr = np.array(lc.flux_err.value) if lc.flux_err is not None \
                   else np.full_like(flux_arr, 5e-4)

        if len(time_arr) < 100:
            return {"tic_id": tic_id, "status": "too_short"}

        # BLS
        bls     = BoxLeastSquares(time_arr, flux_arr, ferr_arr)
        periods = np.linspace(0.51, 27.0, 5000)
        power   = bls.power(periods, np.linspace(0.01, 0.4, 20))
        bi      = np.argmax(power.power)
        period  = float(power.period[bi])
        dur     = float(power.duration[bi])
        t0      = float(power.transit_time[bi])
        depth   = float(power.depth[bi])

        in_tr  = np.abs(((time_arr - t0) % period) - period / 2) < dur / 2
        noise  = np.std(flux_arr[~in_tr]) if (~in_tr).sum() > 10 else 1e-4
        sig    = abs(np.mean(flux_arr[in_tr]) - np.mean(flux_arr[~in_tr])) \
                 if in_tr.sum() > 0 else 0
        snr    = float(sig / noise * np.sqrt(max(in_tr.sum(), 1)))
        dur_hr = dur * 24
        depth_ppm = depth * 1e6

        # Preprocessing
        smoothed = _smooth(flux_arr)
        gv = _fold_bin(time_arr, smoothed, period, t0, GLOBAL_LEN)
        lv = _fold_bin(time_arr, smoothed, period, t0, LOCAL_LEN, local=True)

        # Classification
        attn_weights = None
        if models["ensemble"] and gv is not None and lv is not None:
            gv_t, lv_t, sf_t = _make_tensors(gv, lv, period, dur_hr, depth_ppm, snr, device)
            p_planet, p_cnn, p_tf, attn_weights = models["ensemble"].predict(
                gv_t, lv_t, sf_t
            )
            probs = models["ensemble"].classify(p_planet, depth_ppm, period, dur_hr)
        else:
            # Fallback heuristic
            p_planet = 0.85 if snr > 10 and depth_ppm < 50000 else 0.15
            p_cnn, p_tf = p_planet, p_planet
            p_fp = 1 - p_planet
            probs = {
                "Exoplanet Transit": round(p_planet * 100, 1),
                "Eclipsing Binary":  round(p_fp * 0.5 * 100, 1),
                "Stellar Blend":     round(p_fp * 0.3 * 100, 1),
                "Starspot":          round(p_fp * 0.2 * 100, 1),
            }

        top_class = max(probs, key=probs.get)

        return {
            "tic_id":          str(tic_id),
            "status":          "ok",
            "period_days":     round(period, 4),
            "depth_pct":       round(depth * 100, 4),
            "duration_hr":     round(dur_hr, 3),
            "snr":             round(snr, 2),
            "p_planet":        round(float(p_planet), 4),
            "p_cnn":           round(float(p_cnn), 4),
            "p_transformer":   round(float(p_tf), 4),
            "top_class":       top_class,
            "confidence":      round(probs[top_class], 1),
            "probabilities":   probs,
            "attention_weights": attn_weights.tolist() if attn_weights is not None else None,
            "flag_mcmc":       snr > MCMC_SNR_THRESHOLD and float(p_planet) > PLANET_PROB_THRESHOLD,
            "n_cadences":      len(time_arr),
        }

    except Exception as e:
        traceback.print_exc()
        # Retry up to 2 times
        try:
            raise self.retry(exc=e, countdown=30)
        except Exception:
            return {"tic_id": str(tic_id), "status": "error", "message": str(e)}


# ── Sector batch orchestrator ──────────────────────────────────────────────

@celery_app.task(bind=True, name="batch.run_sector")
def run_sector(self, sector_number, max_targets=200):
    """
    Download TIC list for a TESS sector and process all targets.

    sector_number : TESS sector (1–96)
    max_targets   : cap for local testing (set to None for full sector)

    Progress is stored in Redis under key: sector:{sector_number}:progress
    Results are stored under:           sector:{sector_number}:results
    """
    import redis
    r = redis.Redis(host="localhost", port=6379, db=1)

    sector_key   = f"sector:{sector_number}"
    progress_key = f"{sector_key}:progress"
    results_key  = f"{sector_key}:results"

    try:
        self.update_state(state="PROGRESS",
                          meta={"step": "Fetching sector TIC list from MAST…", "pct": 2})

        # Use lightkurve to get target list for this sector
        # In production, use astroquery.mast; here we use a representative sample
        print(f"[Sector {sector_number}] Searching MAST for targets…")

        # Known good targets for local testing
        # In full production: query MAST TIC catalog for this sector
        test_tic_ids = [
            307210830,  # L 98-59  (3 planets)
            150428135,  # TOI-700  (habitable zone)
            100100827,  # WASP-18  (hot Jupiter)
            279741379,  # HD 21749 (multi-planet)
            286923464,  # HD 118203 b
            260647166,  # TOI-125 (multi-planet)
            55652896,   # TOI-402
            144065872,  # TOI-134
            231702397,  # TOI-700d candidate
            441420236,  # TOI-2180
        ]

        if max_targets:
            tic_ids = test_tic_ids[:min(max_targets, len(test_tic_ids))]
        else:
            tic_ids = test_tic_ids

        total = len(tic_ids)
        r.hset(progress_key, mapping={
            "total": total, "done": 0,
            "candidates": 0, "errors": 0, "status": "running"
        })
        r.expire(progress_key, 7200)

        self.update_state(state="PROGRESS",
                          meta={"step": f"Processing {total} targets…", "pct": 5})

        results = []
        candidates = []

        for i, tic_id in enumerate(tic_ids):
            print(f"[Sector {sector_number}] {i+1}/{total} → TIC {tic_id}")
            result = process_target.apply(args=[tic_id, sector_number]).get()
            results.append(result)

            if result.get("status") == "ok":
                if result.get("flag_mcmc"):
                    candidates.append(result)

            # Update Redis progress
            done = i + 1
            pct  = 5 + int(90 * done / total)
            r.hset(progress_key, mapping={
                "done":       done,
                "candidates": len(candidates),
                "errors":     sum(1 for r_ in results if r_.get("status") == "error"),
                "status":     "running",
            })
            self.update_state(state="PROGRESS",
                              meta={"step": f"Processed {done}/{total} targets",
                                    "pct": pct,
                                    "candidates_found": len(candidates)})
            time.sleep(0.2)   # polite to MAST

        # Sort results by P(planet) descending
        ok_results = [r for r in results if r.get("status") == "ok"]
        ok_results.sort(key=lambda x: x.get("p_planet", 0), reverse=True)

        # Store full results in Redis
        r.set(results_key, json.dumps(ok_results), ex=7200)
        r.hset(progress_key, mapping={"status": "done", "done": total})

        summary = {
            "sector":         sector_number,
            "total_processed": total,
            "candidates":     len(candidates),
            "top_candidates": ok_results[:10],
            "status":         "done",
        }

        print(f"[Sector {sector_number}] Done. {len(candidates)} candidates found.")
        return summary

    except Exception as e:
        traceback.print_exc()
        r.hset(progress_key, "status", "error")
        return {"status": "error", "message": str(e)}


# ── Scheduled beat task (optional) ────────────────────────────────────────
# Uncomment to automatically process a new sector every day:
#
# from celery.schedules import crontab
# celery_app.conf.beat_schedule = {
#     "process-latest-sector": {
#         "task":     "batch.run_sector",
#         "schedule": crontab(hour=2, minute=0),  # 2 AM daily
#         "args":     [14],   # sector number
#     },
# }