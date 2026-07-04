# """
# ExoDetect Backend v2 — Real CNN Classifier
# Loads the trained ExoDetectCNN from exodetect_cnn.pt
# and replaces the heuristic classifier in server.py.

# Drop this file into your exodetect-backend/ folder,
# replacing server.py. Requires exodetect_cnn.pt in the same directory.
# """

# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import numpy as np
# import lightkurve as lk
# from astropy.timeseries import BoxLeastSquares
# from scipy.ndimage import median_filter
# import torch
# import torch.nn as nn
# import traceback, warnings, os
# warnings.filterwarnings("ignore")

# app = Flask(__name__)
# CORS(app)

# # ── Reproduce the exact same architecture from the notebook ────────────────

# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel=5, pool=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
#             nn.BatchNorm1d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(pool),
#         )
#     def forward(self, x):
#         return self.net(x)


# class ExoDetectCNN(nn.Module):
#     def __init__(self, global_len=201, local_len=81, n_stellar=4):
#         super().__init__()
#         self.global_branch = nn.Sequential(
#             ConvBlock(1,  16, kernel=5, pool=2),
#             ConvBlock(16, 32, kernel=5, pool=2),
#             ConvBlock(32, 64, kernel=5, pool=2),
#             ConvBlock(64, 128, kernel=3, pool=2),
#             nn.AdaptiveAvgPool1d(8),
#             nn.Flatten(),
#         )
#         self.local_branch = nn.Sequential(
#             ConvBlock(1,  16, kernel=5, pool=2),
#             ConvBlock(16, 32, kernel=5, pool=2),
#             ConvBlock(32, 64, kernel=3, pool=2),
#             nn.AdaptiveAvgPool1d(4),
#             nn.Flatten(),
#         )
#         fused = (128 * 8) + (64 * 4) + n_stellar
#         self.head = nn.Sequential(
#             nn.Linear(fused, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
#             nn.Linear(512, 256),   nn.ReLU(inplace=True), nn.Dropout(0.3),
#             nn.Linear(256, 2),
#         )

#     def forward(self, gv, lv, sf):
#         g = self.global_branch(gv)
#         l = self.local_branch(lv)
#         return self.head(torch.cat([g, l, sf], dim=1))

#     def predict_proba(self, gv, lv, sf):
#         with torch.no_grad():
#             logits = self.forward(gv, lv, sf)
#             return torch.softmax(logits, dim=1)[:, 1].item()


# # ── Load model ─────────────────────────────────────────────────────────────

# DEVICE     = torch.device("cpu")   # CPU is fine for inference
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "exodetect_cnn.pt")
# MODEL      = None
# MODEL_META = {}

# def load_model():
#     global MODEL, MODEL_META
#     if not os.path.exists(MODEL_PATH):
#         print(f"[WARN] No model file at {MODEL_PATH} — falling back to heuristic classifier")
#         return
#     ckpt       = torch.load(MODEL_PATH, map_location=DEVICE)
#     cfg        = ckpt["model_config"]
#     MODEL      = ExoDetectCNN(**cfg).to(DEVICE)
#     MODEL.load_state_dict(ckpt["model_state_dict"])
#     MODEL.eval()
#     MODEL_META = ckpt.get("metrics", {})
#     prep       = ckpt.get("preprocessing", {})
#     print(f"[INFO] CNN model loaded — Val AUC: {MODEL_META.get('best_val_auc', '?'):.4f}")
#     print(f"[INFO] Preprocessing: global={prep.get('global_view_len')}, local={prep.get('local_view_len')}")


# # ── Preprocessing (must match the notebook exactly) ────────────────────────

# GLOBAL_VIEW_LEN = 201
# LOCAL_VIEW_LEN  = 81

# def median_smooth(flux, window=51):
#     return flux / median_filter(flux.astype(float), size=window, mode='reflect')


# def phase_fold_and_bin(time, flux, period, t0, n_bins, local=False, local_width=0.2):
#     phase = ((time - t0) % period) / period
#     phase = np.where(phase > 0.5, phase - 1, phase)

#     if local:
#         mask  = np.abs(phase) < local_width
#         phase = phase[mask]; flux = flux[mask]
#         if len(phase) < 5:
#             return None
#         p_min, p_max = -local_width, local_width
#     else:
#         p_min, p_max = -0.5, 0.5

#     bins = np.linspace(p_min, p_max, n_bins + 1)
#     idx  = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
#     view = np.array([
#         np.median(flux[idx == b]) if np.any(idx == b) else 1.0
#         for b in range(n_bins)
#     ])
#     view = np.where(np.isfinite(view), view, 1.0)
#     oot  = np.abs(np.linspace(p_min, p_max, n_bins)) > 0.05
#     if oot.sum() > 3:
#         mu = np.median(view[oot]); sig = np.std(view[oot]) + 1e-8
#         view = (view - mu) / sig
#     return view.astype(np.float32)


# def cnn_classify(period, duration_hr, depth_ppm, snr, time, flux):
#     """Run the trained CNN. Returns dict with probabilities for all classes."""
#     if MODEL is None:
#         return heuristic_classify(period, depth_ppm / 1e6 * 100, duration_hr, snr)

#     t0_guess = time[np.argmin(flux)]  # simple transit centre estimate

#     gv = phase_fold_and_bin(time, flux, period, t0_guess, GLOBAL_VIEW_LEN, local=False)
#     lv = phase_fold_and_bin(time, flux, period, t0_guess, LOCAL_VIEW_LEN,  local=True)

#     if gv is None or lv is None:
#         return heuristic_classify(period, depth_ppm / 1e6 * 100, duration_hr, snr)

#     gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0)   # (1,1,201)
#     lv_t = torch.tensor(lv).unsqueeze(0).unsqueeze(0)   # (1,1,81)
#     sf_t = torch.tensor([[
#         float(np.log1p(period)),
#         float(np.log1p(max(duration_hr, 0))),
#         float(np.log1p(max(depth_ppm, 0))) / 10,
#         float(np.clip(snr, 0, 100)) / 100,
#     ]])

#     p_planet = MODEL.predict_proba(gv_t, lv_t, sf_t)
#     p_fp     = 1.0 - p_planet

#     # Split false-positive probability across EB/Blend/Starspot
#     # using a secondary heuristic for sub-classification
#     if depth_ppm > 50000:     # >5% depth → likely EB
#         p_eb, p_blend, p_spot = p_fp * 0.65, p_fp * 0.25, p_fp * 0.10
#     elif duration_hr / (period * 24 + 1e-6) > 0.15:
#         p_eb, p_blend, p_spot = p_fp * 0.20, p_fp * 0.65, p_fp * 0.15
#     else:
#         p_eb, p_blend, p_spot = p_fp * 0.35, p_fp * 0.40, p_fp * 0.25

#     return {
#         "Exoplanet Transit": round(float(p_planet) * 100, 1),
#         "Eclipsing Binary":  round(float(p_eb)     * 100, 1),
#         "Stellar Blend":     round(float(p_blend)   * 100, 1),
#         "Starspot":          round(float(p_spot)    * 100, 1),
#     }


# def heuristic_classify(period, depth_pct, duration_hr, snr):
#     """Fallback if no model file is present."""
#     if depth_pct > 5:
#         return {"Exoplanet Transit": 8.0, "Eclipsing Binary": 81.0,
#                 "Stellar Blend": 7.0, "Starspot": 4.0}
#     if snr < 7:
#         return {"Exoplanet Transit": 30.0, "Eclipsing Binary": 20.0,
#                 "Stellar Blend": 25.0, "Starspot": 25.0}
#     return {"Exoplanet Transit": 91.0, "Eclipsing Binary": 5.0,
#             "Stellar Blend": 3.0, "Starspot": 1.0}


# # ── Known targets ──────────────────────────────────────────────────────────

# KNOWN_TARGETS = {
#     "L 98-59":       {"type": "Exoplanet Transit", "tic": "TIC 307210830", "note": "3 terrestrial planets"},
#     "TOI-700":       {"type": "Exoplanet Transit", "tic": "TIC 150428135", "note": "Habitable-zone Earth-size"},
#     "WASP-18":       {"type": "Exoplanet Transit", "tic": "TIC 100100827", "note": "Hot Jupiter, ~1 day period"},
#     "TIC 286923464": {"type": "Exoplanet Transit", "tic": "TIC 286923464", "note": "HD 118203 b — eccentric"},
#     "HD 21749":      {"type": "Exoplanet Transit", "tic": "TIC 279741379", "note": "Sub-Neptune + super-Earth"},
# }


# # ── Core pipeline ──────────────────────────────────────────────────────────

# def fetch_and_analyse(target_name):
#     print(f"[INFO] Analysing: {target_name}")

#     search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
#     if len(search) == 0:
#         search = lk.search_lightcurve(target_name, mission="TESS")
#     if len(search) == 0:
#         raise ValueError(f"No TESS light curves found for '{target_name}'")

#     lc = search[0].download(flux_column="pdcsap_flux")
#     lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

#     time  = lc.time.value
#     flux  = lc.flux.value
#     ferr  = lc.flux_err.value if lc.flux_err is not None else np.full_like(flux, 0.001)
#     try:
#         sector = int(np.atleast_1d(search[0].sequence_number)[0]) if hasattr(search[0], "sequence_number") else "?"
#     except Exception:
#         sector = "?"
#     try:
#         exptime = float(np.atleast_1d(search[0].exptime.value)[0]) if hasattr(search[0], "exptime") else 120.0
#     except Exception:
#         exptime = 120.0

#     # BLS periodogram
#     t_arr  = np.array(time)
#     f_arr  = np.array(flux)
#     fe_arr = np.array(ferr)

#     bls      = BoxLeastSquares(t_arr, f_arr, fe_arr)
#     periods  = np.linspace(0.5, 27.0, 5000)
#     power    = bls.power(periods, np.linspace(0.05, 0.3, 20))
#     best_idx = np.argmax(power.power)

#     best_period = float(power.period[best_idx])
#     best_dur    = float(power.duration[best_idx])
#     best_t0     = float(power.transit_time[best_idx])
#     best_depth  = float(power.depth[best_idx])

#     # Downsample periodogram for JSON
#     stride  = max(1, len(periods) // 500)
#     periodo = {
#         "periods":     power.period[::stride].tolist(),
#         "power":       (power.power[::stride] / np.max(power.power)).tolist(),
#         "peak_period": best_period,
#     }

#     # SNR
#     in_tr = np.abs(((t_arr - best_t0) % best_period) - best_period / 2) < best_dur / 2
#     noise  = np.std(f_arr[~in_tr]) if (~in_tr).sum() > 10 else 1e-4
#     signal = abs(np.mean(f_arr[in_tr]) - np.mean(f_arr[~in_tr])) if in_tr.sum() > 0 else 0
#     snr    = float(signal / noise * np.sqrt(max(in_tr.sum(), 1)))

#     # Phase-fold
#     phase   = ((t_arr - best_t0 + best_period/2) % best_period) / best_period - 0.5
#     sort_ix = np.argsort(phase)
#     phase_folded = {
#         "phase": phase[sort_ix].tolist(),
#         "flux":  f_arr[sort_ix].tolist(),
#     }

#     n_transits  = max(1, int((t_arr[-1] - t_arr[0]) / best_period))
#     depth_ppm   = best_depth * 1e6

#     # CNN classification (or heuristic fallback)
#     smoothed_flux = median_smooth(f_arr)
#     probs = cnn_classify(best_period, best_dur * 24, depth_ppm, snr, t_arr, smoothed_flux)
#     top_class = max(probs, key=probs.get)

#     cdpp          = float(lc.estimate_cdpp().value) if hasattr(lc, "estimate_cdpp") else 200.0
#     completeness  = float(100 * (1 - np.isnan(lc.flux.value).sum() / len(lc.flux.value)))

#     return {
#         "target":       target_name,
#         "sector":       sector,
#         "exptime_sec":  exptime,
#         "n_cadences":   len(time),
#         "classifier":   "CNN (ExoDetectCNN)" if MODEL is not None else "Heuristic (no model)",
#         "model_auc":    MODEL_META.get("test_auc", None),
#         "light_curve":  {
#             "time":     time.tolist(),
#             "flux":     flux.tolist(),
#             "flux_err": ferr.tolist(),
#         },
#         "bls":          periodo,
#         "phase_folded": phase_folded,
#         "transit_params": {
#             "period_days":    round(best_period, 4),
#             "period_err":     round(best_period * 0.001, 5),
#             "depth_pct":      round(best_depth * 100, 4),
#             "duration_hours": round(best_dur * 24, 3),
#             "duration_err":   round(best_dur * 24 * 0.05, 4),
#             "t0_bjd":         round(best_t0, 5),
#             "n_transits":     n_transits,
#             "snr":            round(snr, 2),
#             "rp_rs":          round(np.sqrt(max(best_depth, 0)), 4),
#         },
#         "classification": {
#             "top_class":     top_class,
#             "confidence":    round(probs[top_class], 1),
#             "probabilities": probs,
#         },
#         "data_quality": {
#             "completeness_pct": round(completeness, 1),
#             "cdpp_ppm":         round(cdpp, 1),
#             "sys_noise_ppm":    round(cdpp * 0.15, 1),
#         },
#     }


# # ── Routes ─────────────────────────────────────────────────────────────────

# @app.route("/api/targets", methods=["GET"])
# def get_targets():
#     return jsonify({"targets": [
#         {"name": k, "tic": v["tic"], "type": v["type"], "note": v["note"]}
#         for k, v in KNOWN_TARGETS.items()
#     ]})


# @app.route("/api/analyse", methods=["GET"])
# def analyse():
#     target = request.args.get("target", "L 98-59")
#     try:
#         data = fetch_and_analyse(target)
#         return jsonify({"status": "ok", "data": data})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "message": str(e)}), 500


# @app.route("/api/model-info", methods=["GET"])
# def model_info():
#     return jsonify({
#         "model_loaded": MODEL is not None,
#         "model_path":   MODEL_PATH,
#         "metrics":      MODEL_META,
#         "architecture": "ExoDetectCNN — dual-branch 1D CNN (global + local view)",
#     })


# @app.route("/api/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "ok",
#         "classifier": "CNN" if MODEL is not None else "heuristic",
#     })


# # ── Startup ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     load_model()
#     print("=" * 55)
#     print("  ExoDetect Backend v2 — CNN Classifier")
#     print("  http://localhost:8000")
#     print("=" * 55)
#     app.run(debug=True, port=8000)


"""
server_phase2.py — ExoDetect Backend v3

New in Phase 2:
  POST /api/mcmc/start   → kick off async MCMC job, returns job_id
  GET  /api/mcmc/status  → poll job progress (step, pct)
  GET  /api/mcmc/result  → fetch completed results (plots, posteriors)
  GET  /api/model-info   → CNN + MCMC metadata

Apple Silicon note:
  PyTorch uses MPS automatically when available.
  All MCMC runs on CPU (numpy/scipy) — MPS not needed for emcee.
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
import torch, torch.nn as nn
import traceback, warnings, os, json
warnings.filterwarnings("ignore")

from tasks import celery_app, run_mcmc_task

app = Flask(__name__)
CORS(app)

# ── Device — MPS has operator gaps (adaptive_avg_pool1d, float64), use CPU ──
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU for inference (MPS skipped – operator gaps)")


# ── CNN model (identical to Phase 1) ─────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )
    def forward(self, x): return self.net(x)


class ExoDetectCNN(nn.Module):
    def __init__(self, global_len=201, local_len=81, n_stellar=4):
        super().__init__()
        self.global_branch = nn.Sequential(
            ConvBlock(1,  16, 5, 2), ConvBlock(16, 32, 5, 2),
            ConvBlock(32, 64, 5, 2), ConvBlock(64, 128, 3, 2),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
        )
        self.local_branch = nn.Sequential(
            ConvBlock(1, 16, 5, 2), ConvBlock(16, 32, 5, 2),
            ConvBlock(32, 64, 3, 2), nn.AdaptiveAvgPool1d(4), nn.Flatten(),
        )
        fused = (128 * 8) + (64 * 4) + n_stellar
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 256),   nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, gv, lv, sf):
        return self.head(torch.cat([self.global_branch(gv),
                                    self.local_branch(lv), sf], dim=1))

    def predict_proba(self, gv, lv, sf):
        with torch.no_grad():
            return torch.softmax(self.forward(gv, lv, sf), dim=1)[:, 1].item()


MODEL, MODEL_META = None, {}

def load_model(path=None):
    global MODEL, MODEL_META
    path = path or os.path.join(os.path.dirname(__file__), "exodetect_cnn.pt")
    if not os.path.exists(path):
        print(f"[WARN] No model at {path} — heuristic fallback active")
        return
    ckpt       = torch.load(path, map_location="cpu")
    cfg        = ckpt["model_config"]
    MODEL      = ExoDetectCNN(**cfg).to(DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    MODEL_META = ckpt.get("metrics", {})
    print(f"[INFO] CNN loaded on {DEVICE} — Val AUC {MODEL_META.get('best_val_auc','?')}")


# ── Preprocessing (must match notebook exactly) ───────────────────────────

GLOBAL_VIEW_LEN = 201
LOCAL_VIEW_LEN  = 81

def _smooth(flux, w=51):
    return flux / median_filter(flux.astype(float), size=w, mode="reflect")

def _fold_bin(time, flux, period, t0, n_bins, local=False, lw=0.2):
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)
    if local:
        m = np.abs(phase) < lw
        phase, flux = phase[m], flux[m]
        if len(phase) < 5: return None
        p0, p1 = -lw, lw
    else:
        p0, p1 = -0.5, 0.5
    bins = np.linspace(p0, p1, n_bins + 1)
    idx  = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    view = np.array([np.median(flux[idx == b]) if np.any(idx == b) else 1.0
                     for b in range(n_bins)])
    view = np.where(np.isfinite(view), view, 1.0)
    oot  = np.abs(np.linspace(p0, p1, n_bins)) > 0.05
    if oot.sum() > 3:
        mu, sig = np.median(view[oot]), np.std(view[oot]) + 1e-8
        view = (view - mu) / sig
    return view.astype(np.float32)

def cnn_classify(period, dur_hr, depth_ppm, snr, time, flux):
    if MODEL is None:
        return _heuristic(period, depth_ppm / 1e4, dur_hr, snr)
    sf = _smooth(flux)
    t0 = time[np.argmin(flux)]
    gv = _fold_bin(time, sf, period, t0, GLOBAL_VIEW_LEN)
    lv = _fold_bin(time, sf, period, t0, LOCAL_VIEW_LEN, local=True)
    if gv is None or lv is None:
        return _heuristic(period, depth_ppm / 1e4, dur_hr, snr)
    gv_t = torch.tensor(gv).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    lv_t = torch.tensor(lv).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    sf_t = torch.tensor([[np.log1p(period),
                          np.log1p(max(dur_hr, 0)),
                          np.log1p(max(depth_ppm, 0)) / 10,
                          np.clip(snr, 0, 100) / 100]]).float().to(DEVICE)
    p_planet = MODEL.predict_proba(gv_t, lv_t, sf_t)
    p_fp     = 1.0 - p_planet
    if depth_ppm > 50000:
        eb, bl, sp = 0.65, 0.25, 0.10
    elif dur_hr / (period * 24 + 1e-6) > 0.15:
        eb, bl, sp = 0.20, 0.65, 0.15
    else:
        eb, bl, sp = 0.35, 0.40, 0.25
    return {
        "Exoplanet Transit": round(p_planet * 100, 1),
        "Eclipsing Binary":  round(p_fp * eb * 100, 1),
        "Stellar Blend":     round(p_fp * bl * 100, 1),
        "Starspot":          round(p_fp * sp * 100, 1),
    }

def _heuristic(period, depth_pct, dur_hr, snr):
    if depth_pct > 5:
        return {"Exoplanet Transit": 8.0, "Eclipsing Binary": 81.0,
                "Stellar Blend": 7.0, "Starspot": 4.0}
    if snr < 7:
        return {"Exoplanet Transit": 30.0, "Eclipsing Binary": 20.0,
                "Stellar Blend": 25.0, "Starspot": 25.0}
    return {"Exoplanet Transit": 91.0, "Eclipsing Binary": 5.0,
            "Stellar Blend": 3.0, "Starspot": 1.0}


# ── Core BLS pipeline (quick scan) ───────────────────────────────────────

KNOWN_TARGETS = {
    "L 98-59":       {"tic": "TIC 307210830", "note": "3 terrestrial planets"},
    "TOI-700":       {"tic": "TIC 150428135", "note": "Habitable-zone Earth-size"},
    "WASP-18":       {"tic": "TIC 100100827", "note": "Hot Jupiter, ~1 day period"},
    "TIC 286923464": {"tic": "TIC 286923464", "note": "HD 118203 b — eccentric"},
    "HD 21749":      {"tic": "TIC 279741379", "note": "Sub-Neptune + super-Earth"},
}

def fetch_and_analyse(target_name):
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        raise ValueError(f"No TESS data for '{target_name}'")

    lc = search[0].download(flux_column="pdcsap_flux")
    lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

    time  = np.array(lc.time.value)
    flux  = np.array(lc.flux.value)
    ferr  = np.array(lc.flux_err.value) if lc.flux_err is not None \
            else np.full_like(flux, 5e-4)

    sector  = int(search[0].sequence_number) if hasattr(search[0], "sequence_number") else "?"
    exptime = float(np.atleast_1d(search[0].exptime.value)[0]) if hasattr(search[0], "exptime") else 120.0

    bls      = BoxLeastSquares(time, flux, ferr)
    periods  = np.linspace(0.5, 27.0, 5000)
    # max duration must be strictly < min period (0.5 d); cap at 0.4 d
    durations = np.linspace(0.01, 0.4, 20)
    power    = bls.power(periods, durations)
    bi       = np.argmax(power.power)

    period   = float(power.period[bi])
    dur      = float(power.duration[bi])
    t0       = float(power.transit_time[bi])
    depth    = float(power.depth[bi])

    stride = max(1, len(periods) // 500)
    bls_out = {
        "periods":     power.period[::stride].tolist(),
        "power":       (power.power[::stride] / np.max(power.power)).tolist(),
        "peak_period": period,
    }

    in_tr  = np.abs(((time - t0) % period) - period / 2) < dur / 2
    noise  = np.std(flux[~in_tr]) if (~in_tr).sum() > 10 else 1e-4
    sig    = abs(np.mean(flux[in_tr]) - np.mean(flux[~in_tr])) if in_tr.sum() > 0 else 0
    snr    = float(sig / noise * np.sqrt(max(in_tr.sum(), 1)))

    phase   = ((time - t0 + period / 2) % period) / period - 0.5
    srt     = np.argsort(phase)
    phase_folded = {"phase": phase[srt].tolist(), "flux": flux[srt].tolist()}

    probs     = cnn_classify(period, dur * 24, depth * 1e6, snr, time, flux)
    top_class = max(probs, key=probs.get)

    cdpp         = float(lc.estimate_cdpp().value) if hasattr(lc, "estimate_cdpp") else 200.0
    completeness = 100 * (1 - np.isnan(lc.flux.value).sum() / len(lc.flux.value))

    return {
        "target":      target_name,
        "sector":      sector,
        "exptime_sec": exptime,
        "n_cadences":  len(time),
        "classifier":  "CNN (ExoDetectCNN)" if MODEL else "Heuristic",
        "model_auc":   MODEL_META.get("test_auc"),
        "light_curve": {"time": time.tolist(), "flux": flux.tolist(),
                        "flux_err": ferr.tolist()},
        "bls":         bls_out,
        "phase_folded": phase_folded,
        "transit_params": {
            "period_days":    round(period, 4),
            "period_err":     round(period * 0.001, 5),
            "depth_pct":      round(depth * 100, 4),
            "duration_hours": round(dur * 24, 3),
            "duration_err":   round(dur * 24 * 0.05, 4),
            "t0_bjd":         round(t0, 5),
            "n_transits":     max(1, int((time[-1] - time[0]) / period)),
            "snr":            round(snr, 2),
            "rp_rs":          round(np.sqrt(max(depth, 0)), 4),
        },
        "classification": {
            "top_class":     top_class,
            "confidence":    round(probs[top_class], 1),
            "probabilities": probs,
        },
        "data_quality": {
            "completeness_pct": round(float(completeness), 1),
            "cdpp_ppm":         round(float(cdpp), 1),
            "sys_noise_ppm":    round(float(cdpp) * 0.15, 1),
        },
    }


# ── Routes — existing ─────────────────────────────────────────────────────

@app.route("/api/targets")
def get_targets():
    return jsonify({"targets": [
        {"name": k, **v} for k, v in KNOWN_TARGETS.items()
    ]})

@app.route("/api/analyse")
def analyse():
    target = request.args.get("target", "L 98-59")
    try:
        return jsonify({"status": "ok", "data": fetch_and_analyse(target)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({
        "status":     "ok",
        "device":     str(DEVICE),
        "classifier": "CNN" if MODEL else "heuristic",
    })

@app.route("/api/model-info")
def model_info():
    return jsonify({
        "model_loaded": MODEL is not None,
        "device":       str(DEVICE),
        "metrics":      MODEL_META,
        "phase":        2,
        "mcmc_backend": "emcee 3.1 + batman 2.4",
    })


# ── Routes — Phase 2 MCMC ─────────────────────────────────────────────────

@app.route("/api/mcmc/start", methods=["POST"])
def mcmc_start():
    """
    Kick off an async MCMC job.

    Body (JSON):
      { "target": "L 98-59", "n_walkers": 32, "n_steps": 2000, "n_burn": 500 }

    Returns:
      { "job_id": "...", "status": "queued" }
    """
    body      = request.get_json(force=True, silent=True) or {}
    target    = body.get("target", "L 98-59")
    n_walkers = int(body.get("n_walkers", 32))
    n_steps   = int(body.get("n_steps",  2000))
    n_burn    = int(body.get("n_burn",   500))

    task = run_mcmc_task.apply_async(
        args=[target, n_walkers, n_steps, n_burn]
    )
    return jsonify({"job_id": task.id, "status": "queued",
                    "target": target})


@app.route("/api/mcmc/status")
def mcmc_status():
    """
    Poll job progress.
    GET /api/mcmc/status?job_id=<id>

    Returns:
      { "state": "PROGRESS"|"SUCCESS"|"FAILURE",
        "step": "…", "pct": 42 }
    """
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id required"}), 400

    task = celery_app.AsyncResult(job_id)

    if task.state == "PENDING":
        return jsonify({"state": "PENDING", "step": "Queued…", "pct": 0})

    if task.state == "PROGRESS":
        meta = task.info or {}
        return jsonify({"state": "PROGRESS",
                        "step": meta.get("step", "Running…"),
                        "pct":  meta.get("pct",  0)})

    if task.state == "SUCCESS":
        result = task.result
        if result.get("status") == "error":
            return jsonify({"state": "FAILURE",
                            "message": result.get("message", "Unknown error")})
        return jsonify({"state": "SUCCESS", "pct": 100,
                        "step": "Done",
                        "elapsed_sec": result.get("elapsed_sec"),
                        "acceptance":  result.get("acceptance_frac")})

    # FAILURE
    return jsonify({"state": "FAILURE",
                    "message": str(task.info)}), 500


@app.route("/api/mcmc/result")
def mcmc_result():
    """
    Fetch full MCMC result (only after state == SUCCESS).
    GET /api/mcmc/result?job_id=<id>

    Returns the full result dict including:
      - percentiles (median, err_low, err_high per parameter)
      - model_flux  (best-fit transit model over original time array)
      - plots.corner    (base64 PNG)
      - plots.posterior (base64 PNG)
      - light_curve (time + flux arrays)
      - bls_init
    """
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id required"}), 400

    task = celery_app.AsyncResult(job_id)
    if task.state != "SUCCESS":
        return jsonify({"error": f"Job not done yet (state={task.state})"}), 400

    result = task.result
    if result.get("status") == "error":
        return jsonify({"status": "error",
                        "message": result.get("message")}), 500

    return jsonify({"status": "ok", "data": result})


if __name__ == "__main__":
    load_model()
    print("=" * 60)
    print("  ExoDetect Backend v3 — Phase 2 (CNN + MCMC)")
    print(f"  Device : {DEVICE}")
    print("  http://localhost:8000")
    print("=" * 60)
    app.run(debug=True, port=8000)