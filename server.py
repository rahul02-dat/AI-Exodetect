"""
ExoDetect Backend — Real TESS Data Pipeline
Uses lightkurve to fetch real TESS light curves from MAST archive
and runs BLS periodogram + transit parameter estimation.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
import traceback
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── Known confirmed targets for the demo dropdown ──────────────────────────
KNOWN_TARGETS = {
    "L 98-59":        {"type": "Exoplanet Transit",  "tic": "TIC 307210830", "note": "3 terrestrial planets"},
    "TOI-700":        {"type": "Exoplanet Transit",  "tic": "TIC 150428135", "note": "Habitable-zone Earth-size planet"},
    "WASP-18":        {"type": "Exoplanet Transit",  "tic": "TIC 100100827", "note": "Hot Jupiter, ~1 day period"},
    "TIC 286923464":  {"type": "Exoplanet Transit",  "tic": "TIC 286923464", "note": "HD 118203 b — eccentric orbit"},
    "HD 21749":       {"type": "Exoplanet Transit",  "tic": "TIC 279741379", "note": "Sub-Neptune + super-Earth"},
    "Beta Pictoris":  {"type": "Stellar Variability","tic": "TIC 270577175", "note": "Debris disk + direct imaging planet"},
}


def classify_signal(period, depth, duration, snr):
    """
    Heuristic classifier — in production this would be a trained CNN/RF.
    Returns probability dict summing to 1.
    """
    depth_pct = depth * 100

    # Very deep dips → likely eclipsing binary
    if depth_pct > 5:
        return {
            "Exoplanet Transit": 0.08,
            "Eclipsing Binary":  0.81,
            "Stellar Blend":     0.07,
            "Starspot":          0.04,
        }
    # Short period + moderate depth → could be hot Jupiter or EB
    if period < 3 and depth_pct > 0.5:
        return {
            "Exoplanet Transit": 0.62,
            "Eclipsing Binary":  0.28,
            "Stellar Blend":     0.06,
            "Starspot":          0.04,
        }
    # Long duration relative to period → blend
    if duration / (period * 24) > 0.15:
        return {
            "Exoplanet Transit": 0.18,
            "Eclipsing Binary":  0.22,
            "Stellar Blend":     0.52,
            "Starspot":          0.08,
        }
    # SNR too low
    if snr < 7:
        return {
            "Exoplanet Transit": 0.30,
            "Eclipsing Binary":  0.20,
            "Stellar Blend":     0.25,
            "Starspot":          0.25,
        }
    # Default: planet transit
    return {
        "Exoplanet Transit": 0.91,
        "Eclipsing Binary":  0.05,
        "Stellar Blend":     0.03,
        "Starspot":          0.01,
    }


def fetch_and_analyse(target_name):
    print(f"[INFO] Fetching TESS data for: {target_name}")

    # 1. Download light curve from MAST (SPOC 2-min cadence preferred)
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        raise ValueError(f"No TESS light curves found for '{target_name}'")

    print(f"[INFO] Found {len(search)} data products. Downloading first sector...")
    lc = search[0].download(flux_column="pdcsap_flux")
    lc = lc.remove_nans().remove_outliers(sigma=5)

    # Normalise
    lc = lc.normalize()

    time  = lc.time.value.tolist()
    flux  = lc.flux.value.tolist()
    ferr  = lc.flux_err.value.tolist() if lc.flux_err is not None else [0.001]*len(time)

    try:
        sector = int(np.atleast_1d(search[0].sequence_number)[0]) if hasattr(search[0], "sequence_number") else "?"
    except Exception:
        sector = "?"

    try:
        exptime = float(np.atleast_1d(search[0].exptime.value)[0]) if hasattr(search[0], "exptime") else 120.0
    except Exception:
        exptime = 120.0

    print(f"[INFO] Light curve: {len(time)} cadences, sector {sector}")

    # 2. BLS periodogram
    t_arr  = np.array(lc.time.value)
    f_arr  = np.array(lc.flux.value)
    fe_arr = np.array(lc.flux_err.value) if lc.flux_err is not None else np.full_like(f_arr, 0.001)

    duration_grid = np.linspace(0.05, 0.3, 20)   # days
    period_grid   = np.linspace(0.5, 27.0, 5000)

    bls   = BoxLeastSquares(t_arr, f_arr, fe_arr)
    power = bls.power(period_grid, duration_grid)

    best_idx    = np.argmax(power.power)
    best_period = float(power.period[best_idx])
    best_dur    = float(power.duration[best_idx])
    best_t0     = float(power.transit_time[best_idx])
    best_depth  = float(power.depth[best_idx])

    # BLS power array (downsample for JSON)
    stride  = max(1, len(period_grid)//500)
    periodo = {
        "periods": power.period[::stride].tolist(),
        "power":   (power.power[::stride] / np.max(power.power)).tolist(),
        "peak_period": best_period,
    }

    # 3. SNR estimate
    in_transit = np.abs(((t_arr - best_t0) % best_period) - best_period/2) < best_dur/2
    out_transit = ~in_transit
    if out_transit.sum() > 10 and in_transit.sum() > 0:
        noise  = np.std(f_arr[out_transit])
        signal = np.abs(np.mean(f_arr[in_transit]) - np.mean(f_arr[out_transit]))
        snr    = float(signal / noise * np.sqrt(in_transit.sum())) if noise > 0 else 0.0
    else:
        snr = 0.0

    # 4. Phase-fold
    phase    = ((t_arr - best_t0 + best_period/2) % best_period) / best_period - 0.5
    sort_idx = np.argsort(phase)
    phase_folded = {
        "phase": phase[sort_idx].tolist(),
        "flux":  f_arr[sort_idx].tolist(),
    }

    # 5. Count transits
    span       = t_arr[-1] - t_arr[0]
    n_transits = max(1, int(span / best_period))

    # 6. Classification
    probs = classify_signal(best_period, best_depth, best_dur * 24, snr)
    top_class = max(probs, key=probs.get)

    # 7. Data quality
    cdpp = float(lc.estimate_cdpp().value) if hasattr(lc, "estimate_cdpp") else 200.0
    completeness = float(100 * (1 - np.isnan(lc.flux.value).sum() / len(lc.flux.value)))

    result = {
        "target": target_name,
        "sector": sector,
        "exptime_sec": exptime,
        "n_cadences": len(time),
        "light_curve": {
            "time": time,
            "flux": flux,
            "flux_err": ferr,
        },
        "bls": periodo,
        "phase_folded": phase_folded,
        "transit_params": {
            "period_days":    round(best_period, 4),
            "period_err":     round(best_period * 0.001, 5),
            "depth_pct":      round(best_depth * 100, 4),
            "duration_hours": round(best_dur * 24, 3),
            "duration_err":   round(best_dur * 24 * 0.05, 4),
            "t0_bjd":         round(best_t0, 5),
            "n_transits":     n_transits,
            "snr":            round(snr, 2),
            "rp_rs":          round(np.sqrt(max(best_depth, 0)), 4),
        },
        "classification": {
            "top_class":     top_class,
            "confidence":    round(probs[top_class] * 100, 1),
            "probabilities": {k: round(v * 100, 1) for k, v in probs.items()},
        },
        "data_quality": {
            "completeness_pct": round(completeness, 1),
            "cdpp_ppm":         round(cdpp, 1),
            "sys_noise_ppm":    round(cdpp * 0.15, 1),
        },
    }
    return result


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/api/targets", methods=["GET"])
def get_targets():
    return jsonify({"targets": [
        {"name": k, "tic": v["tic"], "type": v["type"], "note": v["note"]}
        for k, v in KNOWN_TARGETS.items()
    ]})


@app.route("/api/analyse", methods=["GET"])
def analyse():
    target = request.args.get("target", "L 98-59")
    try:
        data = fetch_and_analyse(target)
        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ExoDetect backend running"})


if __name__ == "__main__":
    print("=" * 55)
    print("  ExoDetect Backend — Real TESS Data Pipeline")
    print("  http://localhost:8000")
    print("=" * 55)
    app.run(debug=True, port=8000)