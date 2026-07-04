"""
tasks.py — Celery async worker for MCMC fitting jobs

Start worker with:
    celery -A tasks worker --loglevel=info --concurrency=2

Requires Redis running locally:
    brew install redis
    brew services start redis
"""

from celery import Celery
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from mcmc_fitter import run_mcmc
import warnings, traceback
warnings.filterwarnings("ignore")

# ── Celery app ─────────────────────────────────────────────────────────────
# Redis as both broker and result backend
celery_app = Celery(
    "exodetect",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,          # results live for 1 hour
    worker_prefetch_multiplier=1, # one task at a time per worker
    task_track_started=True,
)


# ── Helper: fetch + BLS (shared with server_v2.py) ────────────────────────

def _fetch_light_curve(target_name):
    """Download TESS light curve and run BLS. Returns dict of arrays."""
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        raise ValueError(f"No TESS data found for '{target_name}'")

    lc = search[0].download(flux_column="pdcsap_flux")
    lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

    time  = np.array(lc.time.value,     dtype=np.float64)
    flux  = np.array(lc.flux.value,     dtype=np.float64)
    ferr  = np.array(lc.flux_err.value, dtype=np.float64) \
            if lc.flux_err is not None else np.full_like(flux, 5e-4)

    # BLS for initial parameter estimates
    bls      = BoxLeastSquares(time, flux, ferr)
    periods  = np.linspace(0.51, 27.0, 5000)
    durations = np.linspace(0.01, 0.4, 20)
    power    = bls.power(periods, durations)
    best_idx = np.argmax(power.power)

    return {
        "time":     time,
        "flux":     flux,
        "ferr":     ferr,
        "period":   float(power.period[best_idx]),
        "t0":       float(power.transit_time[best_idx]),
        "depth":    float(power.depth[best_idx]),
        "duration": float(power.duration[best_idx]),
    }


# ── Celery task ────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.run_mcmc_task")
def run_mcmc_task(self, target_name, n_walkers=32, n_steps=2000, n_burn=500):
    """
    Full async MCMC pipeline:
      1. Download TESS light curve
      2. BLS → initial parameter estimates
      3. emcee MCMC sampling
      4. Generate corner plot + posterior model plot
      5. Return results dict to Redis

    Progress is reported via Celery's self.update_state()
    so the Flask endpoint can stream it to the frontend.
    """
    try:
        # Step 1 — Download
        self.update_state(state="PROGRESS",
                          meta={"step": "Downloading light curve from MAST…",
                                "pct": 5})
        lc_data = _fetch_light_curve(target_name)

        # Step 2 — BLS summary
        self.update_state(state="PROGRESS",
                          meta={"step": f"BLS complete — P={lc_data['period']:.4f}d",
                                "pct": 20})

        # Step 3 — MCMC
        def progress_cb(step, total):
            pct = 20 + int(70 * step / total)
            self.update_state(state="PROGRESS",
                              meta={"step": f"MCMC sampling… {step}/{total} steps",
                                    "pct": pct})

        result = run_mcmc(
            time_arr=lc_data["time"],
            flux_arr=lc_data["flux"],
            flux_err=lc_data["ferr"],
            period_bls=lc_data["period"],
            t0_bls=lc_data["t0"],
            depth_bls=lc_data["depth"],
            duration_bls_days=lc_data["duration"],
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burn=n_burn,
            progress_callback=progress_cb,
        )

        # Step 4 — attach light curve for frontend overlay
        result["light_curve"] = {
            "time": lc_data["time"].tolist(),
            "flux": lc_data["flux"].tolist(),
        }
        result["bls_init"] = {
            "period":   lc_data["period"],
            "t0":       lc_data["t0"],
            "depth":    lc_data["depth"],
            "duration": lc_data["duration"],
        }
        result["target"] = target_name

        self.update_state(state="PROGRESS",
                          meta={"step": "Generating plots…", "pct": 95})

        return result

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}