"""
mcmc_fitter.py — Phase 2 MCMC Transit Parameter Estimator

Uses:
  batman  → physical transit light curve model
  emcee   → affine-invariant MCMC ensemble sampler
  corner  → posterior corner plots

Apple Silicon note:
  All numpy/scipy operations run natively on ARM.
  batman uses C extensions that compile fine under Rosetta-free arm64.
"""

import numpy as np
import batman
import emcee
import corner
import matplotlib
matplotlib.use("Agg")          # non-interactive — safe for background workers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import median_filter
import io, base64, warnings, time
warnings.filterwarnings("ignore")


# ── Batman transit model ───────────────────────────────────────────────────

def make_batman_params(theta):
    """
    Map MCMC parameter vector → batman TransitParams.

    theta = [log_period, t0, rp_rs, log_ars, cos_inc, u1, u2]

    Using log(period) and log(a/Rs) keeps walkers from going negative.
    cos(inc) keeps inc in [0, π/2] naturally.
    Kipping (2013) triangular sampling for u1, u2.
    """
    log_period, t0, rp_rs, log_ars, cos_inc, u1, u2 = theta

    p         = batman.TransitParams()
    p.t0      = t0
    p.per     = np.exp(log_period)
    p.rp      = rp_rs                        # Rp / R★
    p.a       = np.exp(log_ars)              # a  / R★
    p.inc     = np.degrees(np.arccos(cos_inc))
    p.ecc     = 0.0                          # circular orbit (Phase 2)
    p.w       = 90.0
    p.u       = [u1, u2]
    p.limb_dark = "quadratic"
    return p


def transit_model(theta, time_arr):
    """Evaluate batman model at given times. Returns flux array."""
    try:
        params = make_batman_params(theta)
        m      = batman.TransitModel(params, time_arr)
        return m.light_curve(params)
    except Exception:
        return np.ones_like(time_arr)


# ── Log-probability for emcee ──────────────────────────────────────────────

def log_prior(theta):
    """
    Uninformative (uniform) priors with hard physical boundaries.

    log_period : 0 → log(30)   days
    t0         : free (transit epoch)
    rp_rs      : 0.01 → 0.30   (1% to 30% radius ratio)
    log_ars    : log(1.5) → log(100)
    cos_inc    : 0 → 1          (inc 0°→90°, grazing→central)
    u1, u2     : Kipping (2013) u1>0, u1+u2<1, u2>-1
    """
    log_per, t0, rp_rs, log_ars, cos_inc, u1, u2 = theta

    if not (0.0 < log_per < np.log(30.0)):
        return -np.inf
    if not (0.01 < rp_rs < 0.30):
        return -np.inf
    if not (np.log(1.5) < log_ars < np.log(100.0)):
        return -np.inf
    if not (0.0 <= cos_inc <= 1.0):
        return -np.inf
    # Kipping (2013) limb-darkening prior
    if not (u1 > 0 and u1 + u2 < 1.0 and u2 > -1.0):
        return -np.inf
    return 0.0


def log_likelihood(theta, time_arr, flux_arr, flux_err):
    model  = transit_model(theta, time_arr)
    sigma2 = flux_err ** 2
    return -0.5 * np.sum((flux_arr - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, time_arr, flux_arr, flux_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, time_arr, flux_arr, flux_err)
    return lp + ll if np.isfinite(ll) else -np.inf


# ── Phase-folded data preparation ──────────────────────────────────────────

def prepare_phased_data(time_arr, flux_arr, flux_err, period, t0, n_durations=3.0):
    """
    Extract only the in-transit + near-transit region for MCMC.
    Speeds up likelihood evaluation significantly.
    Duration estimated from BLS; we keep ±n_durations around each transit.
    """
    phase   = ((time_arr - t0 + period / 2) % period) - period / 2
    # Keep points within ±20% of period around transit centre
    width   = min(period * 0.20, 1.5)    # days
    mask    = np.abs(phase) < width

    if mask.sum() < 20:
        mask = np.ones(len(time_arr), dtype=bool)   # fallback: use all

    return time_arr[mask], flux_arr[mask], flux_err[mask]


# ── Main MCMC runner ───────────────────────────────────────────────────────

def run_mcmc(
    time_arr, flux_arr, flux_err,
    period_bls, t0_bls, depth_bls, duration_bls_days,
    n_walkers=32,
    n_steps=2000,
    n_burn=500,
    progress_callback=None,
):
    """
    Run emcee MCMC transit fit.

    Parameters
    ----------
    time_arr, flux_arr, flux_err : real TESS arrays
    period_bls    : BLS best-fit period (days)
    t0_bls        : BLS transit epoch (BTJD)
    depth_bls     : BLS depth (fraction, not ppm)
    duration_bls_days : BLS transit duration (days)
    n_walkers     : emcee ensemble size (must be even, ≥ 2×ndim)
    n_steps       : MCMC steps per walker
    n_burn        : burn-in steps to discard
    progress_callback : optional fn(step, total) for WebSocket streaming

    Returns
    -------
    dict with samples, percentiles, plots as base64 PNG
    """

    # ── Initial parameter estimates from BLS ──────────────────────────────
    rp_rs_init  = np.sqrt(max(depth_bls, 1e-6))
    # Kepler's third law approximation for a/R★ (assumes M★~M☉, R★~R☉)
    # a/R★ = (G M★ / (4π²))^(1/3) × P^(2/3) / R★  ≈  4.2 × P^(2/3) for solar
    ars_init    = max(4.2 * period_bls ** (2/3), 2.0)
    inc_init    = 85.0                               # degrees — nearly edge-on
    cos_inc_init = np.cos(np.radians(inc_init))

    theta_init  = np.array([
        np.log(period_bls),   # log_period
        t0_bls,               # t0
        rp_rs_init,           # rp_rs
        np.log(ars_init),     # log_ars
        cos_inc_init,         # cos_inc
        0.3,                  # u1  (typical solar-type star)
        0.2,                  # u2
    ])
    ndim = len(theta_init)

    # ── Use only near-transit data for speed ─────────────────────────────
    t_fit, f_fit, fe_fit = prepare_phased_data(
        time_arr, flux_arr, flux_err, period_bls, t0_bls
    )

    print(f"[MCMC] Fitting {len(t_fit)} near-transit cadences")
    print(f"[MCMC] Init: P={period_bls:.4f}d, rp/R★={rp_rs_init:.4f}, a/R★={ars_init:.2f}")

    # ── Initialise walkers with small gaussian ball ────────────────────────
    scales  = np.array([0.001, 0.01, 0.005, 0.02, 0.01, 0.05, 0.05])
    pos     = theta_init + scales * np.random.randn(n_walkers, ndim)

    # Clip to valid prior range
    pos[:, 2] = np.clip(pos[:, 2], 0.01, 0.29)    # rp_rs
    pos[:, 4] = np.clip(pos[:, 4], 0.0,  1.0)     # cos_inc

    # ── Run sampler ───────────────────────────────────────────────────────
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(t_fit, f_fit, fe_fit),
    )

    t_start = time.time()

    # Burn-in
    print(f"[MCMC] Burn-in: {n_burn} steps × {n_walkers} walkers…")
    pos, prob, state = sampler.run_mcmc(pos, n_burn, progress=False)
    sampler.reset()

    # Production run with optional progress streaming
    print(f"[MCMC] Production: {n_steps} steps × {n_walkers} walkers…")
    for i, result in enumerate(sampler.sample(pos, iterations=n_steps, progress=False)):
        if progress_callback and i % 100 == 0:
            progress_callback(i, n_steps)

    elapsed = time.time() - t_start
    print(f"[MCMC] Done in {elapsed:.1f}s")

    # ── Extract flat samples ───────────────────────────────────────────────
    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)

    # ── Derive physical parameters from samples ───────────────────────────
    # Convert log_period → period, log_ars → ars, cos_inc → inc
    derived = np.column_stack([
        np.exp(flat_samples[:, 0]),                         # period
        flat_samples[:, 1],                                  # t0
        flat_samples[:, 2],                                  # rp_rs
        np.exp(flat_samples[:, 3]),                          # ars
        np.degrees(np.arccos(flat_samples[:, 4])),           # inc (degrees)
        flat_samples[:, 5],                                  # u1
        flat_samples[:, 6],                                  # u2
        flat_samples[:, 2] ** 2 * 1e6,                      # depth_ppm  = rp²×1e6
        flat_samples[:, 2] * 109.2,                         # rp_earth   ≈ rp_rs × 109.2 R⊕/R☉
    ])

    param_names = ["Period (d)", "T₀ (BTJD)", "Rp/R★",
                   "a/R★", "Inc (°)", "u₁", "u₂",
                   "Depth (ppm)", "Rp (R⊕)"]

    # ── Percentile summaries ──────────────────────────────────────────────
    percentiles = {}
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(derived[:, i], [16, 50, 84])
        percentiles[name] = {
            "median":   round(float(q50), 5),
            "err_low":  round(float(q50 - q16), 5),
            "err_high": round(float(q84 - q50), 5),
        }

    # Acceptance fraction (healthy: 0.2–0.5)
    acceptance = float(np.mean(sampler.acceptance_fraction))
    print(f"[MCMC] Mean acceptance fraction: {acceptance:.3f}")

    # ── Best-fit transit model overlay ────────────────────────────────────
    theta_med = np.array([
        np.log(percentiles["Period (d)"]["median"]),
        percentiles["T₀ (BTJD)"]["median"],
        percentiles["Rp/R★"]["median"],
        np.log(percentiles["a/R★"]["median"]),
        np.cos(np.radians(percentiles["Inc (°)"]["median"])),
        percentiles["u₁"]["median"],
        percentiles["u₂"]["median"],
    ])

    model_flux      = transit_model(theta_med, time_arr)
    model_flux_list = model_flux.tolist()

    # ── Corner plot ───────────────────────────────────────────────────────
    corner_b64 = _make_corner_plot(derived[:, :5],
                                    param_names[:5],
                                    theta_med)

    # ── Posterior predictive plot ─────────────────────────────────────────
    posterior_b64 = _make_posterior_plot(
        time_arr, flux_arr, model_flux,
        derived, theta_med, period_bls, t0_bls
    )

    return {
        "status":           "done",
        "elapsed_sec":      round(elapsed, 1),
        "n_samples":        len(flat_samples),
        "acceptance_frac":  round(acceptance, 3),
        "percentiles":      percentiles,
        "model_flux":       model_flux_list,
        "plots": {
            "corner":    corner_b64,
            "posterior": posterior_b64,
        },
    }


# ── Plot helpers ───────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0A1628", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _make_corner_plot(samples, labels, theta_med):
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
        color="#00D4FF",
        truth_color="#FFB347",
        truths=theta_med[:len(labels)] if len(theta_med) >= len(labels) else None,
        plot_contours=True,
        fill_contours=True,
        bins=30,
        smooth=1.0,
        fig=None,
    )
    fig.patch.set_facecolor("#0A1628")
    for ax in fig.axes:
        ax.set_facecolor("#060E1E")
        ax.tick_params(colors="#8899BB", labelsize=7)
        ax.xaxis.label.set_color("#8899BB")
        ax.yaxis.label.set_color("#8899BB")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0E2040")
    fig.suptitle("MCMC Posterior — Transit Parameters", color="#E8F0FF",
                 fontsize=11, y=1.01)
    return _fig_to_b64(fig)


def _make_posterior_plot(time_arr, flux_arr, model_flux,
                          derived_samples, theta_med,
                          period, t0):
    """
    Two-panel figure:
      Top  — phase-folded data + best-fit model + 1σ envelope
      Bottom — residuals
    """
    phase = ((time_arr - t0 + period / 2) % period) - period / 2
    sort_idx  = np.argsort(phase)
    ph_sorted = phase[sort_idx]
    fl_sorted = flux_arr[sort_idx]
    mo_sorted = model_flux[sort_idx]

    # Sample 200 random posterior models for the envelope
    rng      = np.random.default_rng(0)
    sample_i = rng.integers(0, len(derived_samples), size=200)
    env_models = []
    for si in sample_i:
        row = derived_samples[si]
        th  = np.array([
            np.log(row[0]),                       # log_period
            theta_med[1],                         # t0 (keep fixed)
            row[2],                               # rp_rs
            np.log(row[3]),                       # log_ars
            np.cos(np.radians(row[4])),           # cos_inc
            row[5], row[6]                        # u1, u2
        ])
        mf = transit_model(th, time_arr)
        env_models.append(mf[sort_idx])

    env_arr = np.array(env_models)
    env_lo  = np.percentile(env_arr, 16, axis=0)
    env_hi  = np.percentile(env_arr, 84, axis=0)

    residuals = fl_sorted - mo_sorted

    fig = plt.figure(figsize=(10, 6), facecolor="#0A1628")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Data
    ax1.scatter(ph_sorted, fl_sorted, s=1.5, alpha=0.4,
                color="#8899BB", label="TESS data", rasterized=True)
    # 1σ envelope
    ax1.fill_between(ph_sorted, env_lo, env_hi,
                     color="#00D4FF", alpha=0.18, label="1σ posterior")
    # Best-fit model
    ax1.plot(ph_sorted, mo_sorted, color="#00D4FF",
             linewidth=2.0, label="Best-fit model", zorder=5)

    ax1.set_ylabel("Normalised flux", color="#AAB8CC", fontsize=10)
    ax1.set_facecolor("#060E1E")
    ax1.tick_params(colors="#8899BB", labelbottom=False)
    ax1.legend(fontsize=8, facecolor="#0A1628", edgecolor="#0E2040",
               labelcolor="#AAB8CC")
    ax1.set_title("Phase-folded transit + MCMC posterior model",
                  color="#E8F0FF", fontsize=11, pad=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#0E2040")

    # Residuals
    ax2.scatter(ph_sorted, residuals, s=1.5, alpha=0.4,
                color="#FFB347", rasterized=True)
    ax2.axhline(0, color="#00D4FF", linewidth=1.0, linestyle="--")
    ax2.set_xlabel("Orbital phase", color="#AAB8CC", fontsize=10)
    ax2.set_ylabel("Residuals", color="#AAB8CC", fontsize=9)
    ax2.set_facecolor("#060E1E")
    ax2.tick_params(colors="#8899BB")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#0E2040")

    # Show only ±0.25 phase
    ax1.set_xlim(-0.25, 0.25)

    return _fig_to_b64(fig)