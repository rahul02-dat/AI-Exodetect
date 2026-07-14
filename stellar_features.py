"""
stellar_features.py — Shared stellar feature engineering

Single source of truth for turning raw transit + stellar parameters into
the fixed-length feature vector fed to both the CNN and TransitFormer
models. This used to be reimplemented independently in server.py,
batch_pipeline.py, and the training scripts — any drift between those
copies would silently break inference (a model trained on one formula,
served with a slightly different one, with no error raised, just quietly
worse predictions). Everything now imports from here instead.

Two schemas are supported, selected by n_stellar, so old and new model
checkpoints both keep working side by side during a gradual rollout:

  v1 (n_stellar=4) — transit-only features, matches every model trained
     before this module existed:
       [log1p(period), log1p(duration_hr), log1p(depth_ppm)/10, snr/100]

  v2 (n_stellar=8) — v1 plus normalized host-star parameters from the
     TESS Input Catalog:
       [..v1.., teff_norm, radius_norm, logg_norm, tmag_norm]

     Missing stellar parameters (catalog lookup failed, field absent for
     that target, etc.) are imputed as 0.0 in normalized space — i.e.
     "assume a roughly Sun-like star" — rather than crashing or requiring
     every caller to special-case missing data.
"""

import re
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ── Normalization constants ──────────────────────────────────────────────
# Centered on a Sun-like FGK dwarf (the bulk of the TESS 2-min target
# population), scaled so roughly +/-2 spans the typical host-star range.
# These are sane physical defaults, not fit to a specific dataset — if you
# want to sharpen them later, recomputing from your merged dataset.npz's
# actual percentiles is a reasonable refinement, but not required to ship.
TEFF_CENTER, TEFF_SCALE = 5772.0, 1500.0   # Kelvin (Sun = 5772 K)
RAD_LOG_SCALE            = 3.0             # applied to log1p(radius / R_sun)
LOGG_CENTER, LOGG_SCALE  = 4.4, 1.0        # cm/s^2, log-scale (Sun ~4.44)
TMAG_CENTER, TMAG_SCALE  = 10.0, 4.0       # typical SPOC 2-min target range

N_STELLAR_LEGACY   = 4
N_STELLAR_EXPANDED = 8


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _norm(value, center, scale, clip=3.0) -> float:
    """Normalize + impute-to-zero + clip a single stellar parameter."""
    v = _safe_float(value)
    if v is None:
        return 0.0
    return float(np.clip((v - center) / scale, -clip, clip))


def _norm_log(value, log_scale, clip=3.0) -> float:
    v = _safe_float(value)
    if v is None or v <= 0:
        return 0.0
    return float(np.clip(np.log1p(v) / log_scale - 1.0, -clip, clip))


# ── Feature vectors ──────────────────────────────────────────────────────

def compute_stellar_features_v1(period, duration_hr, depth_ppm, snr) -> np.ndarray:
    """The original 4-feature schema — transit parameters only."""
    return np.array([
        np.log1p(max(period or 0, 0)),
        np.log1p(max(duration_hr or 0, 0)),
        np.log1p(max(depth_ppm or 0, 0)) / 10,
        np.clip(snr or 0, 0, 100) / 100,
    ], dtype=np.float32)


def compute_stellar_features_v2(period, duration_hr, depth_ppm, snr,
                                 teff=None, rad=None, logg=None, tmag=None) -> np.ndarray:
    """Expanded 8-feature schema — v1 plus normalized host-star parameters."""
    v1 = compute_stellar_features_v1(period, duration_hr, depth_ppm, snr)
    stellar = np.array([
        _norm(teff, TEFF_CENTER, TEFF_SCALE),
        _norm_log(rad, RAD_LOG_SCALE),
        _norm(logg, LOGG_CENTER, LOGG_SCALE),
        _norm(tmag, TMAG_CENTER, TMAG_SCALE),
    ], dtype=np.float32)
    return np.concatenate([v1, stellar]).astype(np.float32)


def expand_v1_to_v2(v1_features: np.ndarray, teff=None, rad=None, logg=None, tmag=None) -> np.ndarray:
    """
    Upgrades an already-computed 4-feature (v1) vector to the 8-feature
    (v2) schema by appending normalized stellar parameters, without
    needing the raw period/duration/depth/snr again. Used when merging
    shards that already stored v1 features alongside raw stellar fields
    (see merge_tess_shards.py).
    """
    stellar = np.array([
        _norm(teff, TEFF_CENTER, TEFF_SCALE),
        _norm_log(rad, RAD_LOG_SCALE),
        _norm(logg, LOGG_CENTER, LOGG_SCALE),
        _norm(tmag, TMAG_CENTER, TMAG_SCALE),
    ], dtype=np.float32)
    return np.concatenate([np.asarray(v1_features, dtype=np.float32), stellar]).astype(np.float32)


def build_stellar_features(n_stellar: int, period, duration_hr, depth_ppm, snr,
                            teff=None, rad=None, logg=None, tmag=None) -> np.ndarray:
    """
    Dispatches to the right schema based on what a given model checkpoint
    expects (its model_config['n_stellar']), so a CNN still on the legacy
    4-feature schema and a TransitFormer already upgraded to 8 features
    (or vice versa) can be ensembled together during a gradual rollout.
    """
    if n_stellar == N_STELLAR_LEGACY:
        return compute_stellar_features_v1(period, duration_hr, depth_ppm, snr)
    if n_stellar == N_STELLAR_EXPANDED:
        return compute_stellar_features_v2(period, duration_hr, depth_ppm, snr,
                                            teff, rad, logg, tmag)
    raise ValueError(
        f"Unsupported n_stellar={n_stellar} — expected {N_STELLAR_LEGACY} (legacy) "
        f"or {N_STELLAR_EXPANDED} (expanded). Did a model checkpoint get saved with "
        f"a custom stellar feature count?"
    )


# ── Live TIC catalog lookups (used at inference time in server.py / batch_pipeline.py) ──

def extract_tic_id(search_result_row) -> Optional[int]:
    """
    Pulls the numeric TIC ID out of a lightkurve SearchResult row. Best
    effort across a couple of attribute-name variants seen across
    lightkurve versions; returns None (never raises) if it can't find
    one — callers must treat a missing TIC ID as "no stellar params
    available", not a fatal error.
    """
    for attr in ("target_name", "targetid", "ID"):
        try:
            val = getattr(search_result_row, attr, None)
            if val is None:
                continue
            if hasattr(val, "__getitem__") and not isinstance(val, str):
                val = val[0]
            m = re.search(r"(\d+)", str(val))
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None


def fetch_stellar_params(tic_id: Optional[int]) -> dict:
    """
    Queries the TESS Input Catalog (TIC) for one target's stellar
    parameters. Never raises — a failed or unavailable lookup returns all
    None, which build_stellar_features() imputes to neutral defaults
    rather than blocking inference. Requires astroquery; degrades
    gracefully (all-None) if it isn't installed.
    """
    empty = {"teff": None, "rad": None, "logg": None, "tmag": None}
    if tic_id is None:
        return empty
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        return empty
    try:
        table = Catalogs.query_criteria(catalog="Tic", ID=str(tic_id))
        if len(table) == 0:
            return empty
        row = table[0]
        cols = table.colnames
        return {
            "teff": _safe_float(row["Teff"]) if "Teff" in cols else None,
            "rad":  _safe_float(row["rad"])  if "rad"  in cols else None,
            "logg": _safe_float(row["logg"]) if "logg" in cols else None,
            "tmag": _safe_float(row["Tmag"]) if "Tmag" in cols else None,
        }
    except Exception:
        return empty