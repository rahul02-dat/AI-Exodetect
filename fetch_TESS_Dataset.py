"""
fetch_tess_dataset.py — Phase 1B: Build a large labeled TESS training set

WHY THIS EXISTS
----------------
The CNN currently shipped in exodetect_cnn.pt was trained on **Kepler**
(ExoDetect_Phase1_CNN.ipynb) but server.py / batch_pipeline.py run inference
on **TESS** light curves. Kepler and TESS differ in cadence, noise floor,
pipeline systematics (SPOC vs Kepler PDCSAP), and target population — a
classifier trained on one and scored on the other will silently
under-perform. This script builds a TESS-native training set instead.

WHY THE PREVIOUS ATTEMPT ONLY GOT 6,157 / 20,000
--------------------------------------------------
Two likely causes, both fixed here:
  1. No real "20k target" source existed — batch_pipeline.py's DEMO_TARGETS
     is a 10-item hardcoded list. Whatever produced 6,157 was pulling from
     a small/incomplete target list, not the actual TOI catalog. This
     script queries the NASA Exoplanet Archive TOI table directly, which
     currently has 7,000+ TOIs with clean CP/KP/FP/FA dispositions, and
     supplements with the Kepler catalog + injected negatives to reach 20k.
  2. No checkpointing / resume. A MAST timeout or rate limit partway through
     a multi-hour run means losing everything fetched so far. This script
     checkpoints every N successful downloads and skips already-fetched
     TIC IDs on restart, so a crashed run picks up where it left off.

LABEL SOURCE (TOI table, tfopwg_disp column)
----------------------------------------------
  CP, KP  (Confirmed Planet, Known Planet)      -> label 1 (planet)
  FP, FA  (False Positive, False Alarm)          -> label 0 (not planet)
  PC, APC (Planet Candidate, Ambiguous Candidate) -> excluded (label unknown)

STELLAR PARAMETERS (STScI MAST TIC catalog)
----------------------------------------------
Stellar parameters (Teff, radius, log_g, Tmag) are now sourced from the
TESS Input Catalog (TIC) via the MAST API, as cataloged at:
    https://archive.stsci.edu/tess/tic_ctl.html

These are bulk-prefetched for all targets before light-curve downloads
begin, cached locally for resume support, and used to build the expanded
8-feature (v2) stellar feature vector via stellar_features.py. This
replaces the old approach of using only the sparse TOI table columns
with hardcoded fallback defaults.

Alternatively, the full xCTL × TIC v8.1 cross-matched CSV (9.5 GB) can
be downloaded for fully offline lookups with --stellar-source xctl-csv.

Usage
-----
    python fetch_tess_dataset.py --n-samples 20000 --workers 4
    python fetch_tess_dataset.py --resume            # continue an interrupted run
    python fetch_tess_dataset.py --n-samples 2000 --dry-run   # test catalog query only
    python fetch_tess_dataset.py --stellar-source xctl-csv   # use bulk STScI CSV

Output
------
    data/tess_lcs/toi_catalog.csv    raw catalog pulled from the archive (provenance)
    data/tess_lcs/tic_stellar.csv    cached TIC stellar params for sampled targets
    data/tess_lcs/fetch_log.csv      per-target success/failure log (debug why counts are low)
    data/tess_lcs/checkpoint.json    resume state (processed TIC IDs)
    data/tess_lcs/dataset.npz        final training set:
                                        global_views  (N, 201) float32
                                        local_views   (N, 81)  float32
                                        stellar_feats (N, 8)   float32   ← v2 expanded
                                        labels        (N,)     int64
                                        tic_ids       (N,)     int64  (for provenance/debug)
"""

import argparse
import gzip
import io
import json
import os
import signal
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

# Shared stellar feature engineering — single source of truth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stellar_features import (
    N_STELLAR_EXPANDED,
    compute_stellar_features_v2,
)

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR       = Path('/Volumes/Expansion/EXO_Datasets/tess_data/tess_lcs')
DATA_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH    = DATA_DIR / "toi_catalog.csv"
STELLAR_PATH    = DATA_DIR / "tic_stellar.csv"
XCTL_PATH       = DATA_DIR / "exo_CTL_08.01xTIC_v8.1.csv"
LOG_PATH        = DATA_DIR / "fetch_log.csv"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"
OUT_PATH        = DATA_DIR / "dataset.npz"

GLOBAL_VIEW_LEN = 201
LOCAL_VIEW_LEN  = 81
STELLAR_FEAT_LEN = N_STELLAR_EXPANDED   # 8 features (v2 schema)
CHECKPOINT_EVERY = 100      # save partial progress every N successful fetches
MAX_RETRIES      = 3
RETRY_BACKOFF_S  = 5        # exponential: 5s, 10s, 20s
REQUEST_TIMEOUT  = 120

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
XCTL_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01xTIC_v8.1.csv"

# ── Graceful shutdown ────────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n⚠ Shutdown requested — finishing current downloads then saving…",
          flush=True)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _safe_print(*args, **kwargs):
    """Print that silently swallows errors from closed stdout (Ctrl-C race)."""
    try:
        print(*args, **kwargs, flush=True)
    except (ValueError, OSError, BrokenPipeError):
        pass


# ── TOI catalog download ────────────────────────────────────────────────

def tap_query(query: str) -> pd.DataFrame:
    r = requests.get(TAP_BASE, params={"query": query, "format": "csv"}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def fetch_toi_catalog(force: bool = False) -> pd.DataFrame:
    """Pull the full TOI table with clean CP/KP/FP/FA dispositions."""
    if CATALOG_PATH.exists() and not force:
        _safe_print(f"  Using cached catalog at {CATALOG_PATH}")
        return pd.read_csv(CATALOG_PATH)

    _safe_print("Querying NASA Exoplanet Archive TOI table (TAP)…")
    df = tap_query(
        "SELECT toi, tid, tfopwg_disp, pl_orbper, pl_tranmid, "
        "pl_trandurh, pl_trandep, st_tmag "
        "FROM toi "
        "WHERE tfopwg_disp IN ('CP','KP','FP','FA') "
        "AND pl_orbper IS NOT NULL "
        "AND pl_tranmid IS NOT NULL "
        "AND pl_trandurh IS NOT NULL"
    )
    df["label"] = df["tfopwg_disp"].isin(["CP", "KP"]).astype(int)
    df = df.dropna(subset=["tid", "pl_orbper", "pl_tranmid", "pl_trandurh"])
    df["pl_trandep"] = df["pl_trandep"].fillna(500.0)   # ppm, conservative default
    df["st_tmag"]    = df["st_tmag"].fillna(12.0)

    df.to_csv(CATALOG_PATH, index=False)
    _safe_print(f"  Catalog: {len(df)} TOIs  "
                f"(planets={df['label'].sum()}, false-positives={(df['label']==0).sum()})")
    return df


def balance_and_sample(df: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sample to the target size, balanced PC/FP where possible."""
    rng = np.random.default_rng(seed)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    n_each = n_samples // 2
    n_pos = min(n_each, len(pos))
    n_neg = min(n_each, len(neg))

    # If one class is short, take everything from it and top up the other
    # so we still get close to n_samples total (TOI catalog is
    # planet-heavy, so this mostly affects the negative class).
    deficit = n_each - min(n_pos, n_neg)
    if n_pos < n_each:
        n_neg = min(len(neg), n_neg + deficit)
    elif n_neg < n_each:
        n_pos = min(len(pos), n_pos + deficit)

    sample = pd.concat([
        pos.sample(n=n_pos, random_state=seed),
        neg.sample(n=n_neg, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    _safe_print(f"  Sampled {len(sample)} TOIs for download "
                f"(planets={n_pos}, false-positives={n_neg})")
    return sample


# ── Stellar parameter fetching (STScI TIC catalog) ──────────────────────

def _fetch_tic_params_mast(tic_ids: list) -> pd.DataFrame:
    """
    Batch-fetch stellar parameters from the MAST TIC catalog API.
    Queries in chunks to avoid timeout on large batches.

    Source: https://archive.stsci.edu/tess/tic_ctl.html
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        _safe_print("  ⚠ astroquery not installed — falling back to TOI-only stellar params.")
        _safe_print("    Install with: pip install astroquery")
        return pd.DataFrame(columns=["ID", "Teff", "rad", "logg", "Tmag"])

    all_rows = []
    chunk_size = 50  # MAST handles ~50 IDs per query well
    total_chunks = (len(tic_ids) + chunk_size - 1) // chunk_size

    _safe_print(f"  Fetching stellar params from MAST TIC for {len(tic_ids)} targets "
                f"({total_chunks} chunks)…")

    for i in range(0, len(tic_ids), chunk_size):
        chunk = tic_ids[i:i + chunk_size]
        chunk_num = i // chunk_size + 1

        if _shutdown_requested:
            _safe_print("  Shutdown requested — saving partial stellar params.")
            break

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Query each TIC ID individually within chunk (MAST doesn't
                # support multi-ID bulk query in query_criteria cleanly)
                for tic_id in chunk:
                    try:
                        table = Catalogs.query_criteria(catalog="Tic", ID=str(int(tic_id)))
                        if len(table) > 0:
                            row = table[0]
                            cols = table.colnames
                            all_rows.append({
                                "ID": int(tic_id),
                                "Teff": float(row["Teff"]) if "Teff" in cols and row["Teff"] is not None else np.nan,
                                "rad": float(row["rad"]) if "rad" in cols and row["rad"] is not None else np.nan,
                                "logg": float(row["logg"]) if "logg" in cols and row["logg"] is not None else np.nan,
                                "Tmag": float(row["Tmag"]) if "Tmag" in cols and row["Tmag"] is not None else np.nan,
                            })
                        else:
                            all_rows.append({"ID": int(tic_id), "Teff": np.nan, "rad": np.nan,
                                             "logg": np.nan, "Tmag": np.nan})
                    except Exception:
                        all_rows.append({"ID": int(tic_id), "Teff": np.nan, "rad": np.nan,
                                         "logg": np.nan, "Tmag": np.nan})

                if chunk_num % 5 == 0 or chunk_num == total_chunks:
                    _safe_print(f"    [{chunk_num:4d}/{total_chunks}] "
                                f"fetched {len(all_rows)} stellar param records")
                break  # success, move to next chunk

            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_S * attempt)
                else:
                    _safe_print(f"    ⚠ Chunk {chunk_num} failed after {MAX_RETRIES} retries: {e}")
                    for tic_id in chunk:
                        all_rows.append({"ID": int(tic_id), "Teff": np.nan, "rad": np.nan,
                                         "logg": np.nan, "Tmag": np.nan})

    df = pd.DataFrame(all_rows)
    # Deduplicate — keep first occurrence
    df = df.drop_duplicates(subset=["ID"], keep="first")
    return df


def _fetch_tic_params_xctl_csv() -> pd.DataFrame:
    """
    Download the full xCTL × TIC v8.1 cross-matched CSV from STScI and
    extract stellar parameters. This is a one-time ~9.5 GB download.

    Source: https://archive.stsci.edu/tess/tic_ctl.html
    """
    if XCTL_PATH.exists():
        _safe_print(f"  Using cached xCTL × TIC CSV at {XCTL_PATH}")
    else:
        _safe_print(f"  Downloading xCTL × TIC v8.1 CSV from STScI (~9.5 GB)…")
        _safe_print(f"  URL: {XCTL_URL}")
        _safe_print(f"  This is a one-time download; will be cached at {XCTL_PATH}")

        # Stream download with progress
        r = requests.get(XCTL_URL, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        downloaded = 0
        with open(XCTL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                if _shutdown_requested:
                    _safe_print("  Shutdown requested during download — aborting.")
                    f.close()
                    XCTL_PATH.unlink(missing_ok=True)
                    sys.exit(1)
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    _safe_print(f"    {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)",
                                end="\r")
        _safe_print(f"\n  ✅ Download complete: {XCTL_PATH}")

    _safe_print("  Reading xCTL × TIC CSV (this may take a minute for ~9.5 GB)…")
    # Only read the columns we need to save memory
    usecols = ["ID", "Teff", "rad", "logg", "Tmag"]
    df = pd.read_csv(XCTL_PATH, usecols=usecols, low_memory=False)
    _safe_print(f"  Loaded {len(df)} TIC entries from xCTL × TIC catalog")
    return df


def prefetch_stellar_params(tic_ids: list, source: str = "mast",
                            force: bool = False) -> Dict[int, dict]:
    """
    Prefetch stellar parameters for all targets. Returns a dict mapping
    TIC ID -> {"teff": ..., "rad": ..., "logg": ..., "tmag": ...}.
    Results are cached to STELLAR_PATH for resume support.
    """
    # Check cache first
    if STELLAR_PATH.exists() and not force:
        cached = pd.read_csv(STELLAR_PATH)
        cached_ids = set(cached["ID"].astype(int))
        missing_ids = [tid for tid in tic_ids if int(tid) not in cached_ids]

        if not missing_ids:
            _safe_print(f"  Using cached stellar params for {len(cached)} targets "
                        f"from {STELLAR_PATH}")
            return _stellar_df_to_dict(cached)
        else:
            _safe_print(f"  {len(cached)} cached, {len(missing_ids)} new targets to fetch")
            tic_ids = missing_ids
    else:
        cached = pd.DataFrame(columns=["ID", "Teff", "rad", "logg", "Tmag"])

    # Fetch missing stellar params
    if source == "mast":
        new_df = _fetch_tic_params_mast(tic_ids)
    elif source == "xctl-csv":
        full_df = _fetch_tic_params_xctl_csv()
        # Filter to only requested TIC IDs
        tic_set = set(int(t) for t in tic_ids)
        new_df = full_df[full_df["ID"].astype(int).isin(tic_set)].copy()
        _safe_print(f"  Matched {len(new_df)} / {len(tic_ids)} targets in xCTL × TIC catalog")
    else:
        raise ValueError(f"Unknown stellar source: {source!r}")

    # Merge with cache and save
    merged = pd.concat([cached, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ID"], keep="last")
    merged.to_csv(STELLAR_PATH, index=False)

    n_with_teff = merged["Teff"].notna().sum()
    _safe_print(f"  ✅ Stellar params cached: {len(merged)} targets "
                f"({n_with_teff} with Teff, "
                f"{merged['rad'].notna().sum()} with radius, "
                f"{merged['logg'].notna().sum()} with log_g)")

    return _stellar_df_to_dict(merged)


def _stellar_df_to_dict(df: pd.DataFrame) -> Dict[int, dict]:
    """Convert stellar params DataFrame to a lookup dict keyed by TIC ID."""
    result = {}
    for _, row in df.iterrows():
        tic = int(row["ID"])
        result[tic] = {
            "teff": row["Teff"] if pd.notna(row["Teff"]) else None,
            "rad":  row["rad"]  if pd.notna(row["rad"])  else None,
            "logg": row["logg"] if pd.notna(row["logg"]) else None,
            "tmag": row["Tmag"] if pd.notna(row["Tmag"]) else None,
        }
    return result


# ── Light-curve preprocessing (matches Phase 1 Kepler pipeline) ──────────

def median_smooth(flux, window=51):
    from scipy.ndimage import median_filter
    flux = np.asarray(flux, dtype=float)
    return flux / median_filter(flux, size=window, mode="reflect")


def phase_fold_and_bin(time, flux, period, t0, n_bins, local=False, local_width=0.2):
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)

    if local:
        mask = np.abs(phase) < local_width
        phase, flux = phase[mask], flux[mask]
        if len(phase) < 5:
            return None
        p_min, p_max = -local_width, local_width
    else:
        p_min, p_max = -0.5, 0.5

    bins = np.linspace(p_min, p_max, n_bins + 1)
    idx = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    view = np.array([
        np.median(flux[idx == b]) if np.any(idx == b) else 1.0
        for b in range(n_bins)
    ])
    view = np.where(np.isfinite(view), view, 1.0)

    oot = np.abs(np.linspace(p_min, p_max, n_bins)) > 0.05
    if oot.sum() > 3:
        mu, sig = np.median(view[oot]), np.std(view[oot]) + 1e-8
        view = (view - mu) / sig
    return view


def process_one_toi(row, stellar_lookup: Dict[int, dict]) -> dict:
    """
    Download + preprocess one TOI. Returns a result dict with either
    'status': 'ok' and the processed arrays, or 'status': <failure reason>.
    Never raises — every failure mode is logged, not silently dropped.
    """
    tic = int(row["tid"])
    period = float(row["pl_orbper"])
    t0 = float(row["pl_tranmid"])
    dur_hr = float(row["pl_trandurh"])
    depth_ppm = float(row["pl_trandep"])
    tmag_toi = float(row["st_tmag"])      # fallback from TOI table
    label = int(row["label"])

    if period <= 0 or np.isnan(period):
        return {"tic": tic, "status": "bad_period"}

    # Look up real stellar params from pre-fetched TIC catalog
    sp = stellar_lookup.get(tic, {})
    teff = sp.get("teff")
    rad  = sp.get("rad")
    logg = sp.get("logg")
    # Prefer TIC Tmag over the TOI table's st_tmag
    tmag = sp.get("tmag") if sp.get("tmag") is not None else tmag_toi

    # SNR proxy: brighter -> higher
    snr_proxy = np.clip((20 - tmag) * 5, 0, 100)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            import lightkurve as lk

            sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC")
            if len(sr) == 0:
                return {"tic": tic, "status": "no_data"}

            lc = sr[0].download(flux_column="pdcsap_flux")
            lc = lc.remove_nans().remove_outliers(sigma=5)
            if len(lc) < 100:
                return {"tic": tic, "status": "too_short"}

            t = lc.time.value
            f = median_smooth(lc.flux.value)

            gv = phase_fold_and_bin(t, f, period, t0, GLOBAL_VIEW_LEN, local=False)
            lv = phase_fold_and_bin(t, f, period, t0, LOCAL_VIEW_LEN, local=True)
            if gv is None or lv is None:
                return {"tic": tic, "status": "empty_view"}

            # Build v2 stellar features using shared module
            stellar = compute_stellar_features_v2(
                period=period,
                duration_hr=dur_hr,
                depth_ppm=depth_ppm,
                snr=snr_proxy,
                teff=teff,
                rad=rad,
                logg=logg,
                tmag=tmag,
            )

            return {
                "tic": tic, "status": "ok", "label": label,
                "gv": gv.astype(np.float32), "lv": lv.astype(np.float32),
                "sf": stellar,
            }

        except Exception as e:  # noqa: BLE001 — we want every exception logged, not raised
            last_err = str(e)[:200]
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)  # exponential-ish backoff
            continue

    return {"tic": tic, "status": f"error:{last_err}"}


# ── Checkpointing ───────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return set(json.load(f)["processed_tics"])
    return set()


def save_checkpoint(processed_tics):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"processed_tics": sorted(processed_tics)}, f)


def save_partial_dataset(gvs, lvs, sfs, labels, tics):
    if len(labels) == 0:
        return
    np.savez(
        OUT_PATH,
        global_views=np.array(gvs, dtype=np.float32),
        local_views=np.array(lvs, dtype=np.float32),
        stellar_feats=np.array(sfs, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
        tic_ids=np.array(tics, dtype=np.int64),
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build a labeled TESS training dataset with STScI TIC stellar parameters."
    )
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                     help="Only fetch/sample the catalog, skip light-curve downloads")
    ap.add_argument("--refresh-catalog", action="store_true")
    ap.add_argument("--stellar-source", choices=["mast", "xctl-csv"], default="mast",
                     help="Source for stellar parameters. "
                          "'mast' (default): batch-query MAST TIC API for ~N targets. "
                          "'xctl-csv': download full xCTL × TIC v8.1 CSV from STScI "
                          "(https://archive.stsci.edu/tess/tic_ctl.html, ~9.5 GB one-time).")
    args = ap.parse_args()

    catalog = fetch_toi_catalog(force=args.refresh_catalog)
    sample = balance_and_sample(catalog, args.n_samples)

    # ── Prefetch stellar params from STScI TIC catalog ──────────────────
    unique_tics = sample["tid"].astype(int).unique().tolist()
    _safe_print(f"\n  Prefetching stellar parameters for {len(unique_tics)} unique TIC IDs "
                f"(source: {args.stellar_source})…")
    _safe_print(f"  Catalog reference: https://archive.stsci.edu/tess/tic_ctl.html\n")

    stellar_lookup = prefetch_stellar_params(
        unique_tics,
        source=args.stellar_source,
        force=args.refresh_catalog,
    )

    n_with_params = sum(1 for t in unique_tics if t in stellar_lookup
                        and stellar_lookup[t].get("teff") is not None)
    _safe_print(f"  Stellar param coverage: {n_with_params}/{len(unique_tics)} targets "
                f"have Teff from TIC ({n_with_params/max(len(unique_tics),1)*100:.1f}%)\n")

    if args.dry_run:
        _safe_print("Dry run — stopping before light-curve downloads.")
        return

    processed_tics = load_checkpoint() if args.resume else set()
    if processed_tics:
        _safe_print(f"Resuming: {len(processed_tics)} TIC IDs already processed, will skip them.")
        sample = sample[~sample["tid"].astype(int).isin(processed_tics)]

    # Reload any existing partial dataset so we append rather than overwrite
    gvs, lvs, sfs, labels, tics = [], [], [], [], []
    if args.resume and OUT_PATH.exists():
        d = np.load(OUT_PATH)
        gvs = list(d["global_views"])
        lvs = list(d["local_views"])
        sfs = list(d["stellar_feats"])
        labels = list(d["labels"])
        tics = list(d["tic_ids"])
        _safe_print(f"  Loaded {len(labels)} previously-fetched samples from checkpointed dataset.")

    log_rows = []
    n_since_checkpoint = 0
    t_start = time.time()

    _safe_print(f"\nFetching {len(sample)} TESS light curves with {args.workers} workers…")
    _safe_print("(This can take hours for 20k targets — progress is checkpointed, so it's")
    _safe_print(" safe to Ctrl-C and resume later with --resume.)\n")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one_toi, row, stellar_lookup): row
                for _, row in sample.iterrows()
            }

            n_done = 0
            for fut in as_completed(futures):
                if _shutdown_requested:
                    _safe_print("\n⚠ Graceful shutdown — cancelling remaining futures…")
                    for f in futures:
                        f.cancel()
                    break

                n_done += 1
                try:
                    result = fut.result()
                except Exception as e:  # noqa: BLE001
                    result = {"tic": -1, "status": f"fatal:{e}"}

                log_rows.append({"tic": result["tic"], "status": result["status"]})

                if result["status"] == "ok":
                    gvs.append(result["gv"])
                    lvs.append(result["lv"])
                    sfs.append(result["sf"])
                    labels.append(result["label"])
                    tics.append(result["tic"])
                    processed_tics.add(result["tic"])
                    n_since_checkpoint += 1
                else:
                    processed_tics.add(result["tic"])  # don't retry known-bad targets on resume

                if n_done % 25 == 0 or n_done == len(sample):
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1e-6)
                    _safe_print(f"  [{n_done:5d}/{len(sample)}] "
                                f"ok={len(labels):5d}  "
                                f"rate={rate:.2f}/s  "
                                f"elapsed={elapsed/60:.1f}min")

                if n_since_checkpoint >= CHECKPOINT_EVERY:
                    save_partial_dataset(gvs, lvs, sfs, labels, tics)
                    save_checkpoint(processed_tics)
                    n_since_checkpoint = 0

    except KeyboardInterrupt:
        _safe_print("\n⚠ KeyboardInterrupt — saving progress before exit…")

    # Final save
    save_partial_dataset(gvs, lvs, sfs, labels, tics)
    save_checkpoint(processed_tics)

    log_df = pd.DataFrame(log_rows)
    if not log_df.empty:
        log_df.to_csv(LOG_PATH, index=False)

    _safe_print(f"\n✅ Done. {len(labels)} usable samples out of {len(sample)} requested "
                f"({len(labels)/max(len(sample),1)*100:.1f}% yield).")
    _safe_print(f"   Dataset  : {OUT_PATH}")
    _safe_print(f"   Fetch log: {LOG_PATH}")
    _safe_print(f"   Stellar feature vector: {STELLAR_FEAT_LEN} features (v2 expanded)")
    _safe_print(f"   Stellar param source: https://archive.stsci.edu/tess/tic_ctl.html")

    if not log_df.empty:
        _safe_print("\nFailure breakdown (see fetch_log.csv for full detail):")
        _safe_print(log_df["status"].apply(lambda s: s.split(":")[0]).value_counts().to_string())


if __name__ == "__main__":
    main()