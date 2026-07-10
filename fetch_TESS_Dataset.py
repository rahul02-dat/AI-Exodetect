"""
fetch_tess_dataset.py — Phase 4: Large-scale TESS dataset acquisition

Builds a labeled dataset of 20,000+ TESS light curves from the official
TOI (TESS Objects of Interest) catalog on ExoFOP, replacing the Phase 1
synthetic-negative hack (period-shifted duplicates) with:

  - POSITIVES : TOIs dispositioned CP / KP / PC / APC (confirmed, known,
                or candidate planets), using the catalog's own period,
                epoch, duration and depth — no BLS guessing needed.
  - NEGATIVES : (a) TOIs dispositioned FP / FA (human-vetted false
                positives — eclipsing binaries, background blends, etc.),
                using the same catalog parameters, and (b) random TIC
                field stars with no TOI at all, where a BLS search is run
                to find the "best" period exactly as a real candidate
                would be vetted, so the model learns to reject noise/
                stellar-variability bumps rather than obviously flat curves.

Design goals for local (Mac) execution:
  - I/O-bound work (network + disk), so this uses a ThreadPoolExecutor
    rather than multiprocessing — avoids macOS fork/spawn issues with
    lightkurve's astropy-based objects, and is gentler on MAST's servers.
  - Fully resumable. Every target's status lives in manifest.csv, so
    Ctrl+C (or a crash) followed by rerunning the *same command* picks up
    exactly where it left off — nothing is re-downloaded.
  - Sharded on-disk storage (.npz shards of ~200 samples each), so memory
    use stays flat whether you're fetching 500 or 50,000 targets.
  - Small worker pool + defensive per-target try/except, so one bad
    target (missing data, transient MAST error, corrupt FITS) can never
    kill the whole run.

Setup:
    pip install lightkurve astropy pandas requests scipy numpy
    pip install astroquery      # optional, only needed for random-field negatives

Usage:
    # Smoke test first — cheap, fast, confirms your environment works
    python fetch_tess_dataset.py --target-total 200 --workers 4

    # The real run (expect several hours — see the runtime note at the
    # bottom of this file). Run it in tmux/screen, or with nohup, since
    # it's long-lived:
    nohup python fetch_tess_dataset.py --target-total 20000 --workers 6 &

    # Resume after an interruption — identical command, picks up where
    # it stopped:
    python fetch_tess_dataset.py --target-total 20000 --workers 6

Output:
    data/tess/toi_catalog.csv     — cached TOI table
    data/tess/manifest.csv        — per-target status (resumability)
    data/tess/shards/shard_*.npz  — global_views, local_views,
                                     stellar_feats, labels, tic_ids
    data/tess/fetch.log           — full run log
"""

import argparse
import logging
import signal
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────

#DATA_DIR      = Path(__file__).parent / "data" / "tess"
DATA_DIR = Path('/Volumes/Expansion/EXO_Datasets/tess_data/tess')
SHARD_DIR     = DATA_DIR / "shards"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
TOI_CSV_PATH  = DATA_DIR / "toi_catalog.csv"
TOI_URL       = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"

GLOBAL_VIEW_LEN = 201
LOCAL_VIEW_LEN  = 81
SHARD_SIZE      = 200           # samples per .npz shard
BTJD_OFFSET     = 2457000.0     # TESS light curve timestamps are BJD - 2457000

POSITIVE_DISPOSITIONS = {"CP", "KP", "PC", "APC"}
NEGATIVE_DISPOSITIONS = {"FP", "FA"}

TOI_COLUMN_CANDIDATES = {
    "tic":         ["TIC ID", "TIC"],
    "disposition": ["TFOPWG Disposition", "Disposition"],
    "period":      ["Period (days)", "Orbital Period (days)", "Period (Days)"],
    "epoch":       ["Epoch (BJD)", "Transit Epoch (BJD)", "Epoch (TBJD)"],
    "duration":    ["Duration (hours)", "Transit Duration (hours)", "Duration (Hours)"],
    "depth":       ["Depth (ppm)", "Transit Depth (ppm)", "Depth (Ppm)"],
    "teff":        ["Stellar Eff Temp (K)", "Stellar Effective Temperature (K)"],
    "srad":        ["Stellar Radius (R_Sun)", "Stellar Radius (R_sun)"],
    "slogg":       ["Stellar log(g) (cm/s^2)", "Stellar log(g) (cm/s2)"],
    "tmag":        ["TESS Mag", "TESS Magnitude"],
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "fetch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fetch_tess")

_stop_requested = threading.Event()


def _handle_sigint(signum, frame):
    log.warning("Interrupt received — finishing in-flight downloads, then checkpointing…")
    log.warning("(Press Ctrl+C again to force-quit without a final checkpoint.)")
    _stop_requested.set()


signal.signal(signal.SIGINT, _handle_sigint)


# ── TOI catalog ──────────────────────────────────────────────────────────

def fetch_toi_catalog(force: bool = False) -> pd.DataFrame:
    """Download (or load cached) the ExoFOP-TESS TOI table."""
    if TOI_CSV_PATH.exists() and not force:
        log.info(f"Loading cached TOI catalog from {TOI_CSV_PATH}")
        return pd.read_csv(TOI_CSV_PATH)

    log.info("Downloading TOI catalog from ExoFOP-TESS…")
    r = requests.get(TOI_URL, timeout=60)
    r.raise_for_status()
    TOI_CSV_PATH.write_bytes(r.content)
    df = pd.read_csv(TOI_CSV_PATH)
    log.info(f"  Downloaded {len(df)} TOI rows, {len(df.columns)} columns")
    return df


def resolve_toi_columns(df: pd.DataFrame) -> dict:
    """Map our canonical field names onto whatever the CSV actually calls them."""
    resolved = {}
    for key, candidates in TOI_COLUMN_CANDIDATES.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            log.warning(f"  Could not find a column for '{key}' (tried {candidates}) "
                        f"— related fields will be left blank.")
        resolved[key] = found
    if resolved["tic"] is None or resolved["disposition"] is None:
        raise RuntimeError(
            "TOI catalog is missing TIC ID and/or disposition columns — "
            "ExoFOP may have changed their export format. Inspect "
            f"{TOI_CSV_PATH} and update TOI_COLUMN_CANDIDATES accordingly."
        )
    return resolved


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def to_btjd(bjd_or_btjd: Optional[float]) -> Optional[float]:
    """
    Normalize a catalog epoch to BTJD (= BJD - 2457000), matching the units
    of lightkurve's `lc.time.value` for TESS. ExoFOP TOI epochs are usually
    given in full BJD (~2458xxx); some fields are already BTJD (~1xxx-3xxx).
    We detect which convention we were given by magnitude.
    """
    if bjd_or_btjd is None:
        return None
    return bjd_or_btjd - BTJD_OFFSET if bjd_or_btjd > 2_400_000 else bjd_or_btjd


def fetch_negative_pool(n_needed: int, exclude_tics: set, seed: int = 42) -> list:
    """
    Random "field star" TIC IDs with no TOI at all — used as negatives that
    the model must vet from scratch via BLS, rather than relying only on
    already-flagged false positives. Optional: skipped gracefully if
    astroquery isn't installed.
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        log.warning("astroquery not installed — skipping random-field negatives. "
                    "Run `pip install astroquery` to enable this. Falling back to "
                    "catalog false positives only.")
        return []

    log.info(f"Querying TIC catalog for up to {n_needed} random negative candidates…")
    rng = np.random.default_rng(seed)
    tic_ids: list = []
    attempts = 0
    while len(tic_ids) < n_needed and attempts < 40 and not _stop_requested.is_set():
        attempts += 1
        ra, dec = rng.uniform(0, 360), rng.uniform(-90, 90)
        try:
            table = Catalogs.query_region(f"{ra} {dec}", radius=1.5, catalog="TIC", Tmag=[6, 12])
        except Exception as e:
            log.debug(f"  TIC region query failed ({e}), retrying elsewhere…")
            continue
        for row in table:
            tic = int(row["ID"])
            if tic not in exclude_tics and tic not in tic_ids:
                tic_ids.append(tic)
        if attempts % 5 == 0:
            log.info(f"  Random negative pool: {len(tic_ids)}/{n_needed}")

    return tic_ids[:n_needed]


# ── Target specification & list building ────────────────────────────────

@dataclass
class TargetSpec:
    tic_id: int
    label: int                            # 1 = planet, 0 = false positive / negative
    disposition: str = ""
    period: Optional[float] = None
    t0: Optional[float] = None            # BTJD
    duration_hr: Optional[float] = None
    depth_ppm: Optional[float] = None
    st_teff: Optional[float] = None
    st_rad: Optional[float] = None
    st_logg: Optional[float] = None
    st_tmag: Optional[float] = None


def build_target_list(toi_df: pd.DataFrame, cols: dict, target_total: int,
                       positive_frac: float, include_random_negatives: bool,
                       seed: int = 42) -> list:
    tic_col, disp_col = cols["tic"], cols["disposition"]

    df = toi_df.dropna(subset=[tic_col]).copy()
    df[tic_col] = df[tic_col].astype(int)
    df = df.drop_duplicates(subset=[tic_col])          # one row per TIC

    pos_df = df[df[disp_col].isin(POSITIVE_DISPOSITIONS)]
    neg_df = df[df[disp_col].isin(NEGATIVE_DISPOSITIONS)]

    n_pos = min(int(target_total * positive_frac), len(pos_df))
    pos_sample = pos_df.sample(n=n_pos, random_state=seed)

    n_neg = target_total - n_pos
    n_neg_catalog = min(n_neg, len(neg_df))
    neg_catalog_sample = neg_df.sample(n=n_neg_catalog, random_state=seed)

    n_neg_random = n_neg - n_neg_catalog
    random_tic_ids = []
    if n_neg_random > 0 and include_random_negatives:
        exclude = set(df[tic_col].tolist())
        random_tic_ids = fetch_negative_pool(n_neg_random, exclude, seed=seed)
        if len(random_tic_ids) < n_neg_random:
            log.warning(f"  Only found {len(random_tic_ids)}/{n_neg_random} random "
                        f"negatives — final dataset will be smaller than requested "
                        f"unless you rerun with more negative headroom.")

    targets = []
    for _, row in pos_sample.iterrows():
        targets.append(TargetSpec(tic_id=int(row[tic_col]), label=1, disposition=str(row[disp_col])))
    for _, row in neg_catalog_sample.iterrows():
        targets.append(TargetSpec(tic_id=int(row[tic_col]), label=0, disposition=str(row[disp_col])))
    for tic in random_tic_ids:
        targets.append(TargetSpec(tic_id=int(tic), label=0, disposition="RANDOM_FIELD"))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(targets))
    targets = [targets[i] for i in order]

    log.info(f"Target list built: {n_pos} positive, {n_neg_catalog} catalog-FP negative, "
              f"{len(random_tic_ids)} random-field negative  (total {len(targets)})")
    return targets


def _row_to_spec(row: pd.Series, toi_lookup: Optional[pd.DataFrame], cols: dict) -> TargetSpec:
    """Rebuild a TargetSpec from a manifest row, re-attaching catalog fields."""
    tic_id = int(row["tic_id"])
    label = int(row["label"])
    disposition = row.get("disposition", "") or ""

    period = t0 = duration_hr = depth_ppm = None
    st_teff = st_rad = st_logg = st_tmag = None

    if toi_lookup is not None and tic_id in toi_lookup.index:
        toi_row = toi_lookup.loc[tic_id]
        if isinstance(toi_row, pd.DataFrame):        # multiple TOIs on one TIC
            toi_row = toi_row.iloc[0]

        def g(key):
            col = cols.get(key)
            return toi_row.get(col) if col else None

        period      = _safe_float(g("period"))
        t0          = to_btjd(_safe_float(g("epoch")))
        duration_hr = _safe_float(g("duration"))
        depth_ppm   = _safe_float(g("depth"))
        st_teff     = _safe_float(g("teff"))
        st_rad      = _safe_float(g("srad"))
        st_logg     = _safe_float(g("slogg"))
        st_tmag     = _safe_float(g("tmag"))

    return TargetSpec(
        tic_id=tic_id, label=label, disposition=disposition,
        period=period, t0=t0, duration_hr=duration_hr, depth_ppm=depth_ppm,
        st_teff=st_teff, st_rad=st_rad, st_logg=st_logg, st_tmag=st_tmag,
    )


# ── Light curve preprocessing (mirrors the Phase 1 notebook, plus stitching) ──

def median_smooth(flux, window: int = 51):
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


def stitch_sectors(lcs):
    """
    Concatenate multiple sectors into one light curve. Each sector is
    normalized to its own median first — SPOC PDCSAP flux is already
    roughly unity-centered per sector, but small baseline offsets between
    sectors would otherwise show up as spurious long-period signal.
    """
    times, fluxes = [], []
    for lc in lcs:
        t = np.asarray(lc.time.value, dtype=np.float64)
        f = np.asarray(lc.flux.value, dtype=np.float64)
        med = np.nanmedian(f)
        if med and np.isfinite(med):
            f = f / med
        times.append(t)
        fluxes.append(f)
    time_arr = np.concatenate(times)
    flux_arr = np.concatenate(fluxes)
    order = np.argsort(time_arr)
    return time_arr[order], flux_arr[order]


def process_target(spec: TargetSpec, max_sectors: int = 6) -> Optional[dict]:
    """
    Download all available SPOC 2-min sectors for one target, stitch them,
    resolve transit parameters (from the catalog if known, else via BLS),
    and produce global/local phase-folded views + stellar features.

    Returns None on any failure — the caller marks the target 'no_data' in
    the manifest and moves on. A single bad target must never kill the batch.
    """
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares

    try:
        sr = lk.search_lightcurve(f"TIC {spec.tic_id}", mission="TESS", author="SPOC")
        if len(sr) == 0:
            return None
        sr = sr[:max_sectors]

        lcs = []
        for s in sr:
            try:
                lc = s.download(flux_column="pdcsap_flux")
                lc = lc.remove_nans().remove_outliers(sigma=5)
                if len(lc) >= 50:
                    lcs.append(lc)
            except Exception:
                continue   # one bad sector shouldn't sink the whole target

        if not lcs:
            return None

        time_arr, flux_arr = stitch_sectors(lcs)
        if len(time_arr) < 200:
            return None

        period, t0 = spec.period, spec.t0
        duration_hr, depth_ppm = spec.duration_hr, spec.depth_ppm

        if period is None or t0 is None:
            # No catalog ephemeris (random field negative) — vet it exactly
            # like a real candidate would be: run BLS and fold on its
            # strongest periodic signal, whatever that turns out to be.
            bls = BoxLeastSquares(time_arr, flux_arr)
            periods = np.linspace(0.5, 20.0, 3000)
            power = bls.power(periods, np.linspace(0.05, 0.4, 12))
            bi = int(np.argmax(power.power))
            period = float(power.period[bi])
            t0 = float(power.transit_time[bi])
            duration_hr = float(power.duration[bi]) * 24
            depth_ppm = float(power.depth[bi]) * 1e6

        if not (period and period > 0):
            return None

        smoothed = median_smooth(flux_arr)
        gv = phase_fold_and_bin(time_arr, smoothed, period, t0, GLOBAL_VIEW_LEN, local=False)
        lv = phase_fold_and_bin(time_arr, smoothed, period, t0, LOCAL_VIEW_LEN, local=True)
        if gv is None or lv is None:
            return None

        in_tr = np.abs(((time_arr - t0) % period) - period / 2) < (duration_hr / 24) / 2
        noise = np.std(flux_arr[~in_tr]) if (~in_tr).sum() > 10 else 1e-4
        sig = abs(np.mean(flux_arr[in_tr]) - np.mean(flux_arr[~in_tr])) if in_tr.sum() > 0 else 0.0
        snr = float(sig / noise * np.sqrt(max(in_tr.sum(), 1)))

        stellar = np.array([
            np.log1p(period),
            np.log1p(max(duration_hr or 0, 0)),
            np.log1p(max(depth_ppm or 0, 0)) / 10,
            np.clip(snr, 0, 100) / 100,
        ], dtype=np.float32)

        return {
            "tic_id": spec.tic_id,
            "label": spec.label,
            "global_view": gv.astype(np.float32),
            "local_view": lv.astype(np.float32),
            "stellar_feat": stellar,
            "period": period,
            "n_sectors": len(lcs),
            "n_cadences": len(time_arr),
            "st_teff": spec.st_teff if spec.st_teff is not None else np.nan,
            "st_rad": spec.st_rad if spec.st_rad is not None else np.nan,
            "st_logg": spec.st_logg if spec.st_logg is not None else np.nan,
            "st_tmag": spec.st_tmag if spec.st_tmag is not None else np.nan,
            "disposition": spec.disposition,
        }
    except Exception as e:
        log.debug(f"TIC {spec.tic_id} failed: {e}")
        return None


# ── Manifest & sharded storage ───────────────────────────────────────────

MANIFEST_COLUMNS = ["tic_id", "label", "disposition", "status", "shard_idx"]


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame):
    df.to_csv(MANIFEST_PATH, index=False)


def _next_shard_idx() -> int:
    existing = sorted(SHARD_DIR.glob("shard_*.npz"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


class ShardWriter:
    """Buffers processed samples and flushes to a compressed .npz every SHARD_SIZE."""

    def __init__(self, start_shard_idx: int = 0):
        self.buffer = []
        self.shard_idx = start_shard_idx

    def add(self, sample: dict) -> int:
        self.buffer.append(sample)
        if len(self.buffer) >= SHARD_SIZE:
            self.flush()
        return self.shard_idx

    def flush(self):
        if not self.buffer:
            return
        path = SHARD_DIR / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            global_views  = np.stack([s["global_view"] for s in self.buffer]),
            local_views   = np.stack([s["local_view"] for s in self.buffer]),
            stellar_feats = np.stack([s["stellar_feat"] for s in self.buffer]),
            labels        = np.array([s["label"] for s in self.buffer], dtype=np.int64),
            tic_ids       = np.array([s["tic_id"] for s in self.buffer], dtype=np.int64),
            st_teff       = np.array([s["st_teff"] for s in self.buffer], dtype=np.float32),
            st_rad        = np.array([s["st_rad"] for s in self.buffer], dtype=np.float32),
            st_logg       = np.array([s["st_logg"] for s in self.buffer], dtype=np.float32),
            st_tmag       = np.array([s["st_tmag"] for s in self.buffer], dtype=np.float32),
        )
        log.info(f"  Wrote {path.name} ({len(self.buffer)} samples)")
        self.buffer = []
        self.shard_idx += 1


# ── Main ─────────────────────────────────────────────────────────────────

def print_dataset_summary(manifest: pd.DataFrame):
    done = manifest[manifest["status"] == "done"]
    n_pos = (done["label"] == 1).sum()
    n_neg = (done["label"] == 0).sum()
    n_shards = len(list(SHARD_DIR.glob("shard_*.npz")))
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Done      : {len(done)}  ({n_pos} positive / {n_neg} negative)")
    print(f"  Failed    : {(manifest['status'] == 'no_data').sum()}")
    print(f"  Pending   : {(manifest['status'] == 'pending').sum()}")
    print(f"  Shards    : {n_shards}  in {SHARD_DIR}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fetch a large-scale TESS transit dataset")
    parser.add_argument("--target-total", type=int, default=20000,
                         help="Number of targets to *attempt* (final dataset will be "
                              "somewhat smaller, since not every target has usable data)")
    parser.add_argument("--positive-frac", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6,
                         help="Thread pool size — 4-8 is a reasonable range for MAST")
    parser.add_argument("--max-sectors", type=int, default=6,
                         help="Cap sectors downloaded per target (keeps runtime bounded)")
    parser.add_argument("--refresh-catalog", action="store_true",
                         help="Re-download the TOI catalog even if a cached copy exists")
    parser.add_argument("--no-random-negatives", action="store_true",
                         help="Skip astroquery-based random field-star negatives")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                         help="Save manifest.csv after this many completions")
    args = parser.parse_args()

    toi_df = fetch_toi_catalog(force=args.refresh_catalog)
    cols = resolve_toi_columns(toi_df)

    manifest = load_manifest()

    if manifest.empty:
        targets = build_target_list(
            toi_df, cols, args.target_total, args.positive_frac,
            include_random_negatives=not args.no_random_negatives,
        )
        manifest = pd.DataFrame([{
            "tic_id": t.tic_id, "label": t.label, "disposition": t.disposition,
            "status": "pending", "shard_idx": -1,
        } for t in targets])
        save_manifest(manifest)
        log.info(f"Initialized manifest with {len(manifest)} targets")
    else:
        log.info(f"Resuming from existing manifest ({len(manifest)} targets)")

    print_dataset_summary(manifest)

    pending_mask = manifest["status"] == "pending"
    if not pending_mask.any():
        log.info("Nothing to do — all targets already processed. "
                 "Increase --target-total and rerun to fetch more.")
        return

    toi_lookup = toi_df.dropna(subset=[cols["tic"]]).copy()
    toi_lookup[cols["tic"]] = toi_lookup[cols["tic"]].astype(int)
    toi_lookup = toi_lookup.set_index(cols["tic"])

    pending_specs = [_row_to_spec(row, toi_lookup, cols)
                      for _, row in manifest[pending_mask].iterrows()]

    writer = ShardWriter(start_shard_idx=_next_shard_idx())
    lock = threading.Lock()

    def _worker(spec: TargetSpec):
        if _stop_requested.is_set():
            return spec.tic_id, "pending", None   # leave for next run
        result = process_target(spec, max_sectors=args.max_sectors)
        return spec.tic_id, ("done" if result is not None else "no_data"), result

    processed = 0
    t_start = time.time()
    log.info(f"Starting {args.workers} workers on {len(pending_specs)} pending targets…")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, s): s for s in pending_specs}
        for fut in as_completed(futures):
            tic_id, status, result = fut.result()
            with lock:
                manifest.loc[manifest["tic_id"] == tic_id, "status"] = status
                if result is not None:
                    writer.add(result)
                    manifest.loc[manifest["tic_id"] == tic_id, "shard_idx"] = writer.shard_idx
                processed += 1

                if processed % args.checkpoint_every == 0:
                    save_manifest(manifest)
                    elapsed = time.time() - t_start
                    rate = processed / elapsed
                    remaining = len(pending_specs) - processed
                    eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                    n_done_so_far = (manifest["status"] == "done").sum()
                    log.info(f"  [{processed}/{len(pending_specs)}] rate={rate:.2f}/s  "
                              f"ETA={eta_min:.1f} min  total_done={n_done_so_far}")

            if _stop_requested.is_set():
                break

    writer.flush()
    save_manifest(manifest)
    print_dataset_summary(manifest)

    if _stop_requested.is_set():
        log.info("Interrupted — rerun the exact same command to resume.")
    else:
        log.info("✅ Session complete.")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# RUNTIME NOTE
#
# Each target typically takes ~5-15s (a few seconds of MAST search + one
# download per sector). With 6 workers, expect roughly:
#     20,000 targets × ~8s / 6 workers  ≈  7-8 hours
#
# Recommendations for a local Mac run:
#   1. Smoke-test first: `--target-total 200` to confirm the pipeline
#      completes cleanly end-to-end before committing to a multi-hour run.
#   2. Run inside tmux/screen, or with `nohup … &`, since this is a
#      long-lived process you'll want to detach from.
#   3. It's safe to interrupt (Ctrl+C) and resume at any time — nothing
#      already downloaded is repeated.
#   4. Once you have enough shards, the next step is a small script that
#      concatenates shard_*.npz files into the train/val/test split format
#      train_transformer.py expects — that's a natural "training upgrade"
#      follow-up once this data acquisition phase is done.
# ─────────────────────────────────────────────────────────────────────────