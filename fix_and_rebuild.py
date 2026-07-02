"""
fix_and_rebuild.py — Rebuild ExoDetect dataset with fixed median_smooth

Fixes the RuntimeError: Unsupported array type caused by Astropy's
MaskedNDArray being passed to scipy.ndimage.median_filter.

Usage:
    # Quick test (10 samples):
    python fix_and_rebuild.py --n-samples 10

    # Full rebuild (3000 samples):
    python fix_and_rebuild.py --n-samples 3000

    # Custom:
    python fix_and_rebuild.py --n-samples 500 --data-dir ./data/kepler_lcs
"""

import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import lightkurve as lk
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
GLOBAL_VIEW_LEN = 201
LOCAL_VIEW_LEN  = 81


# ── Fixed preprocessing ───────────────────────────────────────────────────

def median_smooth(flux, window=51):
    """Running median filter to remove stellar variability.

    FIX: Cast to plain float64 first — Astropy MaskedNDArray
    with big-endian dtype (>f4) is rejected by scipy.ndimage.median_filter.
    """
    flux = np.asarray(flux, dtype=float)  # ← THE FIX
    return flux / median_filter(flux, size=window, mode='reflect')


def phase_fold_and_bin(time, flux, period, t0, n_bins, local=False, local_width=0.2):
    """Phase-fold a light curve and downsample into n_bins."""
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)

    if local:
        mask  = np.abs(phase) < local_width
        phase = phase[mask]
        flux  = flux[mask]
        if len(phase) < 5:
            return None
        p_min, p_max = -local_width, local_width
    else:
        p_min, p_max = -0.5, 0.5

    bins = np.linspace(p_min, p_max, n_bins + 1)
    idx  = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    view = np.array([
        np.median(flux[idx == b]) if np.any(idx == b) else 1.0
        for b in range(n_bins)
    ])
    view = np.where(np.isfinite(view), view, 1.0)
    oot  = np.abs(np.linspace(p_min, p_max, n_bins)) > 0.05
    if oot.sum() > 3:
        mu  = np.median(view[oot])
        sig = np.std(view[oot]) + 1e-8
        view = (view - mu) / sig
    return view


def process_one_tce(row):
    """Download light curve for one Kepler TCE and return
    (global_view, local_view, stellar_features, label) or None on failure.
    """
    kid = int(row['kepid'])
    try:
        period  = float(row['tce_period'])
        t0      = float(row['tce_time0bk'])
        dur_hr  = float(row['tce_duration'])
        depth   = float(row['tce_depth'])
        snr     = float(row['tce_model_snr'])
        label   = int(row['label'])

        if period <= 0 or np.isnan(period):
            return None

        # Search MAST for this Kepler target
        sr = lk.search_lightcurve(f'KIC {kid}', mission='Kepler', author='Kepler', quarter=5)
        if len(sr) == 0:
            sr = lk.search_lightcurve(f'KIC {kid}', mission='Kepler', author='Kepler')
        if len(sr) == 0:
            return None

        lc = sr[0].download(flux_column='pdcsap_flux')
        lc = lc.remove_nans().remove_outliers(sigma=5)
        if len(lc) < 100:
            return None

        t = lc.time.value
        f = lc.flux.value
        f = median_smooth(f)  # ← Uses the fixed version

        gv = phase_fold_and_bin(t, f, period, t0, GLOBAL_VIEW_LEN, local=False)
        lv = phase_fold_and_bin(t, f, period, t0, LOCAL_VIEW_LEN,  local=True)

        if gv is None or lv is None:
            return None

        stellar = np.array([
            np.log1p(period),
            np.log1p(max(dur_hr, 0)),
            np.log1p(max(depth,  0)) / 10,
            np.clip(snr, 0, 100)     / 100,
        ], dtype=np.float32)

        return (
            gv.astype(np.float32),
            lv.astype(np.float32),
            stellar,
            label
        )
    except Exception as e:
        # FIX: Log the error instead of silently swallowing it
        print(f"  ⚠ Failed KIC {kid}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Rebuild ExoDetect dataset with fixed preprocessing")
    parser.add_argument("--n-samples", type=int, default=3000, help="Total samples to fetch (default: 3000)")
    parser.add_argument("--data-dir", type=str, default="./data/kepler_lcs", help="Data directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset.npz if present")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    cache_path = data_dir / 'dataset.npz'
    tce_path   = data_dir / 'dr25_tce.csv'

    if not tce_path.exists():
        print(f"❌ TCE catalog not found at {tce_path}")
        print("   Run Cell 3 of the notebook first to download the catalog.")
        sys.exit(1)

    # Check for existing cache
    if cache_path.exists():
        d = np.load(cache_path)
        n_cached = len(d.get('labels', []))
        if n_cached > 0 and not args.force:
            print(f"✅ Existing dataset.npz has {n_cached} samples. Use --force to overwrite.")
            sys.exit(0)
        elif n_cached == 0:
            print(f"🗑️  Deleting broken cache (0 samples)...")
            cache_path.unlink()
        else:
            print(f"🗑️  --force specified, overwriting {n_cached}-sample cache...")
            cache_path.unlink()

    # Load catalog
    tce = pd.read_csv(tce_path)
    print(f"📋 TCE catalog: {len(tce):,} labeled events")
    print(f"   Labels: {tce['av_training_set'].value_counts().to_dict()}")

    # Stratified sample
    n_samples = args.n_samples
    pc = tce[tce['label'] == 1].sample(min(n_samples // 2, tce['label'].sum()), random_state=42)
    fp = tce[tce['label'] == 0].sample(min(n_samples // 2, (tce['label'] == 0).sum()), random_state=42)
    sample = pd.concat([pc, fp]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n🔭 Fetching {len(sample)} light curves from MAST...")
    print(f"   (PC={len(pc)}, FP={len(fp)})")
    print()

    global_views, local_views, stellar_feats, labels = [], [], [], []
    failed = 0
    error_counts = {}

    for i, row in tqdm(sample.iterrows(), total=len(sample), desc="Downloading"):
        result = process_one_tce(row)
        if result is None:
            failed += 1
            continue
        gv, lv, sf, lab = result
        global_views.append(gv)
        local_views.append(lv)
        stellar_feats.append(sf)
        labels.append(lab)
        time.sleep(0.1)  # be polite to MAST

    if len(labels) == 0:
        print(f"\n❌ All {failed} samples failed! Something is still wrong.")
        sys.exit(1)

    global_views  = np.array(global_views,  dtype=np.float32)
    local_views   = np.array(local_views,   dtype=np.float32)
    stellar_feats = np.array(stellar_feats, dtype=np.float32)
    labels        = np.array(labels,        dtype=np.int64)

    np.savez(cache_path,
             global_views=global_views, local_views=local_views,
             stellar_feats=stellar_feats, labels=labels)

    print(f"\n✅ Dataset built: {len(labels)} samples ({failed} failed)")
    print(f"\nDataset shape:")
    print(f"  Global views  : {global_views.shape}")
    print(f"  Local views   : {local_views.shape}")
    print(f"  Stellar feats : {stellar_feats.shape}")
    print(f"  Labels        : {labels.shape}  (PC={labels.sum()}, FP={(labels==0).sum()})")
    print(f"\n💾 Saved to {cache_path}")


if __name__ == "__main__":
    main()
