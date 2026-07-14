"""
merge_tess_shards.py — Merge TESS shards into a single training-ready dataset

Reads every data/tess/shards/shard_*.npz produced by fetch_tess_dataset.py,
concatenates them, de-duplicates by TIC ID (in case a target was ever
reprocessed across runs), and writes data/tess/dataset.npz in exactly the
format train_transformer.py expects:

    global_views  (N, 201)   float32
    stellar_feats (N, 4 or 8) float32  -- see --n-stellar below
    labels        (N,)       int64
    tic_ids       (N,)       int64     -- kept for traceability only

Stellar feature schema (--n-stellar):
    4 (legacy)   [log_period, log_duration, log_depth/10, snr/100]
    8 (expanded, default) adds normalized [Teff, radius, log g, Tmag]
                 from the TESS Input Catalog, using stellar_features.py's
                 expand_v1_to_v2() to append them to each shard's
                 already-computed v1 vector. Missing stellar params
                 (common for the random-field negatives) are imputed to
                 neutral "Sun-like" defaults rather than dropped.

Run this any time you want to fold newly-downloaded shards into the
training set — it's cheap (a few seconds even at 20k+ samples) so there's
no harm re-running it after every fetch_tess_dataset.py session.

Usage:
    python merge_tess_shards.py                       # 8-feature schema (default)
    python merge_tess_shards.py --n-stellar 4          # legacy 4-feature schema
    python merge_tess_shards.py --shard-dir data/tess/shards --out data/tess/dataset.npz
"""

import argparse
from pathlib import Path

import numpy as np

from stellar_features import expand_v1_to_v2, N_STELLAR_LEGACY, N_STELLAR_EXPANDED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path,
                         default=Path(__file__).parent / "data" / "tess" / "shards")
    parser.add_argument("--out", type=Path,
                         default=Path(__file__).parent / "data" / "tess" / "dataset.npz")
    parser.add_argument("--n-stellar", type=int, default=N_STELLAR_EXPANDED,
                         choices=[N_STELLAR_LEGACY, N_STELLAR_EXPANDED],
                         help=f"{N_STELLAR_LEGACY} = legacy transit-only features, "
                              f"{N_STELLAR_EXPANDED} (default) = adds normalized "
                              f"Teff/radius/log g/Tmag")
    args = parser.parse_args()

    shard_paths = sorted(args.shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found in {args.shard_dir} — run fetch_tess_dataset.py first."
        )

    print(f"Found {len(shard_paths)} shards in {args.shard_dir}")

    gv_list, sf_list, y_list, tic_list = [], [], [], []
    seen_tics = set()
    n_dupes = 0
    n_bad_shards = 0
    n_stellar_missing = 0

    for p in shard_paths:
        try:
            d = np.load(p)
            gv, sf, y, tics = d["global_views"], d["stellar_feats"], d["labels"], d["tic_ids"]
            st_teff = d["st_teff"] if "st_teff" in d.files else None
            st_rad  = d["st_rad"]  if "st_rad"  in d.files else None
            st_logg = d["st_logg"] if "st_logg" in d.files else None
            st_tmag = d["st_tmag"] if "st_tmag" in d.files else None
        except Exception as e:
            print(f"  ⚠️  Skipping unreadable shard {p.name}: {e}")
            n_bad_shards += 1
            continue

        if args.n_stellar == N_STELLAR_EXPANDED and st_teff is None:
            print(f"  ⚠️  {p.name} has no raw stellar fields (older shard format?) — "
                  f"falling back to legacy features for its samples.")

        for i in range(len(tics)):
            tic = int(tics[i])
            if tic in seen_tics:
                n_dupes += 1
                continue
            seen_tics.add(tic)

            v1 = sf[i]   # always the 4-dim legacy vector, as written by the fetch script
            if args.n_stellar == N_STELLAR_EXPANDED and st_teff is not None:
                teff = float(st_teff[i]) if np.isfinite(st_teff[i]) else None
                rad  = float(st_rad[i])  if np.isfinite(st_rad[i])  else None
                logg = float(st_logg[i]) if np.isfinite(st_logg[i]) else None
                tmag = float(st_tmag[i]) if np.isfinite(st_tmag[i]) else None
                if teff is None and rad is None and logg is None and tmag is None:
                    n_stellar_missing += 1
                stellar_feat = expand_v1_to_v2(v1, teff, rad, logg, tmag)
            else:
                stellar_feat = v1

            gv_list.append(gv[i])
            sf_list.append(stellar_feat)
            y_list.append(y[i])
            tic_list.append(tic)

    if not gv_list:
        raise RuntimeError("No usable samples found across all shards — nothing to merge.")

    global_views  = np.stack(gv_list).astype(np.float32)
    stellar_feats = np.stack(sf_list).astype(np.float32)
    labels        = np.array(y_list, dtype=np.int64)
    tic_ids       = np.array(tic_list, dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, global_views=global_views, stellar_feats=stellar_feats,
             labels=labels, tic_ids=tic_ids)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print(f"\n✅ Merged dataset written to {args.out}")
    print(f"   Samples : {len(labels)}  ({n_pos} positive / {n_neg} negative, "
          f"{n_pos / max(len(labels), 1) * 100:.1f}% positive)")
    if n_dupes:
        print(f"   Skipped {n_dupes} duplicate TIC IDs across shards")
    if n_bad_shards:
        print(f"   ⚠️  Skipped {n_bad_shards} unreadable shard file(s) — consider re-fetching them")
    print(f"   Global view shape : {global_views.shape}")
    print(f"   Stellar feat shape: {stellar_feats.shape}  (n_stellar={args.n_stellar})")
    if args.n_stellar == N_STELLAR_EXPANDED and n_stellar_missing:
        pct = n_stellar_missing / max(len(labels), 1) * 100
        print(f"   Note: {n_stellar_missing} samples ({pct:.1f}%) had no stellar catalog "
              f"params at all — those 4 features were imputed to neutral defaults, which "
              f"is expected for most random-field negatives.")

    if len(labels) < 2000:
        print("\n   Note: still a fairly small dataset — keep fetch_tess_dataset.py running "
              "and re-run this merge periodically as more shards land.")


if __name__ == "__main__":
    main()