"""
merge_tess_shards.py — Merge TESS shards into a single training-ready dataset

Reads every data/tess/shards/shard_*.npz produced by fetch_tess_dataset.py,
concatenates them, de-duplicates by TIC ID (in case a target was ever
reprocessed across runs), and writes data/tess/dataset.npz in exactly the
format train_transformer.py expects:

    global_views  (N, 201)  float32
    stellar_feats (N, 4)    float32   -- log_period, log_duration, log_depth/10, snr/100
    labels        (N,)      int64
    tic_ids       (N,)      int64     -- kept for traceability only

Run this any time you want to fold newly-downloaded shards into the
training set — it's cheap (a few seconds even at 20k+ samples) so there's
no harm re-running it after every fetch_tess_dataset.py session.

Usage:
    python merge_tess_shards.py
    python merge_tess_shards.py --shard-dir data/tess/shards --out data/tess/dataset.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path,
                         default=Path(__file__).parent / "data" / "tess" / "shards")
    parser.add_argument("--out", type=Path,
                         default=Path(__file__).parent / "data" / "tess" / "dataset.npz")
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

    for p in shard_paths:
        try:
            d = np.load(p)
            gv, sf, y, tics = d["global_views"], d["stellar_feats"], d["labels"], d["tic_ids"]
        except Exception as e:
            print(f"  ⚠️  Skipping unreadable shard {p.name}: {e}")
            n_bad_shards += 1
            continue

        for i in range(len(tics)):
            tic = int(tics[i])
            if tic in seen_tics:
                n_dupes += 1
                continue
            seen_tics.add(tic)
            gv_list.append(gv[i])
            sf_list.append(sf[i])
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
    print(f"   Stellar feat shape: {stellar_feats.shape}")

    if len(labels) < 2000:
        print("\n   Note: still a fairly small dataset — keep fetch_tess_dataset.py running "
              "and re-run this merge periodically as more shards land.")


if __name__ == "__main__":
    main()