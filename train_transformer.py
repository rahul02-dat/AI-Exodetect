"""
train_transformer.py — Phase 4: Train TransitFormer on the large-scale TESS dataset

Consumes the dataset.npz built by fetch_tess_dataset.py and trains the
TransitFormer classifier from transit_transformer.py.

Note: fetch_tess_dataset.py now writes its checkpointed dataset.npz
directly to its own DATA_DIR (no intermediate shard_*.npz files), so
--data below defaults to that same path. merge_tess_shards.py is only
relevant if you're merging shard_*.npz files from an older pipeline
version — it's not part of this workflow anymore.

What changed from the Phase 1 version, and why:

  - CLI args instead of hardcoded constants (DATA_PATH, EPOCHS, ...), so
    repeated experiments at 20k+ scale don't require editing the file
    every time.

  - Optional stratified k-fold cross-validation (--kfold N). A held-out
    test set is carved out *before* any folds are made, so test
    performance is never contaminated by fold/model selection — this
    matters more as the dataset grows, because a single lucky split can
    otherwise look like a real improvement. Reports both per-fold test
    AUC and an ensembled (fold-averaged) test AUC; averaging several
    models' predictions is a standard, effective way to raise both
    accuracy and calibration quality ("higher confidence" isn't just a
    bigger number — it has to be trustworthy).

  - Calibration reporting: Expected Calibration Error (ECE), Brier score,
    and a reliability diagram, computed on the held-out test set. ROC-AUC
    alone doesn't tell you whether a model's stated "94% confidence"
    actually corresponds to being right 94% of the time — this makes that
    checkable rather than assumed.

  - Every run's config + fold results + calibration numbers are logged to
    a timestamped JSON under data/tess/training_runs/, so successive
    experiments (more data, different hyperparameters) are comparable.

  - n_stellar is now auto-detected from the dataset (sf.shape[1]) rather
    than hardcoded to 4. fetch_tess_dataset.py writes the expanded
    8-feature vector by default (transit params + normalized Teff/
    radius/log g/Tmag from the TIC catalog), so this script picks that up
    automatically and saves it into the checkpoint's model_config, which
    is what server.py / batch_pipeline.py read to build matching stellar
    tensors at inference time (see stellar_features.py). A CNN and
    TransitFormer on different schema versions can still be ensembled
    together during a gradual rollout.

Usage:
    # Single 70/15/15 split — fast iteration while the dataset is still small
    python train_transformer.py

    # 5-fold CV + ensemble evaluation — do this once you have several
    # thousand samples, before treating the model as production-ready
    python train_transformer.py --kfold 5 --epochs 30

    # Point at a specific dataset snapshot
    python train_transformer.py --data /path/to/dataset.npz
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: this script is meant to run unattended
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from transit_transformer import build_transformer, get_device

warnings.filterwarnings("ignore")

GLOBAL_LEN = 201
PATCH_SIZE = 3


# ── Dataset ──────────────────────────────────────────────────────────────

class TransitDataset(Dataset):
    """Re-uses the merged dataset.npz — only needs global_views + stellar_feats."""

    def __init__(self, gv, sf, y, augment=False):
        self.gv = torch.tensor(gv, dtype=torch.float32)
        self.sf = torch.tensor(sf, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        gv, sf, y = self.gv[idx], self.sf[idx], self.y[idx]
        if self.augment:
            gv = gv + torch.randn_like(gv) * 0.01              # gaussian noise
            gv = gv * (1 + 0.02 * torch.randn(1).item())        # amplitude jitter
        return gv.unsqueeze(0), sf, y   # (1, L), (4,) or (8,), scalar


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at {path}\n"
            "   Run fetch_tess_dataset.py, or "
            "pass --data pointing at your dataset.npz."
        )
    d = np.load(path)
    gv = d["global_views"].astype(np.float32)
    sf = d["stellar_feats"].astype(np.float32)
    y = d["labels"].astype(np.int64)
    n_stellar = sf.shape[1]
    schema = "legacy (transit-only)" if n_stellar == 4 else \
             "expanded (+ Teff/radius/log g/Tmag)" if n_stellar == 8 else "custom"
    print(f"Loaded dataset: {len(y)} samples "
          f"({int((y == 1).sum())} positive / {int((y == 0).sum())} negative)")
    print(f"Stellar feature dimensionality: {n_stellar}  [{schema}]")
    return gv, sf, y


def make_loaders(gv, sf, y, idx_trn, idx_val, batch_size):
    ds_trn = TransitDataset(gv[idx_trn], sf[idx_trn], y[idx_trn], augment=True)
    ds_val = TransitDataset(gv[idx_val], sf[idx_val], y[idx_val])

    counts = np.bincount(y[idx_trn])
    weights = 1.0 / counts[y[idx_trn]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dl_trn = DataLoader(ds_trn, batch_size=batch_size, sampler=sampler, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    return dl_trn, dl_val


# ── Training ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def mixup_data(gv, sf, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = gv.size(0)
    index = torch.randperm(batch_size).to(gv.device)
    
    mixed_gv = lam * gv + (1 - lam) * gv[index, :]
    mixed_sf = lam * sf + (1 - lam) * sf[index, :]
    y_a, y_b = y, y[index]
    return mixed_gv, mixed_sf, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_model(gv, sf, y, idx_trn, idx_val, args, device, n_stellar, tag="fold0"):
    """Trains one TransitFormer to convergence with early stopping.
    Returns (model_with_best_weights, history, best_val_auc, best_epoch)."""
    dl_trn, dl_val = make_loaders(gv, sf, y, idx_trn, idx_val, args.batch_size)

    model = build_transformer(seq_len=GLOBAL_LEN, patch_size=PATCH_SIZE,
                               n_stellar=n_stellar, device=device)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"trn_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_epoch, no_improve = 0.0, 0, 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        trn_loss = 0.0
        for gv_b, sf_b, y_b in dl_trn:
            gv_b, sf_b, y_b = gv_b.to(device), sf_b.to(device), y_b.to(device)
            gv_b, sf_b, y_a, y_b, lam = mixup_data(gv_b, sf_b, y_b, alpha=0.2)
            
            optimizer.zero_grad()
            logits, _ = model(gv_b, sf_b, return_attn=False)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item()
        trn_loss /= len(dl_trn)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        probs_all, labels_all = [], []
        with torch.no_grad():
            for gv_b, sf_b, y_b in dl_val:
                gv_b, sf_b, y_b = gv_b.to(device), sf_b.to(device), y_b.to(device)
                logits, _ = model(gv_b, sf_b)
                val_loss += criterion(logits, y_b).item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_all.extend(probs.cpu().numpy())
                labels_all.extend(y_b.cpu().numpy())
        val_loss /= len(dl_val)
        val_auc = roc_auc_score(labels_all, probs_all)

        history["trn_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc, best_epoch, no_improve = val_auc, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        marker = " ◀ best" if epoch == best_epoch else ""
        print(f"[{tag}] Epoch {epoch:3d}/{args.epochs}  trn={trn_loss:.4f}  "
              f"val={val_loss:.4f}  AUC={val_auc:.4f}{marker}")

        if no_improve >= args.patience:
            print(f"[{tag}] Early stopping at epoch {epoch} "
                  f"(no val AUC improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    return model, history, best_auc, best_epoch


@torch.no_grad()
def predict_probs(model, gv, sf, idx, device, batch_size=128):
    model.eval()
    probs = []
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        gv_b = torch.tensor(gv[chunk], dtype=torch.float32).unsqueeze(1).to(device)
        sf_b = torch.tensor(sf[chunk], dtype=torch.float32).to(device)
        logits, _ = model(gv_b, sf_b)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ── Calibration ──────────────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=10):
    """
    ECE: weighted average gap between predicted confidence and actual
    accuracy, bucketed into n_bins. A well-calibrated model that says
    "90% confident" should be right ~90% of the time within that bucket.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(mask.sum())
        if n == 0:
            bin_stats.append({"lo": lo, "hi": hi, "count": 0, "acc": None, "conf": None})
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (n / len(probs)) * abs(acc - conf)
        bin_stats.append({"lo": lo, "hi": hi, "count": n, "acc": acc, "conf": conf})
    return float(ece), bin_stats


def plot_reliability_diagram(probs, labels, out_path, title="Reliability Diagram"):
    ece, bin_stats = expected_calibration_error(probs, labels)
    confs  = [b["conf"] for b in bin_stats if b["count"] > 0]
    accs   = [b["acc"] for b in bin_stats if b["count"] > 0]
    counts = [b["count"] for b in bin_stats if b["count"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfectly calibrated")
    ax1.plot(confs, accs, marker="o", color="#7C3AED", label="Model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_title(f"{title}\n(ECE = {ece:.4f})")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(confs, counts, width=0.08, color="#00D4FF", alpha=0.8)
    ax2.set_xlabel("Mean predicted probability (bin)")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Prediction confidence distribution")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return ece


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TransitFormer on TESS data")
    parser.add_argument("--data", type=Path,
                         default=Path("/Volumes/Expansion/EXO_Datasets/tess_data/tess_lcs/dataset.npz"),
                         help="Path to dataset.npz produced by fetch_tess_dataset.py "
                              "(defaults to that script's own DATA_DIR)")
    parser.add_argument("--out", type=Path,
                         default=Path(__file__).parent / "exodetect_transformer.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--kfold", type=int, default=1,
                         help="1 = single 70/15/15 split (default, fast). "
                              ">1 = stratified k-fold CV over train+val, evaluated "
                              "against one fixed held-out test set.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    gv, sf, y = load_data(args.data)
    n_stellar = sf.shape[1]

    run_dir = Path(__file__).parent / "data" / "tess" / "training_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    idx_all = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Held-out test set: {len(idx_test)} samples "
          f"(never touched during training or fold selection)")

    fold_results = []
    fold_probs_on_test = []

    if args.kfold <= 1:
        # ── Single split ──────────────────────────────────────────────
        val_frac_of_trainval = 0.15 / (1 - args.test_size)
        idx_trn, idx_val = train_test_split(
            idx_trainval, test_size=val_frac_of_trainval,
            stratify=y[idx_trainval], random_state=args.seed,
        )
        model, history, best_val_auc, best_epoch = train_one_model(
            gv, sf, y, idx_trn, idx_val, args, device, n_stellar, tag="fold0"
        )
        test_probs = predict_probs(model, gv, sf, idx_test, device)
        test_auc = roc_auc_score(y[idx_test], test_probs)
        fold_results.append({"fold": 0, "best_val_auc": best_val_auc,
                              "best_epoch": best_epoch, "test_auc": float(test_auc)})
        fold_probs_on_test.append(test_probs)

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "seq_len": GLOBAL_LEN, "patch_size": PATCH_SIZE, "d_model": 64,
                "n_heads": 4, "n_layers": 4, "ff_dim": 256, "n_stellar": n_stellar,
                "dropout": 0.1, "n_classes": 2,
            },
            "metrics": {"best_val_auc": best_val_auc, "best_epoch": best_epoch,
                        "test_auc": float(test_auc)},
        }, args.out)
        print(f"\n✅ Model saved to {args.out}  "
              f"(val AUC {best_val_auc:.4f}, test AUC {test_auc:.4f})")

    else:
        # ── K-fold CV ─────────────────────────────────────────────────
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        best_overall_auc = -1.0
        best_overall_state = None
        best_overall_config = None

        for fold, (trn_rel, val_rel) in enumerate(skf.split(idx_trainval, y[idx_trainval])):
            idx_trn = idx_trainval[trn_rel]
            idx_val = idx_trainval[val_rel]
            model, history, best_val_auc, best_epoch = train_one_model(
                gv, sf, y, idx_trn, idx_val, args, device, n_stellar, tag=f"fold{fold}"
            )
            test_probs = predict_probs(model, gv, sf, idx_test, device)
            test_auc = roc_auc_score(y[idx_test], test_probs)
            fold_results.append({"fold": fold, "best_val_auc": best_val_auc,
                                  "best_epoch": best_epoch, "test_auc": float(test_auc)})
            fold_probs_on_test.append(test_probs)

            if best_val_auc > best_overall_auc:
                best_overall_auc = best_val_auc
                best_overall_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_overall_config = {
                    "seq_len": GLOBAL_LEN, "patch_size": PATCH_SIZE, "d_model": 64,
                    "n_heads": 4, "n_layers": 4, "ff_dim": 256, "n_stellar": n_stellar,
                    "dropout": 0.1, "n_classes": 2,
                }

        test_aucs = [r["test_auc"] for r in fold_results]
        print(f"\n=== {args.kfold}-Fold CV Results ===")
        for r in fold_results:
            print(f"  Fold {r['fold']}: val AUC={r['best_val_auc']:.4f}  "
                  f"test AUC={r['test_auc']:.4f}")
        print(f"  Mean test AUC : {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

        ensemble_probs = np.mean(fold_probs_on_test, axis=0)
        ensemble_auc = roc_auc_score(y[idx_test], ensemble_probs)
        print(f"  Ensemble (avg of {args.kfold} folds) test AUC: {ensemble_auc:.4f}")

        torch.save({
            "model_state_dict": best_overall_state,
            "model_config": best_overall_config,
            "metrics": {
                "best_val_auc": best_overall_auc,
                "mean_test_auc": float(np.mean(test_aucs)),
                "std_test_auc": float(np.std(test_aucs)),
                "ensemble_test_auc": float(ensemble_auc),
            },
        }, args.out)
        print(f"\n✅ Best single fold saved to {args.out} (val AUC {best_overall_auc:.4f})")
        print(f"   Note: the ENSEMBLE AUC ({ensemble_auc:.4f}) was higher than any single "
              f"fold — for the strongest production confidence, consider serving all "
              f"{args.kfold} fold checkpoints and averaging their predictions in "
              f"server.py rather than deploying just this single best fold.")

        test_probs = ensemble_probs  # calibration report below uses the ensemble

    # ── Final held-out test report + calibration ───────────────────────
    test_labels = y[idx_test]
    test_preds = (test_probs >= 0.5).astype(int)

    print("\n=== Held-Out Test Set Evaluation ===")
    print(f"ROC-AUC     : {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"Brier score : {brier_score_loss(test_labels, test_probs):.4f}  "
          f"(0 = perfect, lower is better calibrated)")
    print()
    print(classification_report(test_labels, test_preds,
                                 target_names=["False Positive", "Planet"]))

    cal_path = run_dir / f"calibration_{run_id}.png"
    ece = plot_reliability_diagram(test_probs, test_labels, cal_path,
                                    title="TransitFormer — Test Set Calibration")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Reliability diagram saved to {cal_path}")

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_log = {
        "run_id": run_id,
        "args": args_dict,
        "n_samples": len(y),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_test": len(idx_test),
        "fold_results": fold_results,
        "test_auc": float(roc_auc_score(test_labels, test_probs)),
        "brier_score": float(brier_score_loss(test_labels, test_probs)),
        "ece": ece,
        "timestamp": datetime.now().isoformat(),
    }
    log_path = run_dir / f"run_{run_id}.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"\nRun metadata logged to {log_path}")


if __name__ == "__main__":
    main()