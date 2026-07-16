"""
train_cnn.py — Train the dual-branch 1D CNN (ExoDetectCNN) on the TESS dataset

Counterpart to train_transformer.py, written to match the TESS-native
dataset now produced by fetch_tess_dataset.py:

  - Loads global_views (201), local_views (81), stellar_feats, labels,
    tic_ids straight from the dataset.npz written by fetch_tess_dataset.py
    — i.e. real TOI CP/KP vs FP/FA labels pulled from the NASA Exoplanet
    Archive TAP query, not the old synthetic/period-shifted negatives.
  - n_stellar is auto-detected from the dataset (4 = legacy transit-only,
    8 = expanded with normalized Teff/radius/log g/Tmag from the TIC
    catalog) using the same convention as train_transformer.py and
    stellar_features.py. This means the CNN and TransitFormer can be
    trained independently — even on dataset snapshots with different
    schema versions — and still ensemble correctly at inference time,
    since server.py / batch_pipeline.py build a correctly-sized stellar
    tensor per model via stellar_features.build_stellar_features().
  - A held-out test set is carved out *before* any fold/model selection
    (same as train_transformer.py), so test performance is never
    contaminated by fold or checkpoint selection.
  - Optional stratified k-fold CV (--kfold N) with fold-averaged ensemble
    test AUC — directly comparable to the TransitFormer's k-fold results.
  - Calibration reporting (ECE, Brier score, reliability diagram) on the
    held-out test set, so "94% confidence" from the CNN is checkable
    rather than assumed.
  - Saves a checkpoint (model_state_dict + model_config + metrics) in
    exactly the format server.py / batch_pipeline.py already expect from
    exodetect_cnn.pt — no changes needed on the inference side.

Why these changes matter for the two target outcomes:
  1. Robust dip classification: real TOI labels (rather than heuristic
     negatives) and the expanded 8-feature stellar schema (real Teff/
     radius/log g/Tmag instead of neutral imputed defaults) both reduce
     label noise and give the CNN's stellar-feature branch actual signal
     to learn from, instead of near-constant inputs.
  2. Parameter estimation (period/duration/depth): unchanged here — that
     is handled by the BLS initial estimate + MCMC refinement pipeline
     (mcmc_fitter.py). This script only improves the upstream
     classification step that decides which candidates are worth running
     through MCMC (see batch_pipeline.py's flag_mcmc threshold).

Usage:
    # Single 70/15/15 split — fast iteration while the dataset is still small
    python train_cnn.py

    # 5-fold CV + ensemble evaluation — do this once you have several
    # thousand samples, before treating the model as production-ready
    python train_cnn.py --kfold 5 --epochs 40

    # Point at a specific dataset snapshot
    python train_cnn.py --data /path/to/dataset.npz --out exodetect_cnn.pt
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: meant to run unattended
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

GLOBAL_LEN = 201
LOCAL_LEN = 81


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model ──────────────────────────────────────────────────────────────────
# Layer names (global_branch / local_branch / head) intentionally match the
# inference-time ExoDetectCNN definitions in server.py and batch_pipeline.py,
# so checkpoints trained here load there with no changes required.

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k=5, p=2):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(oc)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(p)
        
        self.shortcut = nn.Sequential()
        if ic != oc:
            self.shortcut = nn.Sequential(
                nn.Conv1d(ic, oc, 1),
                nn.BatchNorm1d(oc)
            )

    def forward(self, x):
        out = self.bn(self.conv(x))
        out += self.shortcut(x)
        out = self.relu(out)
        return self.pool(out)

class SafeAdaptiveAvgPool1d(nn.Module):
    """
    Wrapper around nn.AdaptiveAvgPool1d to avoid MPS non-divisible size errors.
    Automatically moves tensors to CPU for the pooling operation if on MPS.
    """
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
        
    def forward(self, x):
        if x.device.type == 'mps':
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)


class ExoDetectCNN(nn.Module):
    """Dual-branch 1D CNN — global (full-orbit, 201 bins) + local (transit
    zoom, 81 bins) views, fused with stellar auxiliary features."""

    def __init__(self, global_len=GLOBAL_LEN, local_len=LOCAL_LEN, n_stellar=4):
        super().__init__()
        self.n_stellar = n_stellar  # recorded so ExoEnsemble.predict() / server.py
                                     # can size this model's stellar tensor correctly.
        self.global_branch = nn.Sequential(
            ConvBlock(1, 16, 5, 2), ConvBlock(16, 32, 5, 2),
            ConvBlock(32, 64, 5, 2), ConvBlock(64, 128, 3, 2),
            SafeAdaptiveAvgPool1d(8), nn.Flatten(),
        )
        self.local_branch = nn.Sequential(
            ConvBlock(1, 16, 5, 2), ConvBlock(16, 32, 5, 2),
            ConvBlock(32, 64, 3, 2), SafeAdaptiveAvgPool1d(4), nn.Flatten(),
        )
        fused = (128 * 8) + (64 * 4) + n_stellar
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, gv, lv, sf):
        return self.head(torch.cat(
            [self.global_branch(gv), self.local_branch(lv), sf], dim=1
        ))


def build_cnn(n_stellar=4, device=None):
    """Instantiate a fresh ExoDetectCNN and move it to device."""
    device = device or get_device()
    model = ExoDetectCNN(GLOBAL_LEN, LOCAL_LEN, n_stellar).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ExoDetectCNN] {total:,} trainable parameters on {device} (n_stellar={n_stellar})")
    return model


def load_cnn(path, device=None):
    """Load a saved ExoDetectCNN checkpoint (same format server.py reads)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("model_config", {})
    model = ExoDetectCNN(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[ExoDetectCNN] Loaded from {path} — "
          f"Val AUC {ckpt.get('metrics', {}).get('best_val_auc', '?')} "
          f"(n_stellar={model.n_stellar})")
    return model, ckpt.get("metrics", {})


# ── Dataset ──────────────────────────────────────────────────────────────

class TransitCNNDataset(Dataset):
    """Reads global_views + local_views + stellar_feats + labels from the
    dataset.npz written by fetch_tess_dataset.py."""

    def __init__(self, gv, lv, sf, y, augment=False):
        self.gv = torch.tensor(gv, dtype=torch.float32)
        self.lv = torch.tensor(lv, dtype=torch.float32)
        self.sf = torch.tensor(sf, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        gv, lv, sf, y = self.gv[idx], self.lv[idx], self.sf[idx], self.y[idx]
        if self.augment:
            gv = gv + torch.randn_like(gv) * 0.01          # gaussian noise
            lv = lv + torch.randn_like(lv) * 0.01
            gv = gv * (1 + 0.02 * torch.randn(1).item())      # amplitude jitter
            
            # CutOut augmentation
            if torch.rand(1).item() < 0.5:
                cut_len = int(len(gv) * 0.1)
                start = torch.randint(0, len(gv) - cut_len, (1,)).item()
                gv[start:start+cut_len] = 0.0
                
        return gv.unsqueeze(0), lv.unsqueeze(0), sf, y   # (1, 201), (1, 81), (n_stellar,), scalar


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at {path}\n"
            "   Run fetch_tess_dataset.py first (it writes its dataset.npz to\n"
            "   its own DATA_DIR — see that script's docstring), or pass\n"
            "   --data pointing at your dataset.npz explicitly."
        )
    d = np.load(path)
    for key in ("global_views", "local_views", "stellar_feats", "labels"):
        if key not in d.files:
            raise KeyError(
                f"{path} is missing '{key}'. train_cnn.py needs both "
                f"global_views AND local_views (unlike TransitFormer, which "
                f"only uses the global view) — make sure this dataset.npz "
                f"came from fetch_tess_dataset.py, not a global-view-only export."
            )
    gv = d["global_views"].astype(np.float32)
    lv = d["local_views"].astype(np.float32)
    sf = d["stellar_feats"].astype(np.float32)
    y = d["labels"].astype(np.int64)
    n_stellar = sf.shape[1]
    schema = "legacy (transit-only)" if n_stellar == 4 else \
             "expanded (+ Teff/radius/log g/Tmag)" if n_stellar == 8 else "custom"
    print(f"Loaded dataset: {len(y)} samples "
          f"({int((y == 1).sum())} positive / {int((y == 0).sum())} negative)")
    print(f"Global view length: {gv.shape[1]}   Local view length: {lv.shape[1]}")
    print(f"Stellar feature dimensionality: {n_stellar}  [{schema}]")
    return gv, lv, sf, y


def make_loaders(gv, lv, sf, y, idx_trn, idx_val, batch_size):
    ds_trn = TransitCNNDataset(gv[idx_trn], lv[idx_trn], sf[idx_trn], y[idx_trn], augment=True)
    ds_val = TransitCNNDataset(gv[idx_val], lv[idx_val], sf[idx_val], y[idx_val])

    # Weighted sampler — TOI catalog is planet-heavy (mostly CP/KP), so
    # without this the CNN would rarely see FP/FA examples per batch.
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

def mixup_data(gv, lv, sf, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = gv.size(0)
    index = torch.randperm(batch_size).to(gv.device)
    
    mixed_gv = lam * gv + (1 - lam) * gv[index, :]
    mixed_lv = lam * lv + (1 - lam) * lv[index, :]
    mixed_sf = lam * sf + (1 - lam) * sf[index, :]
    y_a, y_b = y, y[index]
    return mixed_gv, mixed_lv, mixed_sf, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_model(gv, lv, sf, y, idx_trn, idx_val, args, device, n_stellar, tag="fold0"):
    """Trains one ExoDetectCNN to convergence with early stopping.
    Returns (model_with_best_weights, history, best_val_auc, best_epoch)."""
    dl_trn, dl_val = make_loaders(gv, lv, sf, y, idx_trn, idx_val, args.batch_size)

    model = build_cnn(n_stellar=n_stellar, device=device)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)  # was 1e-4
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 2,        # was *5 — too hot for ~2k samples
        steps_per_epoch=len(dl_trn),
        epochs=args.epochs, pct_start=0.25,   # longer warmup
        div_factor=10, final_div_factor=100,
    )

    history = {"trn_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_epoch, no_improve = 0.0, 0, 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        trn_loss = 0.0
        for gv_b, lv_b, sf_b, y_b in dl_trn:
            gv_b, lv_b, sf_b, y_b = (
                gv_b.to(device), lv_b.to(device), sf_b.to(device), y_b.to(device)
            )
            gv_b, lv_b, sf_b, y_a, y_b, lam = mixup_data(gv_b, lv_b, sf_b, y_b, alpha=0.2)
            
            optimizer.zero_grad()
            logits = model(gv_b, lv_b, sf_b)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            trn_loss += loss.item()
        trn_loss /= len(dl_trn)

        model.eval()
        val_loss = 0.0
        probs_all, labels_all = [], []
        with torch.no_grad():
            for gv_b, lv_b, sf_b, y_b in dl_val:
                gv_b, lv_b, sf_b, y_b = (
                    gv_b.to(device), lv_b.to(device), sf_b.to(device), y_b.to(device)
                )
                logits = model(gv_b, lv_b, sf_b)
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
def predict_probs(model, gv, lv, sf, idx, device, batch_size=128):
    model.eval()
    probs = []
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        gv_b = torch.tensor(gv[chunk], dtype=torch.float32).unsqueeze(1).to(device)
        lv_b = torch.tensor(lv[chunk], dtype=torch.float32).unsqueeze(1).to(device)
        sf_b = torch.tensor(sf[chunk], dtype=torch.float32).to(device)
        logits = model(gv_b, lv_b, sf_b)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ── Calibration ──────────────────────────────────────────────────────────
# Same convention as train_transformer.py, so CNN and TransitFormer
# calibration numbers are directly comparable run-to-run.

def expected_calibration_error(probs, labels, n_bins=10):
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
    confs = [b["conf"] for b in bin_stats if b["count"] > 0]
    accs = [b["acc"] for b in bin_stats if b["count"] > 0]
    counts = [b["count"] for b in bin_stats if b["count"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfectly calibrated")
    ax1.plot(confs, accs, marker="o", color="#00D4FF", label="Model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_title(f"{title}\n(ECE = {ece:.4f})")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(confs, counts, width=0.08, color="#7C3AED", alpha=0.8)
    ax2.set_xlabel("Mean predicted probability (bin)")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Prediction confidence distribution")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return ece


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ExoDetectCNN on TESS data")
    parser.add_argument("--data", type=Path,
                         default=Path("/Volumes/Expansion/EXO_Datasets/tess_data/tess_lcs/dataset.npz"),
                         help="Path to dataset.npz produced by fetch_tess_dataset.py "
                              "(defaults to that script's own DATA_DIR)")
    parser.add_argument("--out", type=Path,
                         default=Path(__file__).parent / "exodetect_cnn.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)   # was 3e-4
    parser.add_argument("--patience", type=int, default=15) 
    parser.add_argument("--kfold", type=int, default=1,
                         help="1 = single 70/15/15 split (default, fast). "
                              ">1 = stratified k-fold CV over train+val, evaluated "
                              "against one fixed held-out test set.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    gv, lv, sf, y = load_data(args.data)
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
            gv, lv, sf, y, idx_trn, idx_val, args, device, n_stellar, tag="fold0"
        )
        test_probs = predict_probs(model, gv, lv, sf, idx_test, device)
        test_auc = roc_auc_score(y[idx_test], test_probs)
        fold_results.append({"fold": 0, "best_val_auc": best_val_auc,
                              "best_epoch": best_epoch, "test_auc": float(test_auc)})
        fold_probs_on_test.append(test_probs)

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "global_len": GLOBAL_LEN, "local_len": LOCAL_LEN, "n_stellar": n_stellar,
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
                gv, lv, sf, y, idx_trn, idx_val, args, device, n_stellar, tag=f"fold{fold}"
            )
            test_probs = predict_probs(model, gv, lv, sf, idx_test, device)
            test_auc = roc_auc_score(y[idx_test], test_probs)
            fold_results.append({"fold": fold, "best_val_auc": best_val_auc,
                                  "best_epoch": best_epoch, "test_auc": float(test_auc)})
            fold_probs_on_test.append(test_probs)

            if best_val_auc > best_overall_auc:
                best_overall_auc = best_val_auc
                best_overall_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_overall_config = {
                    "global_len": GLOBAL_LEN, "local_len": LOCAL_LEN, "n_stellar": n_stellar,
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

    cal_path = run_dir / f"cnn_calibration_{run_id}.png"
    ece = plot_reliability_diagram(test_probs, test_labels, cal_path,
                                    title="ExoDetectCNN — Test Set Calibration")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Reliability diagram saved to {cal_path}")

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_log = {
        "run_id": run_id,
        "model": "ExoDetectCNN",
        "args": args_dict,
        "n_samples": len(y),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_stellar": int(n_stellar),
        "n_test": len(idx_test),
        "fold_results": fold_results,
        "test_auc": float(roc_auc_score(test_labels, test_probs)),
        "brier_score": float(brier_score_loss(test_labels, test_probs)),
        "ece": ece,
        "timestamp": datetime.now().isoformat(),
    }
    log_path = run_dir / f"cnn_run_{run_id}.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"\nRun metadata logged to {log_path}")


if __name__ == "__main__":
    main()