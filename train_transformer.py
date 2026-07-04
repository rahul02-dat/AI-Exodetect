"""
train_transformer.py — Train TransitFormer on your local Mac (Apple Silicon MPS)

Usage:
    python train_transformer.py

Expects the dataset cache from Phase 1 at:
    ~/kepler_lcs/dataset.npz   (or set DATA_PATH below)

If you don't have it, run Phase 1 notebook first to build the cache,
then download dataset.npz from Colab Files panel.

Saves: exodetect_transformer.pt  (alongside exodetect_cnn.pt)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time, warnings
warnings.filterwarnings("ignore")

from transit_transformer import build_transformer, get_device, TransitFormer

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH    = Path(__file__).parent / "data" / "kepler_lcs" / "dataset.npz"
OUT_PATH     = Path(__file__).parent / "exodetect_transformer.pt"
GLOBAL_LEN   = 201
PATCH_SIZE   = 3       # 201 / 3 = 67 patches
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 2e-4
DEVICE       = get_device()

print(f"Device        : {DEVICE}")
print(f"Dataset path  : {DATA_PATH}")
print(f"Output path   : {OUT_PATH}")


# ── Dataset ─────────────────────────────────────────────────────────────────

class TransitDataset(Dataset):
    """Re-uses Phase 1 dataset.npz — only needs global_views."""
    def __init__(self, gv, sf, y, augment=False):
        self.gv      = torch.tensor(gv, dtype=torch.float32)
        self.sf      = torch.tensor(sf, dtype=torch.float32)
        self.y       = torch.tensor(y,  dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        gv = self.gv[idx]
        sf = self.sf[idx]
        y  = self.y[idx]
        if self.augment:
            # Gaussian noise
            gv = gv + torch.randn_like(gv) * 0.01
            # Random roll (phase shift)
            gv = torch.roll(gv, int(torch.randint(0, len(gv), (1,)).item()))
            # Amplitude jitter
            gv = gv * (1 + 0.02 * torch.randn(1).item())
        return gv.unsqueeze(0), sf, y   # (1, L), (4,), scalar


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at {DATA_PATH}\n"
            "   Download dataset.npz from your Colab session (Files panel)\n"
            "   and place it at the path above, OR update DATA_PATH in this script."
        )
    d = np.load(DATA_PATH)
    print(f"Loaded dataset: {d['global_views'].shape[0]} samples")
    return (d["global_views"].astype(np.float32),
            d["stellar_feats"].astype(np.float32),
            d["labels"].astype(np.int64))


# ── Training loop ───────────────────────────────────────────────────────────

def train():
    gv, sf, labels = load_data()

    # 70 / 15 / 15 split
    idx = np.arange(len(labels))
    idx_trn, idx_tmp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
    idx_val, idx_tst = train_test_split(idx_tmp, test_size=0.50,
                                         stratify=labels[idx_tmp], random_state=42)

    ds_trn = TransitDataset(gv[idx_trn], sf[idx_trn], labels[idx_trn], augment=True)
    ds_val = TransitDataset(gv[idx_val], sf[idx_val], labels[idx_val])
    ds_tst = TransitDataset(gv[idx_tst], sf[idx_tst], labels[idx_tst])

    # Weighted sampler for class imbalance
    counts  = np.bincount(labels[idx_trn])
    weights = 1.0 / counts[labels[idx_trn]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dl_trn = DataLoader(ds_trn, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)
    dl_tst = DataLoader(ds_tst, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    print(f"Train: {len(ds_trn)}  Val: {len(ds_val)}  Test: {len(ds_tst)}")

    # Build model
    model = build_transformer(seq_len=GLOBAL_LEN, patch_size=PATCH_SIZE, device=DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history    = {"trn_loss": [], "val_loss": [], "val_auc": []}
    best_auc   = 0.0
    best_epoch = 0
    patience   = 10
    no_improve = 0

    print(f"\nTraining TransitFormer for {EPOCHS} epochs on {DEVICE}…\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        trn_loss = 0.0
        for gv_b, sf_b, y_b in dl_trn:
            gv_b, sf_b, y_b = gv_b.to(DEVICE), sf_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(gv_b, sf_b, return_attn=False)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item()
        trn_loss /= len(dl_trn)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        probs_all, labels_all = [], []
        with torch.no_grad():
            for gv_b, sf_b, y_b in dl_val:
                gv_b, sf_b, y_b = gv_b.to(DEVICE), sf_b.to(DEVICE), y_b.to(DEVICE)
                logits, _ = model(gv_b, sf_b)
                val_loss += criterion(logits, y_b).item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_all.extend(probs.cpu().numpy())
                labels_all.extend(y_b.cpu().numpy())
        val_loss /= len(dl_val)
        val_auc   = roc_auc_score(labels_all, probs_all)

        history["trn_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Early stopping
        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "seq_len":    GLOBAL_LEN,
                    "patch_size": PATCH_SIZE,
                    "d_model":    64,
                    "n_heads":    4,
                    "n_layers":   4,
                    "ff_dim":     256,
                    "n_stellar":  4,
                    "dropout":    0.1,
                    "n_classes":  2,
                },
                "metrics": {
                    "best_val_auc": best_auc,
                    "best_epoch":   best_epoch,
                },
            }, OUT_PATH)
        else:
            no_improve += 1

        marker = " ◀ best" if epoch == best_epoch else ""
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"trn={trn_loss:.4f}  val={val_loss:.4f}  AUC={val_auc:.4f}{marker}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── Test evaluation ─────────────────────────────────────────────────────
    print(f"\n✅ Best val AUC: {best_auc:.4f} at epoch {best_epoch}")
    print("Loading best checkpoint for test evaluation…")

    ckpt = torch.load(OUT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    probs_tst, labels_tst = [], []
    with torch.no_grad():
        for gv_b, sf_b, y_b in dl_tst:
            gv_b, sf_b = gv_b.to(DEVICE), sf_b.to(DEVICE)
            logits, _  = model(gv_b, sf_b)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_tst.extend(probs.cpu().numpy())
            labels_tst.extend(y_b.numpy())

    probs_tst  = np.array(probs_tst)
    labels_tst = np.array(labels_tst)
    test_auc   = roc_auc_score(labels_tst, probs_tst)

    print(f"\n=== Test Set Evaluation ===")
    print(f"ROC-AUC : {test_auc:.4f}")
    print(classification_report(labels_tst, (probs_tst >= 0.5).astype(int),
                                 target_names=["False Positive", "Planet"]))

    # Update checkpoint with test AUC
    ckpt["metrics"]["test_auc"] = test_auc
    torch.save(ckpt, OUT_PATH)

    # ── Training curves ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["trn_loss"]) + 1)
    ax1.plot(ep, history["trn_loss"], label="Train", color="#A78BFA")
    ax1.plot(ep, history["val_loss"],  label="Val",   color="#FF4757")
    ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("TransitFormer Loss"); ax1.legend()
    ax2.plot(ep, history["val_auc"], color="#00D4FF", linewidth=2)
    ax2.axhline(best_auc, color="gray", linestyle="--")
    ax2.set_title(f"Val AUC (best={best_auc:.4f})"); ax2.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plot_path = OUT_PATH.parent / "transformer_training.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n✅ Training plot saved to {plot_path}")
    print(f"✅ Model saved to {OUT_PATH}")


if __name__ == "__main__":
    train()