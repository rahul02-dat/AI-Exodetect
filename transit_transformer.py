"""
transit_transformer.py — Phase 3: TransitFormer

A Transformer-based transit classifier that:
  1. Processes the full phase-folded light curve as a sequence
  2. Uses multi-head self-attention — learns WHICH time steps matter
  3. Returns attention weights for XAI heatmap overlay
  4. Fuses with CNN output in an ensemble head

Architecture:
  Input  → Patch embedding (light curve split into N patches)
         → Positional encoding
         → L Transformer encoder layers (multi-head attention)
         → [CLS] token → classification head
         → Also returns: attention_weights per head per patch

Apple Silicon note:
  torch.backends.mps.is_available() is checked at runtime.
  All operations are MPS-compatible (no custom CUDA kernels).

Phase 4 update:
  TransitFormer now records its own n_stellar (self.n_stellar), and
  ExoEnsemble.predict() builds each submodel's stellar feature tensor
  internally via stellar_features.build_stellar_features(), sized to that
  submodel's own n_stellar. This lets a CNN still on the legacy 4-feature
  schema and a TransitFormer already retrained on the expanded 8-feature
  schema (or vice versa) be ensembled together during a gradual rollout,
  instead of requiring both models to be upgraded in lockstep.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings("ignore")

from stellar_features import build_stellar_features


# ── Device ─────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Works for sequences up to max_len steps.
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Split the 1D light curve into overlapping patches using 1D convolution,
    which helps maintain local continuity across patch boundaries.

    light_curve: (B, 1, L)  →  patches: (B, n_patches, d_model)
    """
    def __init__(self, seq_len, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        
        # Overlapping patches: stride < patch_size
        stride = max(1, patch_size // 2)
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)
        
        # Calculate resulting number of patches
        self.n_patches = ((seq_len - patch_size) // stride) + 1

    def forward(self, x):
        # x: (B, 1, L)
        x = self.proj(x)          # (B, d_model, n_patches)
        return x.transpose(1, 2)  # (B, n_patches, d_model)


# ── Transformer Encoder with Attention Export ──────────────────────────────

class AttentionLayer(nn.Module):
    """
    Single multi-head attention layer that also returns the
    attention weight matrix — needed for XAI heatmap.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        if return_attn:
            # Average attention across heads: (B, T, T)
            return out, attn.mean(dim=1)
        return out, None


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn   = AttentionLayer(d_model, n_heads, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ── TransitFormer ──────────────────────────────────────────────────────────

class TransitFormer(nn.Module):
    """
    Transformer classifier for exoplanet transit light curves.

    Input:
      global_view : (B, 1, 201)  — phase-folded, full orbit
      stellar_feats: (B, n_stellar) — period, duration, depth, SNR
                       (+ Teff, radius, log g, Tmag if n_stellar=8)

    Output:
      logits       : (B, 2)      — planet / false-positive
      attn_weights : (B, n_patches) — XAI signal per patch

    Parameters (defaults optimised for TESS 2-min cadence):
      seq_len    = 201   (global view length)
      patch_size = 3     → 67 patches
      d_model    = 64
      n_heads    = 4
      n_layers   = 4
      ff_dim     = 256
      n_stellar  = 4     (legacy) or 8 (expanded, includes host-star params)
    """

    def __init__(
        self,
        seq_len     = 201,
        patch_size  = 3,
        d_model     = 64,
        n_heads     = 4,
        n_layers    = 4,
        ff_dim      = 256,
        n_stellar   = 4,
        dropout     = 0.1,
        n_classes   = 2,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.patch_size = patch_size
        stride = max(1, patch_size // 2)
        self.n_patches = ((seq_len - patch_size) // stride) + 1
        self.n_stellar  = n_stellar

        # Patch embedding
        self.patch_embed = PatchEmbedding(seq_len, patch_size, d_model)

        # CLS token (learnable) — aggregates global sequence info
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding for n_patches + 1 (CLS)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.n_patches + 1, dropout=dropout)

        # Transformer encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Stellar feature MLP
        self.stellar_mlp = nn.Sequential(
            nn.Linear(n_stellar, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, global_view, stellar_feats, return_attn=False):
        """
        global_view   : (B, 1, seq_len)
        stellar_feats : (B, n_stellar)
        return_attn   : if True, also return attention weights
        """
        B = global_view.size(0)

        # Patch embedding → (B, n_patches, d_model)
        x = self.patch_embed(global_view)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)          # (B, n_patches+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer layers — collect attention from last layer
        last_attn = None
        for i, layer in enumerate(self.layers):
            capture = return_attn and (i == len(self.layers) - 1)
            x, attn = layer(x, return_attn=capture)
            if capture:
                last_attn = attn   # (B, n_patches+1, n_patches+1)

        x = self.norm(x)

        # CLS token output → classifier
        cls_out = x[:, 0]   # (B, d_model)

        # Stellar features
        sf_out = self.stellar_mlp(stellar_feats)   # (B, 32)

        # Fuse and classify
        logits = self.head(torch.cat([cls_out, sf_out], dim=1))

        # XAI: attention from CLS → patches (skip CLS→CLS self-attention)
        attn_weights = None
        if return_attn and last_attn is not None:
            # last_attn: (B, T+1, T+1), row 0 = CLS attending to all patches
            attn_weights = last_attn[:, 0, 1:]   # (B, n_patches)

        return logits, attn_weights

    def predict_proba(self, global_view, stellar_feats):
        """Returns P(planet) scalar and attention weights as numpy arrays."""
        with torch.no_grad():
            logits, attn = self.forward(global_view, stellar_feats, return_attn=True)
            prob  = torch.softmax(logits, dim=1)[:, 1]
            return prob.cpu().numpy(), attn.cpu().numpy() if attn is not None else None


# ── Ensemble: CNN + TransitFormer ─────────────────────────────────────────

class ExoEnsemble:
    """
    Wraps both models and fuses their predictions.
    Soft voting: P_ensemble = w_cnn * P_cnn + w_tf * P_transformer
    Weights can be tuned based on validation AUC.
    """

    def __init__(self, cnn_model, transformer_model,
                 cnn_weight=0.45, tf_weight=0.55,
                 device=None):
        self.cnn = cnn_model
        self.tf  = transformer_model
        self.w_cnn = cnn_weight
        self.w_tf  = tf_weight
        self.device = device or get_device()

    def predict(self, global_view_t, local_view_t, period, dur_hr, depth_ppm, snr,
                teff=None, rad=None, logg=None, tmag=None):
        """
        global_view_t / local_view_t are pre-built tensors on self.device.
        Everything else is a raw (unnormalized) scalar — each submodel's
        stellar feature tensor is built here internally, sized to that
        submodel's own n_stellar (self.cnn.n_stellar / self.tf.n_stellar),
        so mismatched schema versions between CNN and TransitFormer never
        cause a shape error.

        Returns:
          p_planet      : float  — ensemble P(planet)
          p_cnn         : float  — CNN alone
          p_transformer : float  — TransitFormer alone
          attn_weights  : np.ndarray (n_patches,) — for XAI
        """
        sf_cnn = build_stellar_features(self.cnn.n_stellar, period, dur_hr, depth_ppm, snr,
                                         teff, rad, logg, tmag)
        sf_tf  = build_stellar_features(self.tf.n_stellar, period, dur_hr, depth_ppm, snr,
                                         teff, rad, logg, tmag)
        sf_cnn_t = torch.tensor([sf_cnn]).to(self.device)
        sf_tf_t  = torch.tensor([sf_tf]).to(self.device)

        # CNN
        with torch.no_grad():
            cnn_logits = self.cnn(global_view_t, local_view_t, sf_cnn_t)
            p_cnn = float(torch.softmax(cnn_logits, dim=1)[:, 1].item())

        # TransitFormer
        p_tf_arr, attn = self.tf.predict_proba(global_view_t, sf_tf_t)
        p_tf = float(p_tf_arr[0])

        # Soft ensemble
        p_ensemble = self.w_cnn * p_cnn + self.w_tf * p_tf

        attn_1d = attn[0] if attn is not None else None

        return p_ensemble, p_cnn, p_tf, attn_1d

    def classify(self, p_planet, depth_ppm, period, duration_hr):
        """Convert P(planet) into full 4-class probability dict."""
        p_fp = 1.0 - p_planet
        if depth_ppm > 50000:
            eb, bl, sp = 0.65, 0.25, 0.10
        elif duration_hr / (period * 24 + 1e-6) > 0.15:
            eb, bl, sp = 0.20, 0.65, 0.15
        else:
            eb, bl, sp = 0.35, 0.40, 0.25
        return {
            "Exoplanet Transit": round(p_planet * 100, 1),
            "Eclipsing Binary":  round(p_fp * eb * 100, 1),
            "Stellar Blend":     round(p_fp * bl * 100, 1),
            "Starspot":          round(p_fp * sp * 100, 1),
        }


# ── Training helper (used in training script) ─────────────────────────────

def build_transformer(seq_len=201, patch_size=3, n_stellar=4, device=None):
    """Instantiate a fresh TransitFormer and move to device."""
    device = device or get_device()
    model  = TransitFormer(
        seq_len    = seq_len,
        patch_size = patch_size,
        d_model    = 64,
        n_heads    = 4,
        n_layers   = 4,
        ff_dim     = 256,
        n_stellar  = n_stellar,
        dropout    = 0.1,
        n_classes  = 2,
    ).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TransitFormer] {total:,} trainable parameters on {device} (n_stellar={n_stellar})")
    return model


def load_transformer(path, device=None):
    """Load a saved TransitFormer checkpoint."""
    device = device or get_device()
    ckpt   = torch.load(path, map_location="cpu")
    cfg    = ckpt.get("model_config", {})
    model  = TransitFormer(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[TransitFormer] Loaded from {path} — Val AUC {ckpt.get('metrics',{}).get('best_val_auc','?')} "
          f"(n_stellar={model.n_stellar})")
    return model, ckpt.get("metrics", {})


if __name__ == "__main__":
    # Quick smoke test
    dev = get_device()
    print(f"Device: {dev}")
    m   = build_transformer(device=dev)
    gv  = torch.randn(4, 1, 201).to(dev)
    sf  = torch.randn(4, 4).to(dev)
    logits, attn = m(gv, sf, return_attn=True)
    print(f"Logits shape : {logits.shape}")     # (4, 2)
    print(f"Attn shape   : {attn.shape}")       # (4, 67)
    print("✅ TransitFormer smoke test passed")