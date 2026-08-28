#!/usr/bin/env python3
"""v10: v9 backbone (BiGRU + FiLM + attention + prototype/Focal) + optional
StructuralConstraintGate (G_struct from Expert M4), trained on multi-clade data
(placentals + opossum), platypus held out for LONO zero-shot.

Clock source: K2P (monotonic fix) or v9 original fixed_clock (baseline).

Usage:
  python3 train_v10.py --clock k2p --struct 1 --name v10_k2p_struct
  python3 train_v10.py --clock orig --struct 0 --name v10_orig_nost
  python3 train_v10.py --clock k2p --struct 0 --name v10_k2p_nost
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE = "/mnt/c/work/profiler_testing/dl_model"
CLOCK_DIRS = {"k2p": f"{BASE}/cached_npz_k2p", "orig": f"{BASE}/cached_npz_origclock"}

# v9 per-exon feature order (17 continuous + 3 binary + 2 extra = 22)
EXON_FEATS = [
    "kmer4_cosine",
    "transition_ratio",
    "rscu_distance",
    "gc3_diff",
    "four_fold_id",
    "grantham_mean",
    "grantham_max",
    "nonsyn_spatial_entropy",
    "cpg_tpg_rate",
    "period3_power",
    "align_longest_ungapped_norm",
    "aa_freq_cosine",
    "max_kaks_window",
    "aa_blosum62_mean",
    "aa_blosum62_negative_frac",
    "aa_longest_conserved_run_norm",
    "aa_pairwise_coverage",
]
BINARY = ["is_ks_near_zero", "is_ks_saturated", "is_kaks_var_missing"]
EXTRA = ["delta_exon", "total_exons"]
V9_FEAT_NAMES = EXON_FEATS + BINARY + EXTRA
N_CONT = len(EXON_FEATS)
SKEWED = ["grantham_mean", "max_kaks_window"]
MACRO_FEATS = [
    "pct_intact_orf",
    "nonsyn_spatial_entropy",
    "aa_blosum62_mean",
    "polarity_preserving_frac",
]


# ---------------------------------------------------------------------------
# Structural Constraint Gate (G_struct) — from Expert M4
# ---------------------------------------------------------------------------
class StructuralConstraintGate(nn.Module):
    def __init__(self, macro_dim=4, hidden_dim=256):
        super().__init__()
        self.macro_proj = nn.Sequential(
            nn.Linear(macro_dim + 1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(macro_dim + 1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, e_sequence, v_macro, d):
        macro_in = torch.cat([v_macro, d], dim=-1)
        g = self.gate_net(macro_in)
        e_macro = self.macro_proj(macro_in)
        return (1.0 - g) * e_sequence + g * e_macro


class ExonRNNV10(nn.Module):
    def __init__(self, exon_dim, veto_idx, use_struct=True, hidden_dim=128):
        super().__init__()
        self.use_struct = use_struct
        rnn_dim = exon_dim - 1 if veto_idx >= 0 else exon_dim
        self.veto_idx = veto_idx
        self.rnn = nn.GRU(
            rnn_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )
        self.film_gamma = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim * 2)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim * 2)
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        if use_struct:
            self.struct_gate = StructuralConstraintGate(
                macro_dim=len(MACRO_FEATS), hidden_dim=hidden_dim * 2
            )
        mlp_in_dim = hidden_dim * 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths, d_ext, v_macro):
        if self.veto_idx >= 0:
            x_seq = torch.cat(
                [x[:, :, : self.veto_idx], x[:, :, self.veto_idx + 1 :]], dim=-1
            )
        else:
            x_seq = x
        lengths_cpu = lengths if lengths.device.type == "cpu" else lengths.cpu()
        lengths_dev = lengths_cpu.to(x.device)
        packed = pack_padded_sequence(
            x_seq, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(
            0
        ) < lengths_dev.unsqueeze(1)
        d = d_ext.to(x.device).float().unsqueeze(1)
        gamma = self.film_gamma(d).unsqueeze(1)
        beta = self.film_beta(d).unsqueeze(1)
        outputs = outputs * (1.0 + gamma) + beta
        outputs = outputs * mask.unsqueeze(-1)
        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        eps = 1e-8
        max_entropy = torch.log(lengths_dev.float().clamp(min=2))
        entropy = (
            -torch.sum(weights * torch.log(weights + eps), dim=1) / max_entropy
        ).mean()
        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (B, 256)
        if self.use_struct:
            e = self.struct_gate(attended, v_macro.to(x.device), d)
        else:
            e = attended
        mlp_input = torch.cat([e, d], dim=1)
        logits = self.mlp(mlp_input).squeeze(1)
        return logits, attended, entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha_pos=0.15, alpha_neg=0.85, gamma_pos=1.0, gamma_neg=5.0):
        super().__init__()
        self.alpha_pos, self.alpha_neg = alpha_pos, alpha_neg
        self.gamma_pos, self.gamma_neg = gamma_pos, gamma_neg

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        alpha_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        gamma_t = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        return (alpha_t * (1 - pt) ** gamma_t * bce).mean()


class ExonDataset(Dataset):
    def __init__(self, X_exons, y, d_clocks, v_macros):
        self.X_exons, self.y, self.d_clocks, self.v_macros = (
            X_exons,
            y,
            d_clocks,
            v_macros,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_exons[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.d_clocks[idx], dtype=torch.float32),
            torch.from_numpy(self.v_macros[idx]),
        )


def collate_fn(batch):
    exons = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    d_clocks = torch.stack([item[2] for item in batch])
    v_macros = torch.stack([item[3] for item in batch])
    lengths = torch.tensor([e.shape[0] for e in exons])
    padded = nn.utils.rnn.pad_sequence(exons, batch_first=True)
    return padded, lengths, labels, d_clocks, v_macros


# ---------------------------------------------------------------------------
# Data loading (from 39-feature cache, faithful to v9's feature pipeline)
# ---------------------------------------------------------------------------
class SplitBundle:
    def __init__(self, X_exons, y, species, d, v_macros):
        self.X_exons, self.y, self.species, self.d, self.v_macros = (
            X_exons,
            y,
            species,
            d,
            v_macros,
        )


def _reorder_and_skew(X_exons_39, feat_names, macro_medians):
    feat_idx = [feat_names.index(f) for f in V9_FEAT_NAMES]
    macro_idx = [feat_names.index(f) for f in MACRO_FEATS]
    skew_idx = [EXON_FEATS.index(k) for k in SKEWED]
    X_out, vm_out = [], []
    for exons in X_exons_39:
        sub = exons[:, feat_idx].copy()
        for si in skew_idx:
            sub[:, si] = np.log1p(np.maximum(0.0, np.nan_to_num(sub[:, si])))
        X_out.append(sub.astype(np.float32))
        macro = exons[:, macro_idx].copy()
        for j in range(len(macro_idx)):
            nan_mask = np.isnan(macro[:, j])
            macro[nan_mask, j] = macro_medians[j]
        vm_out.append(macro.mean(axis=0).astype(np.float32))
    return X_out, vm_out


def _compute_macro_medians(raw_caches, feat_names):
    macro_idx = [feat_names.index(f) for f in MACRO_FEATS]
    parts = []
    for c in raw_caches:
        X = c["X_exons"]
        for i in range(len(X)):
            parts.append(X[i][:, macro_idx])
    allm = np.vstack(parts)
    med = np.nanmedian(allm, axis=0)
    return np.nan_to_num(med, nan=0.0)


def load_split(x_npz, clock_npz, macro_medians):
    cache = np.load(x_npz, allow_pickle=True)
    X_39 = cache["X_exons"]
    feat_names = list(cache["feature_names"])
    clock = np.load(clock_npz, allow_pickle=True)["d_clocks"]
    X_out, vm_out = _reorder_and_skew(X_39, feat_names, macro_medians)
    return SplitBundle(
        X_out,
        cache["y"].astype(np.float32),
        [str(s) for s in cache["species"]],
        clock.astype(np.float32),
        vm_out,
    )


def fit_scalers(X_exon_list):
    all_exons = np.vstack(X_exon_list)
    cont = all_exons[:, :N_CONT]
    imputer = SimpleImputer(strategy="median")
    cont_imputed = imputer.fit_transform(cont)
    scaler = StandardScaler().fit(cont_imputed)
    return imputer, scaler


def apply_scalers(X_exon_list, imputer, scaler):
    for i, exons in enumerate(X_exon_list):
        cont = imputer.transform(exons[:, :N_CONT])
        cont_sc = scaler.transform(cont)
        binary = np.nan_to_num(exons[:, N_CONT:], nan=0.0)
        X_exon_list[i] = np.hstack([cont_sc, binary]).astype(np.float32)
    return X_exon_list


def get_sampler(species_list, labels, seed=0, neg_oversample=3.0):
    combined = [
        f"{sp}_{int(lbl)}" for sp, lbl in zip(species_list, labels, strict=True)
    ]
    counts = pd.Series(combined).value_counts().to_dict()
    weights = []
    for sp, lbl in zip(species_list, labels, strict=True):
        w = 1.0 / counts[f"{sp}_{int(lbl)}"]
        if lbl == 0:
            w *= neg_oversample
        weights.append(w)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.DoubleTensor(weights),  # type: ignore[call-arg]
        num_samples=len(weights),
        replacement=True,
        generator=gen,
    )


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", choices=["k2p", "orig"], default="k2p")
    ap.add_argument("--struct", type=int, default=1)
    ap.add_argument("--name", default="v10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    set_seed(args.seed)
    cdir = CLOCK_DIRS[args.clock]
    logger.info(
        f"=== v10 training: clock={args.clock} struct={args.struct} name={args.name} ==="
    )

    # Load multi-clade training data (placentals + opossum); platypus held out.
    raw_a = np.load(f"{cdir}/train_cache.npz", allow_pickle=True)
    raw_b = np.load(f"{cdir}/val_monDom5_cache.npz", allow_pickle=True)
    macro_medians = _compute_macro_medians([raw_a, raw_b], list(raw_a["feature_names"]))
    train_a = load_split(
        f"{cdir}/train_cache.npz", f"{cdir}/train_cache.npz", macro_medians
    )
    train_b = load_split(
        f"{cdir}/val_monDom5_cache.npz", f"{cdir}/val_monDom5_cache.npz", macro_medians
    )

    X_ex = train_a.X_exons + train_b.X_exons
    y = np.concatenate([train_a.y, train_b.y])
    sp = train_a.species + train_b.species
    d = np.concatenate([train_a.d, train_b.d])
    vm = np.concatenate([train_a.v_macros, train_b.v_macros])

    imputer, scaler = fit_scalers(X_ex)
    X_ex = apply_scalers(X_ex, imputer, scaler)

    d_mean = float(np.mean(d))
    d_std = float(np.std(d))
    if d_std < 1e-8:
        d_std = 1.0
    d_z = (d - d_mean) / d_std  # standardized clock for training
    logger.info(f"multi-clade train: n={len(y)} d_mean={d_mean:.4f} d_std={d_std:.4f}")

    exon_dim = X_ex[0].shape[1]
    veto_idx = EXON_FEATS.index("max_kaks_window")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExonRNNV10(
        exon_dim=exon_dim, veto_idx=veto_idx, use_struct=bool(args.struct)
    ).to(device)

    sampler = get_sampler(sp, y, seed=args.seed)
    loader = DataLoader(
        ExonDataset(X_ex, y, d_z, vm),
        batch_size=256,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(loader), epochs=args.epochs
    )
    criterion = FocalLoss()
    lambda_entropy, lambda_proto = 0.01, 0.5
    hidden_dim = 256
    c_ortholog = torch.zeros(hidden_dim, device=device)
    c_paralog = torch.zeros(hidden_dim, device=device)
    ema = 0.99

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for X_b, lengths, y_b, d_b, vm_b in loader:
            X_b, y_b, d_b, vm_b = (
                X_b.to(device),
                y_b.to(device),
                d_b.to(device),
                vm_b.to(device),
            )
            optimizer.zero_grad()
            logits, attended, entropy = model(X_b, lengths, d_b, vm_b)
            loss_focal = criterion(logits, y_b)
            with torch.no_grad():
                om, pm = y_b == 1, y_b == 0
                if om.any():
                    c_ortholog = ema * c_ortholog + (1 - ema) * attended[om].mean(dim=0)
                if pm.any():
                    c_paralog = ema * c_paralog + (1 - ema) * attended[pm].mean(dim=0)
            c_on = nn.functional.normalize(c_ortholog.unsqueeze(0), dim=1).expand(
                attended.size(0), -1
            )
            c_pn = nn.functional.normalize(c_paralog.unsqueeze(0), dim=1).expand(
                attended.size(0), -1
            )
            h_n = nn.functional.normalize(attended, dim=1)
            logits_proto = torch.stack(
                [(h_n * c_pn).sum(dim=1), (h_n * c_on).sum(dim=1)], dim=1
            )
            loss_proto = nn.functional.cross_entropy(logits_proto / 0.05, y_b.long())
            loss_orth = torch.relu((c_on[0] * c_pn[0]).sum())
            loss = (
                loss_focal
                + lambda_entropy * (-entropy)
                + lambda_proto * loss_proto
                + 0.5 * loss_orth
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tot += loss.item()
        logger.info(f"Epoch {epoch + 1}/{args.epochs}  Loss={tot / len(loader):.4f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "exon_dim": exon_dim,
        "veto_idx": veto_idx,
        "use_struct": bool(args.struct),
        "clock": args.clock,
        "d_mean": d_mean,
        "d_std": d_std,
        "imputer_statistics": imputer.statistics_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "n_cont": N_CONT,
        "macro_medians": macro_medians,
        "seed": args.seed,
    }
    out = f"{BASE}/dl_v10_{args.name}.pth"
    torch.save(ckpt, out)
    logger.info(f"Saved checkpoint -> {out}")


if __name__ == "__main__":
    main()
