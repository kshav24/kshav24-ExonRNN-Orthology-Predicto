#!/usr/bin/env python3
"""ExonRNN v9 — Fixed-Evolutionary-Distance Orthology Model
=============================================================
SINGLE-VARIABLE change vs v7c (archive_legacy_experiments/train_rnn_v7c.py):

  v7c: FiLM + MLP distance signal = mean(max_kaks_window)  [per-exon kaks proxy]
  v9 : FiLM + MLP distance signal = FIXED d_clock = exact TN93 on ALL 3rd-codon-position
         sites, with a Jukes-Cantor 3rd-position fallback.

Everything else is byte-for-byte identical to v7c:
  - BiGRU over the 21 non-veto exon features (max_kaks_window still vetoed out of the RNN).
  - Same FocalLoss (alpha_pos=0.15, alpha_neg=0.85, gamma_pos=1.0, gamma_neg=5.0).
  - Same entropy reg (0.01), prototype contrastive (0.5) + orthogonality (0.5).
  - Same AdamW(lr=1e-3, wd=1e-4), OneCycleLR, 30 epochs, neg_oversample=3.0, LayerNorm MLP.
  - Same 6-species evaluation block.

WHY: Phase 0 (plans/02_tn93_verdict.md, plans/04_realdata_clock.md) proved the code's TN93
formula is EXACT, but the 4-fold d_clock collapses onto a flat non-physical composite
fallback for 19-38% of non-placental transcripts (vs 1-5% for placentals) due to 4-fold
site scarcity. Computing the exact TN93 on ALL 3rd-codon sites cuts that fallback to
~16% (opossum) / ~28% (platypus) and calibrates to true tree distance (monDom5 median
0.764 vs true 0.799). This is the "fixed evolutionary distance" for v9.
"""

import warnings

warnings.filterwarnings("ignore")

import concurrent.futures
import logging
import math
import os
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_features_minimal import clean_seq, codons, extract_features

# Sequence-only per-exon features (identical to v7c).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SKEWED_FEATS = ["grantham_mean", "max_kaks_window"]
EXON_MASK_NAMES = ["is_ks_near_zero", "is_ks_saturated", "is_kaks_var_missing"]
EXON_EXTRA_NAMES = ["delta_exon", "total_exons"]

N_EXON_CONTINUOUS = len(EXON_FEATS)
N_EXON_BINARY = len(EXON_MASK_NAMES) + len(EXON_EXTRA_NAMES)


# ---------------------------------------------------------------------------
# FIXED EVOLUTIONARY DISTANCE  (the ONLY new signal in v9)
# ---------------------------------------------------------------------------
def fixed_clock(ref_seq, query_seq):
    """Exact TN93 distance on ALL 3rd-codon-position sites.

    Phase-0 verified: the TN93 closed form used here is an EXACT estimator of true
    evolutionary distance (plans/02_tn93_verdict.md). Computing it on all 3rd-position
    sites (not just 4-fold) is robust: low fallback, well calibrated for non-placentals.

    Fallback (rare: <3 sites or TN93 arg<=0 / out of [0,2.5]): Jukes-Cantor on the
    3rd-position difference fraction — always defined for p < 0.75, saturation-capped.
    """
    s1 = clean_seq(ref_seq)
    s2 = clean_seq(query_seq)
    if s1 == s2:
        return 0.0
    c1 = codons(s1)
    c2 = codons(s2)
    L = min(len(c1), len(c2))
    if L == 0:
        return 0.0
    gc = {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0}
    p1 = p2 = q = 0.0
    n = 0
    n_diff = 0
    for i in range(L):
        cod1, cod2 = c1[i], c2[i]
        if len(cod1) != 3 or len(cod2) != 3:
            continue
        b1, b2 = cod1[2], cod2[2]  # 3rd codon position
        if b1 == "-" or b2 == "-":
            continue
        if b1 in gc and b2 in gc:
            n += 1
            gc[b1] += 0.5
            gc[b2] += 0.5
            if b1 != b2:
                n_diff += 1
                if (b1, b2) in [("A", "G"), ("G", "A")]:
                    p1 += 1.0
                elif (b1, b2) in [("C", "T"), ("T", "C")]:
                    p2 += 1.0
                else:
                    q += 1.0
    if n < 3:
        return 0.0
    gA, gC, gG, gT = gc["A"] / n, gc["C"] / n, gc["G"] / n, gc["T"] / n
    gR = gA + gG
    gY = gT + gC
    P1, P2, Q = p1 / n, p2 / n, q / n
    try:
        arg1 = 1.0 - (gR * P1) / max(2.0 * gA * gG, 1e-6) - Q / max(2.0 * gR, 1e-6)
        arg2 = 1.0 - (gY * P2) / max(2.0 * gT * gC, 1e-6) - Q / max(2.0 * gY, 1e-6)
        arg3 = 1.0 - Q / max(2.0 * gR * gY, 1e-6)
        if arg1 > 0 and arg2 > 0 and arg3 > 0:
            d_tn93 = (
                -(2.0 * gA * gG / max(gR, 1e-6)) * math.log(arg1)
                - (2.0 * gT * gC / max(gY, 1e-6)) * math.log(arg2)
                - 2.0
                * (
                    gR * gY
                    - (gA * gG * gY / max(gR, 1e-6))
                    - (gT * gC * gR / max(gY, 1e-6))
                )
                * math.log(arg3)
            )
            if 0.0 <= d_tn93 <= 2.5:
                return float(d_tn93)
    except (ValueError, ZeroDivisionError):
        pass
    # Jukes-Cantor fallback on 3rd-position difference fraction
    p_diff = n_diff / n
    if p_diff < 0.75:
        d_jc = -0.75 * math.log(1.0 - (4.0 / 3.0) * p_diff)
    else:
        d_jc = 2.0
    return float(min(max(d_jc, 0.0), 2.5))


def process_group(group_data):
    _, _, expected_label, group = group_data
    if "exon_number" in group.columns:
        group = group.sort_values("exon_number")
    total_exons = (
        int(group["exon_number"].max())
        if "exon_number" in group.columns
        else len(group)
    )
    exon_feats_seq, ref_seqs, query_seqs = [], [], []
    prev_exon_number = 0
    for _, row in group.iterrows():
        ref = str(row["ref_seq"]) if pd.notnull(row["ref_seq"]) else ""
        query = str(row["query_seq"]) if pd.notnull(row["query_seq"]) else ""
        ref_seqs.append(ref)
        query_seqs.append(query)
        feats = extract_features(
            ref,
            query,
            exon_number=int(row.get("exon_number", 1)),
            total_exons=total_exons,
        )
        current_exon_number = int(row.get("exon_number", 1))
        delta_exon = current_exon_number - prev_exon_number
        prev_exon_number = current_exon_number

        exon_feat = [float(feats.get(k, np.nan)) for k in EXON_FEATS]
        ks_val = feats.get("ks", np.nan)
        is_ks_near_zero = 1.0 if (not np.isnan(ks_val) and ks_val <= 1e-5) else 0.0
        is_ks_saturated = 1.0 if (not np.isnan(ks_val) and ks_val >= 9.9) else 0.0
        is_kaks_var_missing = (
            1.0 if np.isnan(feats.get("kaks_window_variance", np.nan)) else 0.0
        )
        exon_feat.extend([is_ks_near_zero, is_ks_saturated, is_kaks_var_missing])
        for k in SKEWED_FEATS:
            if k in EXON_FEATS:
                idx = EXON_FEATS.index(k)
                val = exon_feat[idx]
                if not np.isnan(val):
                    exon_feat[idx] = np.log1p(max(0, val))
        exon_feat.extend((float(delta_exon), float(total_exons)))
        exon_feats_seq.append(exon_feat)

    label = float(expected_label)
    species = group["species"].iloc[0] if "species" in group.columns else "unknown"
    # V9: transcript-level fixed evolutionary distance (concatenated CDS)
    full_ref = "".join(ref_seqs)
    full_query = "".join(query_seqs)
    d_clock = fixed_clock(full_ref, full_query)
    return (np.array(exon_feats_seq, dtype=np.float32), label, species, d_clock)


def process_dataframe(df, split_name):
    if "label" in df.columns:
        grouped = df.groupby(["transcript_id", "species", "label"])
        group_data_list = [(tid, sp, lbl, grp) for (tid, sp, lbl), grp in grouped]
    else:
        grouped = df.groupby(["transcript_id", "species"])
        group_data_list = [(tid, sp, 0, grp) for (tid, sp), grp in grouped]
    logger.info(f"{split_name}: {len(group_data_list)} transcript groups")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(executor.map(process_group, group_data_list, chunksize=100))
    X_exon, y, species, d_clocks = zip(*results)
    return (
        list(X_exon),
        np.array(y, dtype=np.float32),
        list(species),
        np.array(d_clocks, dtype=np.float32),
    )


def fit_scalers(X_exon_list):
    all_exons = np.vstack(X_exon_list)
    cont = all_exons[:, :N_EXON_CONTINUOUS]
    imputer = SimpleImputer(strategy="median")
    cont_imputed = imputer.fit_transform(cont)
    scaler = StandardScaler().fit(cont_imputed)
    return imputer, scaler


def apply_scalers(X_exon_list, imputer, scaler):
    for i, exons in enumerate(X_exon_list):
        cont = exons[:, :N_EXON_CONTINUOUS]
        cont_imputed = imputer.transform(cont)
        cont_scaled = scaler.transform(cont_imputed)
        binary = np.nan_to_num(exons[:, N_EXON_CONTINUOUS:], nan=0.0)
        X_exon_list[i] = np.hstack([cont_scaled, binary]).astype(np.float32)
    return X_exon_list


class ExonDataset(Dataset):
    def __init__(self, X_exons, y, d_clocks):
        self.X_exons, self.y, self.d_clocks = X_exons, y, d_clocks

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_exons[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.d_clocks[idx], dtype=torch.float32),
        )


def collate_fn(batch):
    exons, labels, d_clocks = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in exons])
    return (
        nn.utils.rnn.pad_sequence(exons, batch_first=True),
        lengths,
        torch.stack(labels),
        torch.stack(d_clocks),
    )


class ExonRNNSeq(nn.Module):
    def __init__(
        self, exon_dim, veto_idx=-1, hidden_dim=128, num_layers=2, dropout=0.4
    ):
        super().__init__()
        self.veto_idx = veto_idx
        rnn_dim = exon_dim - 1 if veto_idx >= 0 else exon_dim
        self.rnn = nn.GRU(
            rnn_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        if self.veto_idx >= 0:
            self.film_gamma = nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim * 2)
            )
            self.film_beta = nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim * 2)
            )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        mlp_in_dim = hidden_dim * 2 + 1 if veto_idx >= 0 else hidden_dim * 2
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

    def forward(self, x, lengths, d_ext):
        # V9: d_ext = standardized fixed d_clock [batch]. max_kaks_window still vetoed from RNN.
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
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        if self.veto_idx >= 0:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(
                0
            ) < lengths_dev.unsqueeze(1)
            d = (
                d_ext.to(x.device).float().unsqueeze(1)
            )  # [batch, 1]  (V9: fixed d_clock)
            gamma = self.film_gamma(d).unsqueeze(1)
            beta = self.film_beta(d).unsqueeze(1)
            outputs = outputs * (1.0 + gamma) + beta
            outputs = outputs * mask.unsqueeze(-1)

        scores = self.attention(outputs).squeeze(-1)
        positions = torch.arange(outputs.size(1), device=x.device).unsqueeze(0)
        scores = scores.masked_fill(
            positions >= lengths_dev.unsqueeze(1), float("-inf")
        )
        weights = torch.softmax(scores, dim=1)

        eps = 1e-8
        max_entropy = torch.log(lengths_dev.float().clamp(min=2))
        entropy = (
            -torch.sum(weights * torch.log(weights + eps), dim=1) / max_entropy
        ).mean()

        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        if self.veto_idx >= 0:
            mlp_input = torch.cat([attended, d], dim=1)  # V9: fixed d_clock scalar
        else:
            mlp_input = attended

        logits = self.mlp(mlp_input).squeeze(1)
        return logits, attended, entropy


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha_pos=0.15,
        alpha_neg=0.85,
        gamma_pos=1.0,
        gamma_neg=5.0,
        reduction="mean",
    ):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE)
        alpha_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        gamma_t = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        loss = alpha_t * (1 - pt) ** gamma_t * BCE
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_species_sampler(species_list, labels, seed=0, neg_oversample=3.0):
    combined = [f"{sp}_{int(lbl)}" for sp, lbl in zip(species_list, labels)]
    df = pd.DataFrame({"class": combined})
    class_counts = df["class"].value_counts().to_dict()
    weights = []
    for sp, lbl in zip(species_list, labels):
        cls = f"{sp}_{int(lbl)}"
        w = 1.0 / class_counts[cls]
        if lbl == 0:
            w *= neg_oversample
        weights.append(w)
    weights = torch.DoubleTensor(weights)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )
    return sampler


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")
    train_path = "../train_cleaned.csv"
    test_path = "../test_cleaned.csv"
    if not os.path.exists(train_path):
        train_path = "train_cleaned.csv"
        test_path = "test_cleaned.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info("Extracting features (train)...")
    X_ex_tr, y_tr, sp_tr, d_tr = process_dataframe(train_df, "Train")
    logger.info("Extracting features (test)...")
    X_ex_te, y_te, sp_te, d_te = process_dataframe(test_df, "Test")

    imputer, exon_scaler = fit_scalers(X_ex_tr)
    X_ex_tr = apply_scalers(X_ex_tr, imputer, exon_scaler)
    X_ex_te = apply_scalers(X_ex_te, imputer, exon_scaler)

    # V9: standardize the fixed d_clock (fit on train) so its scale matches v7c's
    # standardized distance input (mean~0, std~1).
    d_mean = float(np.mean(d_tr))
    d_std = float(np.std(d_tr))
    if d_std < 1e-8:
        d_std = 1.0
    d_tr_std = ((d_tr - d_mean) / d_std).astype(np.float32)
    d_te_std = ((d_te - d_mean) / d_std).astype(np.float32)
    logger.info(f"V9 fixed d_clock: train mean={d_mean:.4f} std={d_std:.4f}")

    exon_dim = X_ex_tr[0].shape[1]
    sampler = get_species_sampler(sp_tr, y_tr, seed=seed, neg_oversample=3.0)
    train_loader = DataLoader(
        ExonDataset(X_ex_tr, y_tr, d_tr_std),
        batch_size=256,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ExonDataset(X_ex_te, y_te, d_te_std),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        veto_idx = EXON_FEATS.index("max_kaks_window")
    except ValueError:
        veto_idx = -1

    model = ExonRNNSeq(exon_dim=exon_dim, veto_idx=veto_idx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )

    criterion = FocalLoss(alpha_pos=0.15, alpha_neg=0.85, gamma_pos=1.0, gamma_neg=5.0)
    lambda_entropy = 0.01
    lambda_proto = 0.5

    hidden_dim = 256
    c_ortholog = torch.zeros(hidden_dim, device=device)
    c_paralog = torch.zeros(hidden_dim, device=device)
    ema_momentum = 0.99

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_ex, lengths, y_b, d_b in train_loader:
            X_ex, y_b, d_b = X_ex.to(device), y_b.to(device), d_b.to(device)
            optimizer.zero_grad()
            logits, attended, entropy = model(X_ex, lengths, d_b)
            loss_focal = criterion(logits, y_b)
            with torch.no_grad():
                ortho_mask = y_b == 1
                para_mask = y_b == 0
                if ortho_mask.any():
                    ortho_mean = attended[ortho_mask].mean(dim=0)
                    c_ortholog = (
                        ema_momentum * c_ortholog + (1 - ema_momentum) * ortho_mean
                    )
                if para_mask.any():
                    para_mean = attended[para_mask].mean(dim=0)
                    c_paralog = (
                        ema_momentum * c_paralog + (1 - ema_momentum) * para_mean
                    )
            c_ortho_n = nn.functional.normalize(c_ortholog.unsqueeze(0), dim=1).expand(
                attended.size(0), -1
            )
            c_para_n = nn.functional.normalize(c_paralog.unsqueeze(0), dim=1).expand(
                attended.size(0), -1
            )
            h_n = nn.functional.normalize(attended, dim=1)
            sim_ortho = (h_n * c_ortho_n).sum(dim=1)
            sim_para = (h_n * c_para_n).sum(dim=1)
            logits_proto = torch.stack([sim_para, sim_ortho], dim=1)
            target_proto = y_b.long()
            loss_proto = nn.functional.cross_entropy(logits_proto / 0.05, target_proto)

            proto_sim = (c_ortho_n[0] * c_para_n[0]).sum()
            loss_orthogonal = torch.relu(proto_sim - 0.0)

            loss = (
                loss_focal
                + lambda_entropy * (-entropy)
                + lambda_proto * loss_proto
                + 0.5 * loss_orthogonal
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        logger.info(
            f"Epoch {epoch + 1}/{epochs}  Loss={total_loss / len(train_loader):.4f}  LR={scheduler.get_last_lr()[0]:.6f}"
        )

    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X_ex, lengths, y_b, d_b in test_loader:
            logits, _, _ = model(X_ex.to(device), lengths, d_b.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(y_b.numpy())
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    best_metric, best_t = -1, 0.5
    for t in np.arange(0.30, 0.85, 0.05):
        preds = (all_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
        prec = precision_score(all_targets, preds, zero_division=0)
        rec = recall_score(all_targets, preds, zero_division=0)
        f05 = (
            (1 + 0.5**2) * prec * rec / ((0.5**2 * prec) + rec) if prec + rec > 0 else 0
        )
        if f05 > best_metric:
            best_metric, best_t = f05, t
        logger.info(
            f"Thresh={t:.2f} Prec={prec:.4f} Rec={rec:.4f} F0.5={f05:.4f} TP={tp} TN={tn} FP={fp} FN={fn}"
        )
    logger.info(f"Best F0.5={best_metric:.4f} at threshold={best_t:.2f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "exon_imputer_statistics": imputer.statistics_,
            "exon_scaler_mean": exon_scaler.mean_,
            "exon_scaler_scale": exon_scaler.scale_,
            "best_threshold": float(best_t),
            "exon_dim": exon_dim,
            "exon_feats": EXON_FEATS,
            "d_clock_mean": d_mean,
            "d_clock_std": d_std,
        },
        "dl_orthology_model_v9.pth",
    )
    logger.info("Saved dl_orthology_model_v9.pth")

    # Automatic 6-Species Evaluation
    logger.info("\n" + "=" * 70)
    logger.info("  RUNNING 6-SPECIES VALIDATION EVALUATION (Threshold = %.2f)", best_t)
    logger.info("=" * 70)
    validation_dir = "/mnt/c/work/profiler_testing/ensembl_validation"
    species_list = ["bisBis1", "monDom5", "myoLuc2", "ornAna2", "phyCat1", "rn6"]
    eval_results = []

    for sp in species_list:
        csv_path = f"{validation_dir}/raw_validation_data_{sp}.csv"
        if not os.path.exists(csv_path):
            continue
        df_sp = pd.read_csv(csv_path)
        X_sp, y_sp, _, d_sp = process_dataframe(df_sp, sp)
        X_sp = apply_scalers(X_sp, imputer, exon_scaler)
        d_sp_std = ((d_sp - d_mean) / d_std).astype(np.float32)
        sp_loader = DataLoader(
            ExonDataset(X_sp, y_sp, d_sp_std),
            batch_size=256,
            shuffle=False,
            collate_fn=collate_fn,
        )

        sp_probs, sp_targets = [], []
        with torch.no_grad():
            for Xb, lengths_b, yb, db in sp_loader:
                logits_b, _, _ = model(Xb.to(device), lengths_b, db.to(device))
                sp_probs.extend(torch.sigmoid(logits_b).cpu().numpy())
                sp_targets.extend(yb.numpy())

        sp_probs = np.array(sp_probs)
        sp_targets = np.array(sp_targets).astype(int)
        sp_preds = (sp_probs >= best_t).astype(int)

        tn, fp, fn, tp = confusion_matrix(sp_targets, sp_preds, labels=[0, 1]).ravel()
        acc = accuracy_score(sp_targets, sp_preds)
        prec = precision_score(sp_targets, sp_preds, zero_division=0)
        rec = recall_score(sp_targets, sp_preds, zero_division=0)
        f1_sp = f1_score(sp_targets, sp_preds, zero_division=0)
        # like-for-like comparison at the v7c bar threshold (0.4)
        sp_preds_04 = (sp_probs >= 0.4).astype(int)
        f1_sp_04 = f1_score(sp_targets, sp_preds_04, zero_division=0)
        rec_sp_04 = recall_score(sp_targets, sp_preds_04, zero_division=0)

        eval_results.append(
            {
                "species": sp,
                "n": len(sp_targets),
                "threshold": best_t,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1_sp,
                "f1_at_0.4": f1_sp_04,
                "recall_at_0.4": rec_sp_04,
            }
        )

    df_eval = pd.DataFrame(eval_results)
    out_csv = "v9_6species_evaluation_summary.csv"
    df_eval.to_csv(out_csv, index=False)
    logger.info("\n%s", df_eval.to_string(index=False))


if __name__ == "__main__":
    main()
