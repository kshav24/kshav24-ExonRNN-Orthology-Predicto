#!/usr/bin/env python3
"""ExonRNN-V8: Self-Calibrating & Domain-Invariant Sequence Classifier
Pillars:
  1. Sequence-Intrinsic Molecular Clock Extraction (TN93 + RSCU + Delta-GC3 fallback)
  2. Distance-Invariant Exon Features (Selection topography, reading frame, length preservation)
  3. Self-FiLM (Feature-wise Linear Modulation on BiGRU states)
  4. Domain-Adversarial Representation Learning (Gradient Reversal Layer for I(E; d) -> 0)
  5. Hyperspherical Angular Margin Loss (ArcFace on S^{d-1} for single threshold T=0.50)
"""
import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import logging
import math
import os
import sys
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_features_minimal import (
    AATOCODONS,
    CODON_TABLE,
    FOURFOLD,
    STOPS,
    clean_seq,
    codons,
    extract_features,
    is_transition,
    rscu_distance,
    ungap,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pillar 1: Sequence-Intrinsic Molecular Clock Extractor
# ---------------------------------------------------------------------------
def extract_intrinsic_clock(ref_seq, query_seq):
    """Calculates sequence-intrinsic evolutionary distance d strictly from CDS pairs.
    Uses Tamura-Nei (TN93) on 4-fold degenerate sites with deep-time RSCU + Delta-GC3 fallback.
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

    # 1. 4-fold degenerate site tallying
    g_counts = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
    p1_diff = 0.0  # A <-> G transition
    p2_diff = 0.0  # C <-> T transition
    q_diff = 0.0   # Transversions
    ff_sites = 0

    valid_gc3_1 = valid_gc3_2 = gc3_1 = gc3_2 = 0

    for cod1, cod2 in zip(c1[:L], c2[:L]):
        if len(cod1) != 3 or len(cod2) != 3:
            continue
        if '-' not in cod1:
            valid_gc3_1 += 1
            gc3_1 += int(cod1[2] in 'GC')
        if '-' not in cod2:
            valid_gc3_2 += 1
            gc3_2 += int(cod2[2] in 'GC')

        if cod1 in FOURFOLD and cod2 in FOURFOLD and '-' not in cod1 and '-' not in cod2:
            b1, b2 = cod1[2], cod2[2]
            if b1 in g_counts and b2 in g_counts:
                ff_sites += 1
                g_counts[b1] += 0.5
                g_counts[b2] += 0.5
                if b1 != b2:
                    if (b1, b2) in [('A', 'G'), ('G', 'A')]:
                        p1_diff += 1.0
                    elif (b1, b2) in [('C', 'T'), ('T', 'C')]:
                        p2_diff += 1.0
                    else:
                        q_diff += 1.0

    delta_gc3 = abs(gc3_1 / max(1, valid_gc3_1) - gc3_2 / max(1, valid_gc3_2))
    rscu_dist = rscu_distance(c1[:L], c2[:L])

    if ff_sites >= 3:
        gA = g_counts['A'] / ff_sites
        gC = g_counts['C'] / ff_sites
        gG = g_counts['G'] / ff_sites
        gT = g_counts['T'] / ff_sites
        gR = gA + gG
        gY = gT + gC
        P1 = p1_diff / ff_sites
        P2 = p2_diff / ff_sites
        Q = q_diff / ff_sites

        try:
            arg1 = 1.0 - (gR * P1) / max(2.0 * gA * gG, 1e-6) - Q / max(2.0 * gR, 1e-6)
            arg2 = 1.0 - (gY * P2) / max(2.0 * gT * gC, 1e-6) - Q / max(2.0 * gY, 1e-6)
            arg3 = 1.0 - Q / max(2.0 * gR * gY, 1e-6)

            if arg1 > 0 and arg2 > 0 and arg3 > 0:
                d_tn93 = (
                    - (2.0 * gA * gG / max(gR, 1e-6)) * math.log(arg1)
                    - (2.0 * gT * gC / max(gY, 1e-6)) * math.log(arg2)
                    - 2.0 * (gR * gY - (gA * gG * gY / max(gR, 1e-6)) - (gT * gC * gR / max(gY, 1e-6))) * math.log(arg3)
                )
                if 0.0 <= d_tn93 <= 2.5:
                    return float(d_tn93)
        except (ValueError, ZeroDivisionError):
            pass

    # Saturated / fallback composite clock
    d_composite = 0.5 * (rscu_dist / 5.0) + delta_gc3
    return float(max(0.0, min(d_composite, 3.0)))


# ---------------------------------------------------------------------------
# Pillar 2: Feature Configuration & Group Processing
# ---------------------------------------------------------------------------
EXON_FEATS = [
    'kmer4_cosine', 'transition_ratio', 'rscu_distance', 'gc3_diff', 'four_fold_id',
    'grantham_mean', 'grantham_max', 'nonsyn_spatial_entropy', 'cpg_tpg_rate',
    'period3_power', 'align_longest_ungapped_norm', 'aa_freq_cosine',
    'max_kaks_window', 'aa_blosum62_mean', 'aa_blosum62_negative_frac',
    'aa_longest_conserved_run_norm', 'aa_pairwise_coverage', 'kaks_ratio',
    'kaks_window_variance', 'pct_intact_orf', 'exon_length_ratio'
]

SKEWED_FEATS = ['grantham_mean', 'max_kaks_window', 'kaks_ratio', 'kaks_window_variance']
EXON_MASK_NAMES = ['is_ks_near_zero', 'is_ks_saturated', 'is_kaks_var_missing']
EXON_EXTRA_NAMES = ['delta_exon', 'total_exons']

N_EXON_CONTINUOUS = len(EXON_FEATS)
N_EXON_BINARY = len(EXON_MASK_NAMES) + len(EXON_EXTRA_NAMES)

def process_group(group_data):
    _, _, group = group_data
    if 'exon_number' in group.columns:
        group = group.sort_values('exon_number')
    total_exons = int(group['exon_number'].max()) if 'exon_number' in group.columns else len(group)
    exon_feats_seq = []
    prev_exon_number = 0

    full_ref = []
    full_query = []

    for _, row in group.iterrows():
        ref = str(row['ref_seq']) if pd.notnull(row['ref_seq']) else ''
        query = str(row['query_seq']) if pd.notnull(row['query_seq']) else ''
        full_ref.append(ref)
        full_query.append(query)

        feats = extract_features(ref, query, exon_number=int(row.get('exon_number', 1)), total_exons=total_exons)
        current_exon_number = int(row.get('exon_number', 1))
        delta_exon = current_exon_number - prev_exon_number
        prev_exon_number = current_exon_number

        exon_feat = [float(feats.get(k, np.nan)) for k in EXON_FEATS]
        ks_val = feats.get('ks', np.nan)
        is_ks_near_zero = 1.0 if (not np.isnan(ks_val) and ks_val <= 1e-5) else 0.0
        is_ks_saturated = 1.0 if (not np.isnan(ks_val) and ks_val >= 9.9) else 0.0
        is_kaks_var_missing = 1.0 if np.isnan(feats.get('kaks_window_variance', np.nan)) else 0.0
        exon_feat.extend([is_ks_near_zero, is_ks_saturated, is_kaks_var_missing])

        for k in SKEWED_FEATS:
            if k in EXON_FEATS:
                idx = EXON_FEATS.index(k)
                val = exon_feat[idx]
                if not np.isnan(val):
                    exon_feat[idx] = np.log1p(max(0, val))

        exon_feat.extend([float(delta_exon), float(total_exons)])
        exon_feats_seq.append(exon_feat)

    # Compute transcript-level intrinsic molecular clock d
    d_clock = extract_intrinsic_clock(''.join(full_ref), ''.join(full_query))

    label = 1 if group['label'].mean() > 0.5 else 0
    species = group['species'].iloc[0] if 'species' in group.columns else 'unknown'
    return (np.array(exon_feats_seq, dtype=np.float32), float(d_clock), label, species)


def process_dataframe(df, split_name):
    grouped = df.groupby(['transcript_id', 'species'])
    group_data_list = [(tid, sp, grp) for (tid, sp), grp in grouped]
    logger.info(f'{split_name}: {len(group_data_list)} transcript groups')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(executor.map(process_group, group_data_list, chunksize=100))
    X_exon, d_clocks, y, species = zip(*results)
    return list(X_exon), np.array(d_clocks, dtype=np.float32), np.array(y, dtype=np.float32), list(species)


def fit_scalers(X_exon_list):
    all_exons = np.vstack(X_exon_list)
    cont = all_exons[:, :N_EXON_CONTINUOUS]
    imputer = SimpleImputer(strategy='median')
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


# ---------------------------------------------------------------------------
# Dataset & Collation
# ---------------------------------------------------------------------------
class ExonDatasetV8(Dataset):
    def __init__(self, X_exons, d_clocks, y):
        self.X_exons = X_exons
        self.d_clocks = d_clocks
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_exons[idx]),
            torch.tensor(self.d_clocks[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def collate_fn_v8(batch):
    exons, d_clocks, labels = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in exons])
    padded_exons = nn.utils.rnn.pad_sequence(exons, batch_first=True)
    return padded_exons, torch.stack(d_clocks), lengths, torch.stack(labels)


# ---------------------------------------------------------------------------
# Pillars 3, 4, 5: Neural Architecture (Self-FiLM, GRL, ArcFace)
# ---------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_weight):
        ctx.lambda_weight = lambda_weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_weight, None


class ArcFaceHead(nn.Module):
    """Pillar 5: Additive Angular Margin Loss on Unit Hypersphere."""
    def __init__(self, in_features=256, num_classes=2, s=32.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(emb_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.training and labels is not None:
            theta = torch.acos(cos_theta)
            target_logits = torch.cos(theta + self.m)
            one_hot = F.one_hot(labels.long(), num_classes=2)
            logits = cos_theta + one_hot * (target_logits - cos_theta)
            return logits * self.s
        else:
            return cos_theta


class ExonRNN_V8(nn.Module):
    def __init__(self, exon_dim, hidden_dim=128, num_layers=2, dropout=0.3, s=32.0, m=0.30):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Sequence BiGRU Encoder
        self.rnn = nn.GRU(
            exon_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        # Pillar 3: Self-FiLM Conditioning MLP (maps scalar clock d -> gamma & beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2 * 2),  # 256 gamma + 256 beta
        )

        # Exon Attention Pooling
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Pillar 4: Domain-Adversarial Regressor (predicts d from E)
        self.domain_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Pillar 5: Hyperspherical ArcFace Head
        self.arcface = ArcFaceHead(in_features=hidden_dim * 2, num_classes=2, s=s, m=m)

    def forward(self, x, d, lengths, labels=None, lambda_adv=0.10):
        # 1. BiGRU Forward Pass
        lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.cpu()
        lengths_dev = lengths_cpu.to(x.device)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # 2. Pillar 3: Self-FiLM affine modulation on hidden states
        d_input = d.unsqueeze(-1)  # (B, 1)
        film_params = self.film_mlp(d_input)  # (B, 512)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # (B, 256), (B, 256)
        outputs = (1.0 + gamma.unsqueeze(1)) * outputs + beta.unsqueeze(1)

        # Mask padding
        mask = torch.arange(outputs.size(1), device=x.device).unsqueeze(0) < lengths_dev.unsqueeze(1)
        outputs = outputs * mask.unsqueeze(-1).float()

        # 3. Attention Pooling
        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)

        # Normalized entropy
        eps = 1e-8
        max_entropy = torch.log(lengths_dev.float().clamp(min=2))
        entropy = (-torch.sum(weights * torch.log(weights + eps), dim=1) / max_entropy).mean()

        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # E in R^{B x 256}

        # 4. Pillar 4: Domain-Adversarial Branch (GRL)
        reversed_emb = GradientReversalFunction.apply(attended, lambda_adv)
        pred_d = self.domain_regressor(reversed_emb).squeeze(-1)

        # 5. Pillar 5: ArcFace Classification
        logits = self.arcface(attended, labels)
        return logits, pred_d, attended, entropy


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    set_seed(seed)
    logger.info(f'=== Starting ExonRNN-V8 Training (Seed: {seed}) ===')

    train_path = '../train_cleaned.csv'
    test_path = '../test_cleaned.csv'
    if not os.path.exists(train_path):
        train_path = 'train_cleaned.csv'
        test_path = 'test_cleaned.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info('Extracting features and molecular clocks (Train)...')
    X_ex_tr, d_tr, y_tr, sp_tr = process_dataframe(train_df, 'Train')

    logger.info('Extracting features and molecular clocks (Test)...')
    X_ex_te, d_te, y_te, sp_te = process_dataframe(test_df, 'Test')

    # Fit Imputer + Scaler strictly on Train
    imputer, exon_scaler = fit_scalers(X_ex_tr)
    X_ex_tr = apply_scalers(X_ex_tr, imputer, exon_scaler)
    X_ex_te = apply_scalers(X_ex_te, imputer, exon_scaler)

    exon_dim = X_ex_tr[0].shape[1]
    logger.info(f'Exon feature dimension: {exon_dim}')

    # Weighted Sampler
    combined = [f"{sp}_{int(lbl)}" for sp, lbl in zip(sp_tr, y_tr)]
    class_counts = pd.Series(combined).value_counts().to_dict()
    weights = [3.0 / class_counts[c] if c.endswith('_0') else 1.0 / class_counts[c] for c in combined]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        ExonDatasetV8(X_ex_tr, d_tr, y_tr),
        batch_size=256, sampler=sampler, collate_fn=collate_fn_v8
    )
    test_loader = DataLoader(
        ExonDatasetV8(X_ex_te, d_te, y_te),
        batch_size=256, shuffle=False, collate_fn=collate_fn_v8
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExonRNN_V8(exon_dim=exon_dim, hidden_dim=128, num_layers=2, dropout=0.3, s=32.0, m=0.30).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    logger.info('Training ExonRNN-V8...')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_arc = 0.0
        total_adv = 0.0

        for X_b, d_b, lengths_b, y_b in train_loader:
            X_b, d_b, y_b = X_b.to(device), d_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            logits, pred_d, attended, entropy = model(X_b, d_b, lengths_b, labels=y_b, lambda_adv=0.10)

            l_arc = loss_ce(logits, y_b.long())
            l_adv = loss_mse(pred_d, d_b)
            loss = l_arc + 0.10 * l_adv + 0.01 * (-entropy)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_arc += l_arc.item()
            total_adv += l_adv.item()

        logger.info(
            f'Epoch {epoch+1:02d}/{epochs} | Total: {total_loss/len(train_loader):.4f} | '
            f'ArcFace: {total_arc/len(train_loader):.4f} | AdvMSE: {total_adv/len(train_loader):.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.6f}'
        )

    # Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'exon_imputer_statistics': imputer.statistics_,
        'exon_scaler_mean': exon_scaler.mean_,
        'exon_scaler_scale': exon_scaler.scale_,
        'exon_dim': exon_dim,
        'exon_feats': EXON_FEATS,
        'best_threshold': 0.50,  # Universal Single Threshold on Angular Geometry
    }
    torch.save(checkpoint, 'dl_orthology_model_v8.pth')
    logger.info('Saved model checkpoint -> dl_orthology_model_v8.pth')


if __name__ == '__main__':
    main()
