#!/usr/bin/env python3
"""ExonRNN v7 — V6 + divergence-reliability signal.

V7 retains V6 features and adds a single continuous reliability scalar
(exp(-d4v/1.5)) as a raw input feature.  The MLP learns how to weight
alignment-dependent features at different divergences rather than using
a hardcoded multiplicative gate (which caused non-monotonic confusion
between zero-divergence and deep-divergence regimes).
"""
import concurrent.futures
import logging
import os
import random
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_features_minimal import extract_features, FEATURE_ORDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

EXON_FEATS = list(FEATURE_ORDER)
SKEWED_FEATS = ['ka', 'ks', 'kaks_ratio', 'grantham_mean', 'kaks_window_variance',
                'max_kaks_window', 'd4v']
EXON_MASK_NAMES = ['is_ks_near_zero', 'is_ks_saturated', 'is_kaks_var_missing']
EXON_EXTRA_NAMES = ['delta_exon', 'total_exons']

# All are continuous and scaled.  Gate and interaction features use RAW feature
# values before V6's log1p transform.  They contain no species or locus identity.
RELIABILITY_NAMES = ['d4v_reliability']
N_BASE_CONTINUOUS = len(EXON_FEATS)
N_EXON_CONTINUOUS = N_BASE_CONTINUOUS + len(RELIABILITY_NAMES)
N_GLOBAL_CONTINUOUS = N_EXON_CONTINUOUS + 1  # + intra-transcript Ka/Ks std


def finite_nonnegative(x):
    return max(0.0, float(x)) if np.isfinite(x) else 0.0


def reliability_block(feats):
    """Divergence reliability signal: 1 near d4v=0, decaying at deeper divergence.
    Fed as a raw feature so the MLP learns the interaction — no manual gating."""
    d4v = finite_nonnegative(feats.get('d4v', np.nan))
    return [float(np.exp(-d4v / 1.5))]


def depth_band_from_d4v(d4v):
    if not np.isfinite(d4v):
        return 'missing'
    if d4v < 0.5:
        return 'recent'
    if d4v < 1.5:
        return 'intermediate'
    return 'deep'


def make_base_features(feats):
    raw = [float(feats.get(k, np.nan)) for k in EXON_FEATS]
    transformed = raw.copy()
    for k in SKEWED_FEATS:
        i = EXON_FEATS.index(k)
        if np.isfinite(transformed[i]):
            transformed[i] = np.log1p(max(0.0, transformed[i]))
    return transformed + reliability_block(feats)


def process_group(group_data):
    _, _, group = group_data
    if 'exon_number' in group.columns:
        group = group.sort_values('exon_number')
    total_exons = int(group['exon_number'].max()) if 'exon_number' in group.columns else len(group)
    exon_rows, ref_seqs, query_seqs, kaks_values = [], [], [], []
    previous_exon = 0

    for _, row in group.iterrows():
        ref = str(row['ref_seq']) if pd.notnull(row['ref_seq']) else ''
        query = str(row['query_seq']) if pd.notnull(row['query_seq']) else ''
        ref_seqs.append(ref)
        query_seqs.append(query)
        exon_number = int(row.get('exon_number', 1))
        feats = extract_features(ref, query, exon_number=exon_number, total_exons=total_exons)
        kaks = feats.get('kaks_ratio', np.nan)
        if np.isfinite(kaks):
            kaks_values.append(float(kaks))

        values = make_base_features(feats)
        ks = feats.get('ks', np.nan)
        masks = [float(np.isfinite(ks) and ks <= 1e-5),
                 float(np.isfinite(ks) and ks >= 9.9),
                 float(not np.isfinite(feats.get('kaks_window_variance', np.nan)))]
        delta = exon_number - previous_exon
        previous_exon = exon_number
        exon_rows.append(values + masks + [float(delta), float(total_exons)])

    global_feats = extract_features(''.join(ref_seqs), ''.join(query_seqs), exon_number=1, total_exons=1)
    global_values = make_base_features(global_feats)
    global_values.append(float(np.std(kaks_values)) if len(kaks_values) > 1 else 0.0)
    ks = global_feats.get('ks', np.nan)
    global_masks = [float(np.isfinite(ks) and ks <= 1e-5),
                    float(np.isfinite(ks) and ks >= 9.9),
                    float(not np.isfinite(global_feats.get('kaks_window_variance', np.nan)))]
    label = int(group['label'].mean() > 0.5)
    species = group['species'].iloc[0] if 'species' in group.columns else 'unknown'
    depth = depth_band_from_d4v(global_feats.get('d4v', np.nan))
    return (np.asarray(exon_rows, dtype=np.float32),
            np.asarray(global_values + global_masks, dtype=np.float32), label, species, depth)


def process_dataframe(df, split_name):
    grouped = df.groupby(['transcript_id', 'species'], sort=False)
    tasks = [(tid, species, group) for (tid, species), group in grouped]
    logger.info('%s: %d transcript/species groups', split_name, len(tasks))
    workers = min(cpu_count(), 8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_group, tasks, chunksize=100))
    x_exon, x_global, labels, species, depths = zip(*results)
    return list(x_exon), np.asarray(x_global, dtype=np.float32), np.asarray(labels, dtype=np.float32), list(species), list(depths)


def fit_scalers(x_exons, x_global):
    exon = np.nan_to_num(np.vstack(x_exons)[:, :N_EXON_CONTINUOUS], nan=0.0, posinf=0.0, neginf=0.0)
    glob = np.nan_to_num(x_global[:, :N_GLOBAL_CONTINUOUS], nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit(exon), StandardScaler().fit(glob)


def apply_scalers(x_exons, x_global, exon_scaler, global_scaler):
    output = []
    for exons in x_exons:
        continuous = exon_scaler.transform(np.nan_to_num(exons[:, :N_EXON_CONTINUOUS], nan=0.0, posinf=0.0, neginf=0.0))
        unscaled = np.nan_to_num(exons[:, N_EXON_CONTINUOUS:], nan=0.0, posinf=0.0, neginf=0.0)
        output.append(np.hstack((continuous, unscaled)).astype(np.float32))
    continuous = global_scaler.transform(np.nan_to_num(x_global[:, :N_GLOBAL_CONTINUOUS], nan=0.0, posinf=0.0, neginf=0.0))
    unscaled = np.nan_to_num(x_global[:, N_GLOBAL_CONTINUOUS:], nan=0.0, posinf=0.0, neginf=0.0)
    return output, np.hstack((continuous, unscaled)).astype(np.float32)


class ExonDataset(Dataset):
    def __init__(self, x_exons, x_globals, labels):
        self.x_exons, self.x_globals, self.labels = x_exons, x_globals, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return torch.from_numpy(self.x_exons[i]), torch.from_numpy(self.x_globals[i]), torch.tensor(self.labels[i], dtype=torch.float32)


def collate_fn(batch):
    batch.sort(key=lambda z: z[0].shape[0], reverse=True)
    exons, globals_, labels = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in exons], dtype=torch.long)
    return nn.utils.rnn.pad_sequence(exons, batch_first=True), lengths, torch.stack(globals_), torch.stack(labels)


class ExonRNN(nn.Module):
    def __init__(self, exon_dim, global_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(exon_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + global_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x, lengths, x_global):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        scores = self.attention(out).squeeze(-1)
        positions = torch.arange(out.size(1), device=out.device).unsqueeze(0)
        scores = scores.masked_fill(positions >= lengths.to(out.device).unsqueeze(1), float('-inf'))
        weights = torch.softmax(scores, dim=1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        attended = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        return self.mlp(torch.cat((attended, x_global), dim=1)).squeeze(1), entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha * (1 - pt).pow(self.gamma) * bce).mean()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def get_species_sampler(species_list, labels, seed=0):
    """V6-style species × label balanced sampler."""
    combined = [f"{sp}_{int(lbl)}" for sp, lbl in zip(species_list, labels)]
    counts = pd.Series(combined).value_counts().to_dict()
    weights = torch.DoubleTensor([1.0 / counts[c] for c in combined])
    return WeightedRandomSampler(weights, len(weights), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def evaluate(model, loader, device, depths):
    model.eval(); probs, targets = [], []
    with torch.no_grad():
        for x, lengths, glob, y in loader:
            logits, _ = model(x.to(device), lengths, glob.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy()); targets.extend(y.numpy())
    probs, targets = np.asarray(probs), np.asarray(targets).astype(int)
    best = (-1.0, 0.5)
    for threshold in np.arange(0.30, 0.801, 0.05):
        pred = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(targets, pred, average='binary', zero_division=0)
        if f1 > best[0]: best = (float(f1), float(threshold))
    threshold = best[1]; pred = (probs >= threshold).astype(int)
    logger.info('Best test F1=%.4f at threshold=%.2f; accuracy=%.4f', best[0], threshold, accuracy_score(targets, pred))
    for band in ('recent', 'intermediate', 'deep', 'missing'):
        idx = np.array([d == band for d in depths])
        if idx.any():
            p, r, f1, _ = precision_recall_fscore_support(targets[idx], pred[idx], average='binary', zero_division=0)
            logger.info('Test %-12s n=%d precision=%.4f recall=%.4f F1=%.4f', band, idx.sum(), p, r, f1)
    return threshold


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    set_seed(seed)
    train_path = '../train_cleaned.csv' if os.path.exists('../train_cleaned.csv') else 'train_cleaned.csv'
    test_path = '../test_cleaned.csv' if os.path.exists('../test_cleaned.csv') else 'test_cleaned.csv'
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    xtr, gtr, ytr, sptr, dtr = process_dataframe(train_df, 'Train')
    xte, gte, yte, spte, dte = process_dataframe(test_df, 'Test')
    exon_scaler, global_scaler = fit_scalers(xtr, gtr)
    xtr, gtr = apply_scalers(xtr, gtr, exon_scaler, global_scaler)
    xte, gte = apply_scalers(xte, gte, exon_scaler, global_scaler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s; exon_dim=%d; global_dim=%d', device, xtr[0].shape[1], gtr.shape[1])
    train_loader = DataLoader(ExonDataset(xtr, gtr, ytr), batch_size=256, sampler=get_species_sampler(sptr, ytr, seed), collate_fn=collate_fn)
    test_loader = DataLoader(ExonDataset(xte, gte, yte), batch_size=256, shuffle=False, collate_fn=collate_fn)
    model = ExonRNN(xtr[0].shape[1], gtr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = FocalLoss(); lambda_entropy = 0.01
    for epoch in range(epochs):
        model.train(); total = 0.0
        for x, lengths, glob, y in train_loader:
            optimizer.zero_grad()
            logits, entropy = model(x.to(device), lengths, glob.to(device))
            loss = criterion(logits, y.to(device)) - lambda_entropy * entropy
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); total += loss.item()
        logger.info('Epoch %d/%d loss=%.4f lr=%.6f', epoch + 1, epochs, total / len(train_loader), scheduler.get_last_lr()[0])
    threshold = evaluate(model, test_loader, device, dte)
    output = 'dl_orthology_model_v7.pth'
    torch.save({'model_state_dict': model.state_dict(), 'exon_scaler_mean': exon_scaler.mean_, 'exon_scaler_scale': exon_scaler.scale_,
                'global_scaler_mean': global_scaler.mean_, 'global_scaler_scale': global_scaler.scale_, 'best_threshold': threshold,
                'exon_dim': xtr[0].shape[1], 'global_dim': gtr.shape[1], 'n_exon_continuous': N_EXON_CONTINUOUS,
                'n_global_continuous': N_GLOBAL_CONTINUOUS, 'exon_feats': EXON_FEATS, 'reliability_names': RELIABILITY_NAMES,
                'skewed_feats': SKEWED_FEATS, 'version': 'v7-reliability-gated'}, output)
    logger.info('Saved %s', output)

if __name__ == '__main__':
    main()
