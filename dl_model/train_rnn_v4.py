#!/usr/bin/env python3
"""ExonRNN v4 — BiGRU with explicit exon-count feature and attention pooling.

Changes from v3:
  - Adds ``total_exons`` as the 13th per-exon feature.
  - Uses learned, length-masked attention over BiGRU output states.
  - Reduces the false-positive loss weight from 4.0 to 3.0.
"""
import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import logging
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from extract_features_minimal import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

EXON_FEATS = [
    'rscu_distance', 'kaks_window_variance', 'polarity_preserving_frac',
    'grantham_mean', 'gc3_diff', 'gc_diff_overall', 'cpg_tpg_rate',
    'aa_freq_cosine', 'enc_diff', 'n_valid_codons', 'kaks_ratio',
]
GLOBAL_FEATS = [
    'rscu_distance', 'kaks_ratio', 'gc_diff_overall',
    'aa_freq_cosine', 'grantham_mean', 'n_valid_codons',
]


def process_group(group_data):
    _, _, group = group_data
    if 'exon_number' in group.columns:
        group = group.sort_values('exon_number')

    total_exons = int(group['exon_number'].max()) if 'exon_number' in group.columns else len(group)
    exon_feats_seq, ref_seqs, query_seqs = [], [], []
    prev_exon_number = 0
    for _, row in group.iterrows():
        ref = str(row['ref_seq']) if pd.notnull(row['ref_seq']) else ''
        query = str(row['query_seq']) if pd.notnull(row['query_seq']) else ''
        ref_seqs.append(ref)
        query_seqs.append(query)
        feats = extract_features(ref, query)

        current_exon_number = int(row['exon_number']) if 'exon_number' in row else 1
        delta_exon = current_exon_number - prev_exon_number
        prev_exon_number = current_exon_number
        exon_feat = [float(feats.get(k, np.nan)) for k in EXON_FEATS]
        exon_feat.extend((float(delta_exon), float(total_exons)))
        exon_feats_seq.append(exon_feat)

    global_feats_dict = extract_features(''.join(ref_seqs), ''.join(query_seqs))
    global_feats = [float(global_feats_dict.get(k, np.nan)) for k in GLOBAL_FEATS]
    label = 1 if group['label'].mean() > 0.5 else 0
    return (np.array(exon_feats_seq, dtype=np.float32),
            np.array(global_feats, dtype=np.float32), label)


def process_dataframe(df, split_name):
    grouped = df.groupby(['transcript_id', 'species'])
    group_data_list = [(tid, sp, grp) for (tid, sp), grp in grouped]
    logger.info(f'{split_name}: {len(group_data_list)} transcript groups')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(executor.map(process_group, group_data_list, chunksize=100))
    X_exon, X_global, y = zip(*results)
    return list(X_exon), np.array(X_global, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_scalers(X_exon_list, X_global):
    """Fit feature scalers using training data only (13 exon features)."""
    all_exons = np.nan_to_num(np.vstack(X_exon_list), nan=0.0)
    return (StandardScaler().fit(all_exons),
            StandardScaler().fit(np.nan_to_num(X_global, nan=0.0)))


def apply_scalers(X_exon_list, X_global, exon_scaler, global_scaler):
    for i, exons in enumerate(X_exon_list):
        X_exon_list[i] = exon_scaler.transform(np.nan_to_num(exons, nan=0.0)).astype(np.float32)
    X_global = global_scaler.transform(np.nan_to_num(X_global, nan=0.0)).astype(np.float32)
    return X_exon_list, X_global


class ExonDataset(Dataset):
    def __init__(self, X_exons, X_globals, y):
        self.X_exons, self.X_globals, self.y = X_exons, X_globals, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X_exons[idx]), torch.from_numpy(self.X_globals[idx]),
                torch.tensor(self.y[idx], dtype=torch.float32))


def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    exons, globs, labels = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in exons])
    return nn.utils.rnn.pad_sequence(exons, batch_first=True), lengths, torch.stack(globs), torch.stack(labels)


class ExonRNN(nn.Module):
    def __init__(self, exon_dim=13, global_dim=6, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(exon_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x, lengths, x_global):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)  # (B, T, 2H)

        scores = self.attention(outputs).squeeze(-1)
        positions = torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
        scores = scores.masked_fill(positions >= lengths.to(outputs.device).unsqueeze(1), float('-inf'))
        weights = torch.softmax(scores, dim=1)
        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return self.mlp(torch.cat((attended, x_global), dim=-1)).squeeze(1)


def main():
    logger.info('Loading CSVs...')
    train_df, test_df = pd.read_csv('train_cleaned.csv'), pd.read_csv('test_cleaned.csv')
    logger.info('Extracting features (train)...')
    X_ex_tr, X_g_tr, y_tr = process_dataframe(train_df, 'Train')
    logger.info('Extracting features (test)...')
    X_ex_te, X_g_te, y_te = process_dataframe(test_df, 'Test')
    logger.info('Fitting scalers on training data...')
    exon_scaler, global_scaler = fit_scalers(X_ex_tr, X_g_tr)
    X_ex_tr, X_g_tr = apply_scalers(X_ex_tr, X_g_tr, exon_scaler, global_scaler)
    X_ex_te, X_g_te = apply_scalers(X_ex_te, X_g_te, exon_scaler, global_scaler)

    train_loader = DataLoader(ExonDataset(X_ex_tr, X_g_tr, y_tr), batch_size=256, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(ExonDataset(X_ex_te, X_g_te, y_te), batch_size=256, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    model = ExonRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    fp_weight, fn_weight = 3.0, 1.0

    def asym_loss(logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weights = torch.where(targets == 1.0, torch.tensor(fn_weight, device=device),
                              torch.tensor(fp_weight, device=device))
        return (bce * weights).mean()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_ex, lengths, X_g, y_b in train_loader:
            X_ex, X_g, y_b = X_ex.to(device), X_g.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = asym_loss(model(X_ex, lengths, X_g), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        logger.info(f'Epoch {epoch + 1}/{epochs}  Loss={total_loss / len(train_loader):.4f}  LR={scheduler.get_last_lr()[0]:.6f}')

    logger.info('Evaluating on test set...')
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X_ex, lengths, X_g, y_b in test_loader:
            logits = model(X_ex.to(device), lengths, X_g.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(y_b.numpy())
    all_probs, all_targets = np.array(all_probs), np.array(all_targets)
    best_f1, best_t = -1, 0.5
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds = (all_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
        acc, prec, rec = accuracy_score(all_targets, preds), precision_score(all_targets, preds, zero_division=0), recall_score(all_targets, preds, zero_division=0)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
        logger.info(f'Threshold={t:.2f} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} TP={tp:,} TN={tn:,} FP={fp:,} FN={fn:,}')
    logger.info(f'Best F1={best_f1:.4f} at threshold={best_t:.2f}')
    torch.save({'model_state_dict': model.state_dict(), 'exon_scaler_mean': exon_scaler.mean_,
                'exon_scaler_scale': exon_scaler.scale_, 'global_scaler_mean': global_scaler.mean_,
                'global_scaler_scale': global_scaler.scale_, 'best_threshold': best_t},
               'dl_orthology_model_v4.pth')
    logger.info('Model + scalers saved to dl_orthology_model_v4.pth')


if __name__ == '__main__':
    main()
