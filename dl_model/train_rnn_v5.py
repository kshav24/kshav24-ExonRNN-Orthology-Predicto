#!/usr/bin/env python3
"""ExonRNN v5 — BiGRU with all Sequence-Only 10-point plan improvements.
"""
import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import logging
import sys
import os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Add parent directory to path since extract_features_minimal.py is in the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_features_minimal import extract_features, FEATURE_ORDER_V5

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Frozen V5 feature contract.  V6 training should use FEATURE_ORDER so it also
# receives the appended d4v clock feature.
EXON_FEATS = list(FEATURE_ORDER_V5)

# Skewed features to log1p transform before scaling
SKEWED_FEATS = ['ka', 'ks', 'kaks_ratio', 'grantham_mean', 'kaks_window_variance', 'max_kaks_window']

# Binary mask names appended after EXON_FEATS (these must NOT be scaled)
EXON_MASK_NAMES = ['is_ks_near_zero', 'is_ks_saturated', 'is_kaks_var_missing']
# Extra positional features appended after masks (also not scaled — they are ordinal integers)
EXON_EXTRA_NAMES = ['delta_exon', 'total_exons']

# Number of continuous features that get scaled = len(EXON_FEATS)
# Total exon dim = len(EXON_FEATS) + len(EXON_MASK_NAMES) + len(EXON_EXTRA_NAMES)
N_EXON_CONTINUOUS = len(EXON_FEATS)
N_EXON_BINARY = len(EXON_MASK_NAMES) + len(EXON_EXTRA_NAMES)

# Global: same continuous features + intra_transcript_kaks_std (continuous) + 3 masks (binary)
N_GLOBAL_CONTINUOUS = len(EXON_FEATS) + 1  # +1 for intra_transcript_kaks_std
N_GLOBAL_BINARY = len(EXON_MASK_NAMES)

def process_group(group_data):
    _, _, group = group_data
    if 'exon_number' in group.columns:
        group = group.sort_values('exon_number')

    total_exons = int(group['exon_number'].max()) if 'exon_number' in group.columns else len(group)
    exon_feats_seq, ref_seqs, query_seqs = [], [], []
    prev_exon_number = 0
    kaks_list = []
    
    for _, row in group.iterrows():
        ref = str(row['ref_seq']) if pd.notnull(row['ref_seq']) else ''
        query = str(row['query_seq']) if pd.notnull(row['query_seq']) else ''
        ref_seqs.append(ref)
        query_seqs.append(query)
        feats = extract_features(ref, query, exon_number=int(row.get('exon_number', 1)), total_exons=total_exons)
        current_exon_number = int(row.get('exon_number', 1))
        delta_exon = current_exon_number - prev_exon_number
        prev_exon_number = current_exon_number
        
        # Collect kaks for transcript consistency before log-transforming
        if not np.isnan(feats.get('kaks_ratio', np.nan)):
            kaks_list.append(float(feats['kaks_ratio']))

        exon_feat = [float(feats.get(k, np.nan)) for k in EXON_FEATS]
        
        # Append ks_near_zero, ks_saturated, kaks_var_missing
        ks_val = feats.get('ks', np.nan)
        is_ks_near_zero = 1.0 if (not np.isnan(ks_val) and ks_val <= 1e-5) else 0.0
        is_ks_saturated = 1.0 if (not np.isnan(ks_val) and ks_val >= 9.9) else 0.0
        is_kaks_var_missing = 1.0 if np.isnan(feats.get('kaks_window_variance', np.nan)) else 0.0
        
        exon_feat.extend([is_ks_near_zero, is_ks_saturated, is_kaks_var_missing])
        
        # Apply log1p to skewed features directly in the array
        for k in SKEWED_FEATS:
            idx = EXON_FEATS.index(k)
            val = exon_feat[idx]
            if not np.isnan(val):
                exon_feat[idx] = np.log1p(max(0, val))
                
        exon_feat.extend((float(delta_exon), float(total_exons)))
        exon_feats_seq.append(exon_feat)

    global_feats_dict = extract_features(''.join(ref_seqs), ''.join(query_seqs), exon_number=1, total_exons=1)
    global_feat = [float(global_feats_dict.get(k, np.nan)) for k in EXON_FEATS]
    
    intra_transcript_kaks_std = np.std(kaks_list) if len(kaks_list) > 1 else 0.0
    global_feat.append(float(intra_transcript_kaks_std))
    
    # Missing masks and log1p for global
    ks_val = global_feats_dict.get('ks', np.nan)
    is_ks_near_zero = 1.0 if (not np.isnan(ks_val) and ks_val <= 1e-5) else 0.0
    is_ks_saturated = 1.0 if (not np.isnan(ks_val) and ks_val >= 9.9) else 0.0
    is_kaks_var_missing = 1.0 if np.isnan(global_feats_dict.get('kaks_window_variance', np.nan)) else 0.0
    
    global_feat.extend([is_ks_near_zero, is_ks_saturated, is_kaks_var_missing])
    
    for k in SKEWED_FEATS:
        idx = EXON_FEATS.index(k)
        val = global_feat[idx]
        if not np.isnan(val):
            global_feat[idx] = np.log1p(max(0, val))

    label = 1 if group['label'].mean() > 0.5 else 0
    species = group['species'].iloc[0] if 'species' in group.columns else 'unknown'
    
    return (np.array(exon_feats_seq, dtype=np.float32),
            np.array(global_feat, dtype=np.float32), label, species)


def process_dataframe(df, split_name):
    grouped = df.groupby(['transcript_id', 'species'])
    group_data_list = [(tid, sp, grp) for (tid, sp), grp in grouped]
    logger.info(f'{split_name}: {len(group_data_list)} transcript groups')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(executor.map(process_group, group_data_list, chunksize=100))
    X_exon, X_global, y, species = zip(*results)
    return list(X_exon), np.array(X_global, dtype=np.float32), np.array(y, dtype=np.float32), list(species)

def fit_scalers(X_exon_list, X_global):
    """Fit scalers on ONLY the continuous features, leaving binary masks untouched."""
    all_exons = np.vstack(X_exon_list)
    cont = np.nan_to_num(all_exons[:, :N_EXON_CONTINUOUS], nan=0.0)
    exon_scaler = StandardScaler().fit(cont)

    glob_cont = np.nan_to_num(X_global[:, :N_GLOBAL_CONTINUOUS], nan=0.0)
    global_scaler = StandardScaler().fit(glob_cont)
    return exon_scaler, global_scaler


def apply_scalers(X_exon_list, X_global, exon_scaler, global_scaler):
    """Scale only continuous columns; binary masks pass through unmodified."""
    for i, exons in enumerate(X_exon_list):
        cont = exon_scaler.transform(np.nan_to_num(exons[:, :N_EXON_CONTINUOUS], nan=0.0))
        binary = exons[:, N_EXON_CONTINUOUS:]  # masks + delta_exon + total_exons
        # nan_to_num on binary part just in case (shouldn't have NaNs, but defensive)
        binary = np.nan_to_num(binary, nan=0.0)
        X_exon_list[i] = np.hstack([cont, binary]).astype(np.float32)

    glob_cont = global_scaler.transform(np.nan_to_num(X_global[:, :N_GLOBAL_CONTINUOUS], nan=0.0))
    glob_binary = np.nan_to_num(X_global[:, N_GLOBAL_CONTINUOUS:], nan=0.0)
    X_global = np.hstack([glob_cont, glob_binary]).astype(np.float32)
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
    def __init__(self, exon_dim, global_dim, hidden_dim=128, num_layers=2, dropout=0.3):
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
        
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=1).mean()
        
        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return self.mlp(torch.cat((attended, x_global), dim=-1)).squeeze(1), entropy

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_species_sampler(species_list, labels, seed=0):
    combined = [f"{sp}_{lbl}" for sp, lbl in zip(species_list, labels)]
    df = pd.DataFrame({'class': combined})
    class_counts = df['class'].value_counts().to_dict()
    weights = [1.0 / class_counts[c] for c in combined]
    weights = torch.DoubleTensor(weights)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    return sampler

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    set_seed(seed)
    logger.info(f'Using random seed: {seed}')
    logger.info('Loading CSVs...')
    train_path = '../train_cleaned.csv'
    test_path = '../test_cleaned.csv'
    if not os.path.exists(train_path):
        train_path = 'train_cleaned.csv'
        test_path = 'test_cleaned.csv'
        
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    logger.info('Extracting features (train)...')
    X_ex_tr, X_g_tr, y_tr, sp_tr = process_dataframe(train_df, 'Train')
    logger.info('Extracting features (test)...')
    X_ex_te, X_g_te, y_te, sp_te = process_dataframe(test_df, 'Test')
    
    logger.info('Fitting scalers on training data...')
    exon_scaler, global_scaler = fit_scalers(X_ex_tr, X_g_tr)
    X_ex_tr, X_g_tr = apply_scalers(X_ex_tr, X_g_tr, exon_scaler, global_scaler)
    X_ex_te, X_g_te = apply_scalers(X_ex_te, X_g_te, exon_scaler, global_scaler)

    exon_dim = X_ex_tr[0].shape[1]
    global_dim = X_g_tr.shape[1]
    logger.info(f"Using {exon_dim} Exon Features and {global_dim} Global Features")

    sampler = get_species_sampler(sp_tr, y_tr, seed=seed)
    train_loader = DataLoader(ExonDataset(X_ex_tr, X_g_tr, y_tr), batch_size=256, sampler=sampler, collate_fn=collate_fn)
    test_loader = DataLoader(ExonDataset(X_ex_te, X_g_te, y_te), batch_size=256, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    model = ExonRNN(exon_dim=exon_dim, global_dim=global_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    lambda_entropy = 0.01

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_ex, lengths, X_g, y_b in train_loader:
            X_ex, X_g, y_b = X_ex.to(device), X_g.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, entropy = model(X_ex, lengths, X_g)
            loss = criterion(logits, y_b) - lambda_entropy * entropy
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
            logits, _ = model(X_ex.to(device), lengths, X_g.to(device))
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
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'exon_scaler_mean': exon_scaler.mean_,
        'exon_scaler_scale': exon_scaler.scale_,
        'global_scaler_mean': global_scaler.mean_,
        'global_scaler_scale': global_scaler.scale_,
        'best_threshold': best_t,
        'exon_dim': exon_dim,
        'global_dim': global_dim,
        'n_exon_continuous': N_EXON_CONTINUOUS,
        'n_global_continuous': N_GLOBAL_CONTINUOUS,
        'exon_feats': EXON_FEATS,
        'skewed_feats': SKEWED_FEATS,
    }, 'dl_orthology_model_v5.pth')
    logger.info('Model + scalers saved to dl_orthology_model_v5.pth')

if __name__ == '__main__':
    main()
