#!/usr/bin/env python3
"""ExonRNN V7C + Platypus: Original V7C architecture trained with full Platypus (ornAna2) data in training.
Evaluated on all held-out validation species: Rat, Bison, Bat, Whale, Opossum.
"""
import logging
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = '/mnt/c/work/profiler_testing/dl_model/cached_npz'
EVAL_SPECIES = ['rn6', 'bisBis1', 'myoLuc2', 'phyCat1', 'monDom5']

V7C_FEATS_CONT = [
    'kmer4_cosine', 'transition_ratio', 'rscu_distance', 'gc3_diff', 'four_fold_id',
    'grantham_mean', 'grantham_max', 'nonsyn_spatial_entropy', 'cpg_tpg_rate', 'period3_power',
    'align_longest_ungapped_norm', 'aa_freq_cosine', 'max_kaks_window', 'aa_blosum62_mean',
    'aa_blosum62_negative_frac', 'aa_longest_conserved_run_norm', 'aa_pairwise_coverage'
]
V7C_FEATS_DISC = ['is_ks_near_zero', 'is_ks_saturated', 'is_kaks_var_missing', 'delta_exon', 'total_exons']
SKEWED = ['grantham_mean', 'max_kaks_window']
N_CONT = len(V7C_FEATS_CONT)
VETO_FEAT = 'max_kaks_window'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExonDatasetV7C(Dataset):
    def __init__(self, X_exons, y):
        self.X_exons = X_exons
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_exons[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def collate_fn_v7c(batch):
    exons, labels = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in exons])
    padded_exons = nn.utils.rnn.pad_sequence(exons, batch_first=True)
    return padded_exons, lengths, torch.stack(labels)


class ExonRNN_V7C(nn.Module):
    def __init__(self, exon_dim=22, hidden_dim=128, num_layers=2, dropout=0.4, veto_idx=-1):
        super().__init__()
        self.veto_idx = veto_idx

        rnn_dim = exon_dim - 1 if veto_idx >= 0 else exon_dim
        self.rnn = nn.GRU(
            rnn_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        if self.veto_idx >= 0:
            self.film_gamma = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim * 2))
            self.film_beta  = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim * 2))

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
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths):
        if self.veto_idx >= 0:
            x_seq = torch.cat([x[:, :, :self.veto_idx], x[:, :, self.veto_idx + 1:]], dim=-1)
            feat = x[:, :, self.veto_idx]
        else:
            x_seq = x

        lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.cpu()
        lengths_dev = lengths_cpu.to(x.device)

        packed = pack_padded_sequence(x_seq, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths_dev.unsqueeze(1)
        mean_feat = None
        if self.veto_idx >= 0:
            feat_masked = feat * mask.float()
            mean_feat = feat_masked.sum(dim=1) / lengths_dev.float().clamp(min=1)
            d = mean_feat.unsqueeze(1)

            gamma = self.film_gamma(d).unsqueeze(1)
            beta  = self.film_beta(d).unsqueeze(1)
            outputs = outputs * (1.0 + gamma) + beta
            outputs = outputs * mask.unsqueeze(-1)

        scores = self.attention(outputs).squeeze(-1)
        positions = torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
        scores = scores.masked_fill(positions >= lengths_dev.unsqueeze(1), float('-inf'))
        weights = torch.softmax(scores, dim=1)

        eps = 1e-8
        max_entropy = torch.log(lengths_dev.float().clamp(min=2))
        entropy = (-torch.sum(weights * torch.log(weights + eps), dim=1) / max_entropy).mean()

        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)

        if self.veto_idx >= 0:
            mlp_input = torch.cat([attended, mean_feat.unsqueeze(1)], dim=1)
        else:
            mlp_input = attended

        logits = self.mlp(mlp_input).squeeze(1)
        return logits, attended, entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha_pos=0.15, alpha_neg=0.85, gamma_pos=1.0, gamma_neg=5.0, reduction='mean'):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        alpha_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        gamma_t = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        loss = alpha_t * (1 - pt) ** gamma_t * BCE
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


def get_species_sampler(species_list, labels, seed=0, neg_oversample=3.0):
    combined = [f"{sp}_{int(lbl)}" for sp, lbl in zip(species_list, labels)]
    df = pd.DataFrame({'class': combined})
    class_counts = df['class'].value_counts().to_dict()
    weights = []
    for sp, lbl in zip(species_list, labels):
        cls = f"{sp}_{int(lbl)}"
        w = 1.0 / class_counts[cls]
        if lbl == 0:
            w *= neg_oversample
        weights.append(w)
    weights = torch.DoubleTensor(weights)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def preprocess_npz(X_raw, fn, imputer_stats=None, scaler=None, fit=False):
    indices_cont = [fn.index(f) for f in V7C_FEATS_CONT]
    indices_disc = [fn.index(f) for f in V7C_FEATS_DISC]

    all_cont = []
    for exons in X_raw:
        sub = exons[:, indices_cont].copy()
        for s_feat in SKEWED:
            s_idx = V7C_FEATS_CONT.index(s_feat)
            sub[:, s_idx] = np.log1p(np.maximum(0.0, np.nan_to_num(sub[:, s_idx], nan=0.0)))
        all_cont.append(sub)

    if fit:
        stacked = np.vstack(all_cont)
        imputer_stats = np.nanmedian(stacked, axis=0)
        for i in range(len(all_cont)):
            for col in range(all_cont[i].shape[1]):
                m = np.isnan(all_cont[i][:, col])
                all_cont[i][m, col] = imputer_stats[col]
        scaler = StandardScaler()
        scaler.fit(np.vstack(all_cont))
    else:
        for i in range(len(all_cont)):
            for col in range(all_cont[i].shape[1]):
                m = np.isnan(all_cont[i][:, col])
                all_cont[i][m, col] = imputer_stats[col]

    X_proc = []
    for i, exons in enumerate(X_raw):
        sc = scaler.transform(all_cont[i])
        disc = np.nan_to_num(exons[:, indices_disc], nan=0.0)
        X_proc.append(np.hstack([sc, disc]).astype(np.float32))

    return X_proc, imputer_stats, scaler


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    set_seed(seed)

    logger.info('=' * 93)
    logger.info('  TRAINING ExonRNN V7C + PLATYPUS (V7C ARCHITECTURE + ornAna2 IN TRAIN) | SEED=%d', seed)
    logger.info('=' * 93)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load baseline placental training data + FULL Platypus data
    d_train = np.load(f'{CACHE_DIR}/train_cache.npz', allow_pickle=True)
    d_orn = np.load(f'{CACHE_DIR}/val_ornAna2_cache.npz', allow_pickle=True)
    fn = list(d_train['feature_names'])

    X_tr_raw = list(d_train['X_exons']) + list(d_orn['X_exons'])
    y_tr = np.concatenate([d_train['y'], d_orn['y']])
    sp_tr = list(d_train['species']) + ['ornAna2'] * len(d_orn['y'])

    logger.info('Training pool size: %d pairs (Placental: %d, Platypus: %d)',
                len(y_tr), len(d_train['y']), len(d_orn['y']))

    X_tr_proc, imputer_stats, scaler = preprocess_npz(X_tr_raw, fn, fit=True)
    exon_dim = X_tr_proc[0].shape[1]
    veto_idx = V7C_FEATS_CONT.index(VETO_FEAT)

    model = ExonRNN_V7C(exon_dim=exon_dim, veto_idx=veto_idx).to(device)

    sampler = get_species_sampler(sp_tr, y_tr, seed=seed, neg_oversample=3.0)
    train_loader = DataLoader(
        ExonDatasetV7C(X_tr_proc, y_tr),
        batch_size=256, sampler=sampler, collate_fn=collate_fn_v7c,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )

    criterion = FocalLoss(alpha_pos=0.15, alpha_neg=0.85, gamma_pos=1.0, gamma_neg=5.0)
    lambda_entropy = 0.01
    lambda_proto = 0.5

    hidden_dim = 256
    c_ortholog = F.normalize(torch.randn(hidden_dim, device=device), dim=0)
    c_paralog  = F.normalize(torch.randn(hidden_dim, device=device), dim=0)
    ema_momentum = 0.99

    logger.info('Training V7C + Platypus model...')
    start_t = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_b, lengths_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            logits, attended, entropy = model(X_b, lengths_b)
            loss_focal = criterion(logits, y_b)

            with torch.no_grad():
                ortho_mask = y_b == 1
                para_mask  = y_b == 0
                if ortho_mask.any():
                    c_ortholog = ema_momentum * c_ortholog + (1 - ema_momentum) * attended[ortho_mask].mean(dim=0)
                if para_mask.any():
                    c_paralog  = ema_momentum * c_paralog + (1 - ema_momentum) * attended[para_mask].mean(dim=0)

            c_ortho_n = F.normalize(c_ortholog.unsqueeze(0), dim=1).expand(attended.size(0), -1)
            c_para_n  = F.normalize(c_paralog.unsqueeze(0), dim=1).expand(attended.size(0), -1)
            h_n = F.normalize(attended, dim=1)
            sim_ortho = (h_n * c_ortho_n).sum(dim=1)
            sim_para  = (h_n * c_para_n).sum(dim=1)
            logits_proto = torch.stack([sim_para, sim_ortho], dim=1)
            loss_proto = F.cross_entropy(logits_proto / 0.05, y_b.long())

            proto_sim = (c_ortho_n[0] * c_para_n[0]).sum()
            loss_orthogonal = torch.relu(proto_sim - 0.0)

            loss = loss_focal + lambda_entropy * (-entropy) + lambda_proto * loss_proto + 0.5 * loss_orthogonal
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info('Epoch %02d/%d | Loss: %.4f | LR: %.6f',
                        epoch + 1, epochs, total_loss / len(train_loader), scheduler.get_last_lr()[0])

    elapsed = time.time() - start_t
    logger.info('Trained V7C + Platypus in %.1fs', elapsed)

    save_path = f'/mnt/c/work/profiler_testing/dl_model/dl_orthology_model_v7c_platypus.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'exon_dim': exon_dim,
        'veto_idx': veto_idx,
        'exon_feats': V7C_FEATS_CONT,
        'exon_imputer_statistics': imputer_stats,
        'exon_scaler_mean': scaler.mean_,
        'exon_scaler_scale': scaler.scale_,
        'best_threshold': 0.50,
        'seed': seed,
    }, save_path)
    logger.info('Saved checkpoint -> %s', save_path)

    # Evaluate held-out validation species (EXCLUDING Platypus)
    logger.info('\n' + '=' * 93)
    logger.info('  EVALUATION ON ALL HELD-OUT VALIDATION SPECIES (EXCLUDING PLATYPUS)')
    logger.info('=' * 93)

    model.eval()
    for threshold in [0.30, 0.40, 0.50]:
        print(f'\n--- Threshold T = {threshold:.2f} ---')
        records = []
        for sp in EVAL_SPECIES:
            d_val = np.load(f'{CACHE_DIR}/val_{sp}_cache.npz', allow_pickle=True)
            X_val_raw = d_val['X_exons']
            y_val = d_val['y']

            X_val_proc, _, _ = preprocess_npz(X_val_raw, fn, imputer_stats=imputer_stats, scaler=scaler, fit=False)

            v_loader = DataLoader(
                ExonDatasetV7C(X_val_proc, y_val),
                batch_size=256, shuffle=False, collate_fn=collate_fn_v7c
            )

            all_probs = []
            with torch.no_grad():
                for X_b, lengths_b, _ in v_loader:
                    X_b = X_b.to(device)
                    logits, _, _ = model(X_b, lengths_b)
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy())

            probs_arr = np.array(all_probs)
            preds = (probs_arr >= threshold).astype(int)
            y_true = np.array(y_val).astype(int)

            tn, fp, fn_c, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            f1 = f1_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)

            records.append({
                'Species': sp, 'N': len(y_true), 'TP': tp, 'TN': tn,
                'FP': fp, 'FN': fn_c,
                'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4)
            })

        print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
