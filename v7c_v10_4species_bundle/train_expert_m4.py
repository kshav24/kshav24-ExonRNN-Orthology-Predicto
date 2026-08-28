#!/usr/bin/env python3
"""Expert M4: Deep-Time Invariant Sequence Classifier for Deep Divergence Outgroups (d > 0.65).
Key Innovations:
  1. Macroscopic Structural Constraint Gate (G_struct) that shifts embedding reliance from raw sequence matching
     onto Modulo-3 ORF preservation (I_ORF), spatial selection entropy H_S(omega), and polarity conservation.
  2. Multi-Divergence Stratified Training Pool to expose BiGRU weights to true 180-MYA divergence.
  3. Bounded Self-FiLM Envelope + Probabilistic CIDA (Gaussian NLL).
  4. Relaxed Distance-Adaptive ArcFace (AdaArcFace on S^255) preventing deep-time false negatives.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = '/mnt/c/work/profiler_testing/dl_model/cached_npz'
OUTPUT_CHECKPOINT = '/mnt/c/work/profiler_testing/dl_model/dl_expert_m4_deep_invariance.pth'

SELECTED_FEATS_CONT = [
    'kmer4_cosine', 'transition_ratio', 'rscu_distance', 'gc3_diff', 'four_fold_id',
    'grantham_mean', 'grantham_max', 'nonsyn_spatial_entropy', 'polarity_preserving_frac',
    'cpg_tpg_rate', 'period3_power', 'align_longest_ungapped_norm', 'aa_freq_cosine',
    'max_kaks_window', 'd4v', 'aa_blosum62_mean', 'aa_blosum62_negative_frac',
    'aa_longest_conserved_run_norm', 'aa_pairwise_coverage', 'pct_intact_orf',
    'query_internal_stop_fraction'
]
SELECTED_FEATS_DISCRETE = [
    'is_ks_near_zero', 'is_ks_saturated', 'is_kaks_var_missing', 'delta_exon', 'total_exons'
]
SELECTED_FEATS = SELECTED_FEATS_CONT + SELECTED_FEATS_DISCRETE
N_CONT = len(SELECTED_FEATS_CONT)
SKEWED = ['grantham_mean', 'max_kaks_window']

# Indices for macroscopic structural invariant vector V_macro
MACRO_FEATS = ['pct_intact_orf', 'nonsyn_spatial_entropy', 'aa_blosum62_mean', 'polarity_preserving_frac']


# ---------------------------------------------------------------------------
# 1. Structural Constraint Gate (G_struct) Module
# ---------------------------------------------------------------------------
class StructuralConstraintGate(nn.Module):
    """Dynamically gates sequence matching vs. macroscopic structural invariants:
    E_final = (1 - G_struct) * E_sequence + G_struct * phi(V_macro)
    As divergence d increases, G_struct increases reliance on ORF integrity, spatial entropy, and polarity.
    """
    def __init__(self, macro_dim=4, hidden_dim=256):
        super().__init__()
        self.macro_proj = nn.Sequential(
            nn.Linear(macro_dim + 1, 128),  # macro features + clock d
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(macro_dim + 1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, e_sequence, v_macro, d):
        # v_macro: (B, 4), d: (B, 1)
        macro_in = torch.cat([v_macro, d], dim=-1)
        g_struct = self.gate_net(macro_in)  # (B, 256) in [0, 1]
        e_macro = self.macro_proj(macro_in)  # (B, 256)
        e_final = (1.0 - g_struct) * e_sequence + g_struct * e_macro
        return e_final


# ---------------------------------------------------------------------------
# 2. Neural Architecture: Expert M4
# ---------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_weight):
        ctx.lambda_weight = lambda_weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_weight, None


class DistanceAdaptiveArcFaceHead(nn.Module):
    def __init__(self, in_features=256, num_classes=2, s=32.0):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, d=None, labels=None):
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(emb_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.training and labels is not None:
            theta = torch.acos(cos_theta)
            # m(d) = 0.15 * exp(-2.5 * d) + 0.08 (Genuinely relaxed margin for deep outgroups: m(0.88) ≈ 0.097)
            if d is not None:
                m = 0.15 * torch.exp(-2.5 * d.clamp(min=0.0, max=2.0)) + 0.08
                m = m.unsqueeze(1)
            else:
                m = 0.10

            target_logits = torch.cos(theta + m)
            one_hot = F.one_hot(labels.long(), num_classes=2)
            logits = cos_theta + one_hot * (target_logits - cos_theta)
            return logits * self.s
        else:
            return cos_theta


class ExonRNN_ExpertM4(nn.Module):
    def __init__(self, exon_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Sequence BiGRU
        self.rnn = nn.GRU(
            exon_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )

        # Bounded Self-FiLM Envelope
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2 * 2),
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Macroscopic Structural Constraint Gate
        self.struct_gate = StructuralConstraintGate(macro_dim=len(MACRO_FEATS), hidden_dim=hidden_dim * 2)

        # Probabilistic CIDA (Outputs mu_D and log_sigma_D^2)
        self.domain_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # AdaArcFace Head
        self.arcface = DistanceAdaptiveArcFaceHead(in_features=hidden_dim * 2, num_classes=2, s=32.0)

    def forward(self, x, d, lengths, v_macro, labels=None, lambda_adv=0.10):
        lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.cpu()
        lengths_dev = lengths_cpu.to(x.device)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        d_input = d.unsqueeze(-1)
        film_raw = self.film_mlp(d_input)
        gamma_raw, beta_raw = torch.chunk(film_raw, 2, dim=-1)
        gamma = torch.tanh(gamma_raw) * 0.50
        beta = torch.tanh(beta_raw) * 1.00

        outputs = (1.0 + gamma.unsqueeze(1)) * outputs + beta.unsqueeze(1)
        mask = torch.arange(outputs.size(1), device=x.device).unsqueeze(0) < lengths_dev.unsqueeze(1)
        outputs = outputs * mask.unsqueeze(-1).float()

        # Attention Pooling
        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)

        eps = 1e-8
        max_entropy = torch.log(lengths_dev.float().clamp(min=2))
        entropy = (-torch.sum(weights * torch.log(weights + eps), dim=1) / max_entropy).mean()

        e_sequence = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (B, 256)

        # Apply Macroscopic Structural Constraint Gate
        e_final = self.struct_gate(e_sequence, v_macro, d_input)  # (B, 256)

        # Probabilistic CIDA (Gaussian NLL via GRL)
        reversed_emb = GradientReversalFunction.apply(e_final, lambda_adv)
        domain_params = self.domain_regressor(reversed_emb)

        logits = self.arcface(e_final, d, labels)
        return logits, domain_params, e_final, entropy


# ---------------------------------------------------------------------------
# 3. Multi-Divergence Dataset Loading & Training Routine
# ---------------------------------------------------------------------------
class ExonDatasetM4(Dataset):
    def __init__(self, X_exons, d_clocks, y, v_macros):
        self.X_exons = X_exons
        self.d_clocks = d_clocks
        self.y = y
        self.v_macros = v_macros

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_exons[idx]),
            torch.tensor(self.d_clocks[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.from_numpy(self.v_macros[idx]),
        )


def collate_fn_m4(batch):
    exons, d_clocks, labels, v_macros = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in exons])
    padded_exons = nn.utils.rnn.pad_sequence(exons, batch_first=True)
    return padded_exons, torch.stack(d_clocks), lengths, torch.stack(v_macros), torch.stack(labels)


def load_multi_divergence_training_data():
    def load_split(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        return data['X_exons'], data['d_clocks'], data['y'], data['species'], list(data['feature_names'])

    X_base, d_base, y_base, sp_base, fn = load_split(f'{CACHE_DIR}/train_cache.npz')
    X_mon, d_mon, y_mon, sp_mon, _ = load_split(f'{CACHE_DIR}/val_monDom5_cache.npz')
    X_orn, d_orn, y_orn, sp_orn, _ = load_split(f'{CACHE_DIR}/val_ornAna2_cache.npz')

    # 50% stratified split of Monodelphis and Ornithorhynchus for training
    np.random.seed(42)
    def get_50(X, d, y, sp):
        n = len(y)
        perm = np.random.permutation(n)
        half = n // 2
        idx = perm[:half]
        return [X[i] for i in idx], d[idx], y[idx], [sp[0]] * half

    X_mon_tr, d_mon_tr, y_mon_tr, sp_mon_tr = get_50(X_mon, d_mon, y_mon, sp_mon)
    X_orn_tr, d_orn_tr, y_orn_tr, sp_orn_tr = get_50(X_orn, d_orn, y_orn, sp_orn)

    # Pool Multi-Divergence Dataset
    X_pooled = list(X_base) + X_mon_tr + X_orn_tr
    d_pooled = np.concatenate([d_base, d_mon_tr, d_orn_tr])
    y_pooled = np.concatenate([y_base, y_mon_tr, y_orn_tr])
    sp_pooled = list(sp_base) + sp_mon_tr + sp_orn_tr

    # Filter features
    indices = [fn.index(f) for f in SELECTED_FEATS]
    macro_indices_in_cont = [SELECTED_FEATS_CONT.index(f) for f in MACRO_FEATS]

    X_filtered = []
    for exons in X_pooled:
        sub = exons[:, indices].copy()
        for s_feat in SKEWED:
            s_idx = SELECTED_FEATS.index(s_feat)
            sub[:, s_idx] = np.log1p(np.maximum(0.0, np.nan_to_num(sub[:, s_idx], nan=0.0)))
        X_filtered.append(sub)

    # Impute and Scale ONLY Continuous Features (keep discrete/binary unscaled)
    all_exons = np.vstack(X_filtered)
    cont_part = all_exons[:, :N_CONT]
    imputer = SimpleImputer(strategy='median')
    imputer.fit(cont_part)
    scaler = StandardScaler()
    scaler.fit(imputer.transform(cont_part))

    # Compute v_macros using imputer medians for NaN handling
    macro_medians = imputer.statistics_[macro_indices_in_cont]

    v_macros = []
    for exons in X_filtered:
        sub_macro = exons[:, macro_indices_in_cont].copy()
        for j in range(len(macro_indices_in_cont)):
            nan_mask = np.isnan(sub_macro[:, j])
            sub_macro[nan_mask, j] = macro_medians[j]
        m_macro = np.mean(sub_macro, axis=0)
        v_macros.append(m_macro.astype(np.float32))

    X_scaled = []
    for ex in X_filtered:
        cont_imp = imputer.transform(ex[:, :N_CONT])
        cont_sc = scaler.transform(cont_imp)
        disc = np.nan_to_num(ex[:, N_CONT:], nan=0.0)
        combined = np.hstack([cont_sc, disc]).astype(np.float32)
        X_scaled.append(combined)

    v_macros_arr = np.array(v_macros)
    return X_scaled, d_pooled, y_pooled, sp_pooled, v_macros_arr, imputer, scaler, indices, macro_indices_in_cont


def train_expert_m4(epochs=25, batch_size=256, lr=1e-3):
    logger.info('=== Training Expert M4 (Deep-Time Invariant Outgroup Specialist) ===')
    X_tr, d_tr, y_tr, sp_tr, v_macros, imputer, scaler, indices, macro_indices = load_multi_divergence_training_data()
    exon_dim = X_tr[0].shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExonRNN_ExpertM4(exon_dim=exon_dim, hidden_dim=128).to(device)

    # Balanced Sampler
    combined = [f"{sp}_{int(lbl)}" for sp, lbl in zip(sp_tr, y_tr)]
    class_counts = pd.Series(combined).value_counts().to_dict()
    weights = [3.0 / class_counts[c] if c.endswith('_0') else 1.0 / class_counts[c] for c in combined]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

    loader = DataLoader(
        ExonDatasetM4(X_tr, d_tr, y_tr, v_macros),
        batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_m4,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
    loss_ce = nn.CrossEntropyLoss()

    start_t = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_b, d_b, lengths_b, v_mac_b, y_b in loader:
            X_b, d_b, v_mac_b, y_b = X_b.to(device), d_b.to(device), v_mac_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            logits, domain_params, _, entropy = model(X_b, d_b, lengths_b, v_mac_b, labels=y_b, lambda_adv=0.10)
            l_cls = loss_ce(logits, y_b.long())

            mu_d = domain_params[:, 0]
            log_var_d = domain_params[:, 1].clamp(min=-5.0, max=5.0)
            l_pcida = (0.5 * torch.exp(-log_var_d) * (d_b - mu_d)**2 + 0.5 * log_var_d).mean()

            loss = l_cls + 0.10 * l_pcida + 0.01 * (-entropy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        logger.info(f'Epoch {epoch+1:02d}/{epochs} | Total Loss: {total_loss/len(loader):.4f}')

    elapsed = time.time() - start_t
    logger.info(f'Trained Expert M4 in {elapsed:.1f}s')

    # Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'exon_dim': exon_dim,
        'n_cont': N_CONT,
        'selected_features_cont': SELECTED_FEATS_CONT,
        'selected_features_discrete': SELECTED_FEATS_DISCRETE,
        'selected_features': SELECTED_FEATS,
        'macro_features': MACRO_FEATS,
        'imputer_statistics': imputer.statistics_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'expert_role': 'Deep-Time Invariant Outgroup Specialist (d > 0.65)',
    }
    torch.save(checkpoint, OUTPUT_CHECKPOINT)
    logger.info(f'Saved Expert M4 checkpoint -> {OUTPUT_CHECKPOINT}')


if __name__ == '__main__':
    train_expert_m4()
