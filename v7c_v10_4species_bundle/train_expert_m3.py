#!/usr/bin/env python3
"""Expert M3: High-Precision Sequence Classifier for Close/Moderate Divergence (d <= 0.65).
Key Innovations:
  1. Asymmetric Loss (ASL) with probability shift margin (m=0.05, gamma+=0, gamma-=4) to eliminate false positives.
  2. Bounded Self-FiLM Envelope (tanh bounds [-0.5, +0.5]) to stabilize sequence representations.
  3. Probabilistic CIDA (Gaussian NLL on mu_D, sigma_D^2) to drive mutual information I(E; d) -> 0.
  4. Distance-Adaptive ArcFace (AdaArcFace on S^255) with tight angular margin (m(d) approx 0.30 for close mammals).
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
OUTPUT_CHECKPOINT = '/mnt/c/work/profiler_testing/dl_model/dl_expert_m3_high_precision.pth'

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


# ---------------------------------------------------------------------------
# 1. Asymmetric Loss (ASL) for False Positive Elimination
# ---------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for extreme false positive suppression (Ridnik et al., ICCV 2021).
    - gamma_pos = 0.0: Disables focal decay on true orthologs (maintains strong BCE gradients).
    - gamma_neg = 4.0: Non-linear decay for easy negatives, focusing on hard duplicate paralogs.
    - clip_margin = 0.05: Completely zeros gradients for easy negative pairs (p <= 0.05).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip_margin=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits, targets):
        # logits: (B, 2) from ArcFace, targets: (B,) binary {0, 1}
        probs = F.softmax(logits, dim=-1)[:, 1]  # P(Ortholog = 1)
        targets = targets.float()

        # Positive loss (True Orthologs)
        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1.0 - probs) ** self.gamma_pos)

        # Negative loss with asymmetric probability margin shift (Paralogs)
        p_m = (probs - self.clip_margin).clamp(min=0.0)
        loss_neg = (1.0 - targets) * torch.log((1.0 - p_m).clamp(min=self.eps))
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (p_m ** self.gamma_neg)

        loss = -(loss_pos + loss_neg).mean()
        return loss


# ---------------------------------------------------------------------------
# 2. Neural Architecture: Expert M3
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
    """AdaArcFace on Unit Hypersphere S^255."""
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
            # m(d) = 0.20 * exp(-2.0 * d) + 0.10 (High margin for close mammals)
            if d is not None:
                m = 0.20 * torch.exp(-2.0 * d.clamp(min=0.0, max=2.0)) + 0.10
                m = m.unsqueeze(1)
            else:
                m = 0.30

            target_logits = torch.cos(theta + m)
            one_hot = F.one_hot(labels.long(), num_classes=2)
            logits = cos_theta + one_hot * (target_logits - cos_theta)
            return logits * self.s
        else:
            return cos_theta


class ExonRNN_ExpertM3(nn.Module):
    def __init__(self, exon_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Sequential BiGRU Encoder
        self.rnn = nn.GRU(
            exon_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )

        # Saturating Bounded Self-FiLM Envelope
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2 * 2),
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Probabilistic CIDA Domain Discriminator (Outputs mu_D and log_sigma_D^2)
        self.domain_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # AdaArcFace Metric Head
        self.arcface = DistanceAdaptiveArcFaceHead(in_features=hidden_dim * 2, num_classes=2, s=32.0)

    def forward(self, x, d, lengths, labels=None, lambda_adv=0.10):
        lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.cpu()
        lengths_dev = lengths_cpu.to(x.device)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Bounded FiLM Envelope: gamma in [-0.5, +0.5], beta in [-1.0, +1.0]
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

        attended = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)

        # Probabilistic CIDA (Gaussian NLL via GRL)
        reversed_emb = GradientReversalFunction.apply(attended, lambda_adv)
        domain_params = self.domain_regressor(reversed_emb)

        logits = self.arcface(attended, d, labels)
        return logits, domain_params, attended, entropy


# ---------------------------------------------------------------------------
# 3. Data Loading & Training Routine
# ---------------------------------------------------------------------------
class ExonDataset(Dataset):
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


def collate_fn(batch):
    exons, d_clocks, labels = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in exons])
    padded_exons = nn.utils.rnn.pad_sequence(exons, batch_first=True)
    return padded_exons, torch.stack(d_clocks), lengths, torch.stack(labels)


def load_and_preprocess_training_data():
    data = np.load(f'{CACHE_DIR}/train_cache.npz', allow_pickle=True)
    X_raw = data['X_exons']
    d_clocks = data['d_clocks']
    y = data['y']
    species = data['species']
    fn = list(data['feature_names'])

    indices = [fn.index(f) for f in SELECTED_FEATS]
    X_filtered = []
    for exons in X_raw:
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

    X_scaled = []
    for ex in X_filtered:
        cont_imp = imputer.transform(ex[:, :N_CONT])
        cont_sc = scaler.transform(cont_imp)
        disc = np.nan_to_num(ex[:, N_CONT:], nan=0.0)
        combined = np.hstack([cont_sc, disc]).astype(np.float32)
        X_scaled.append(combined)

    return X_scaled, d_clocks, y, species, imputer, scaler


def train_expert_m3(epochs=25, batch_size=256, lr=1e-3):
    logger.info('=== Training Expert M3 (High-Precision Mammalian Specialist) ===')
    X_tr, d_tr, y_tr, sp_tr, imputer, scaler = load_and_preprocess_training_data()
    exon_dim = X_tr[0].shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExonRNN_ExpertM3(exon_dim=exon_dim, hidden_dim=128).to(device)

    # Balanced Sampler
    combined = [f"{sp}_{int(lbl)}" for sp, lbl in zip(sp_tr, y_tr)]
    class_counts = pd.Series(combined).value_counts().to_dict()
    weights = [3.0 / class_counts[c] if c.endswith('_0') else 1.0 / class_counts[c] for c in combined]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

    loader = DataLoader(
        ExonDataset(X_tr, d_tr, y_tr),
        batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
    loss_asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip_margin=0.05)

    start_t = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_b, d_b, lengths_b, y_b in loader:
            X_b, d_b, y_b = X_b.to(device), d_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            logits, domain_params, _, entropy = model(X_b, d_b, lengths_b, labels=y_b, lambda_adv=0.10)
            l_cls = loss_asl(logits, y_b)

            # Probabilistic CIDA: Gaussian NLL
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
    logger.info(f'Trained Expert M3 in {elapsed:.1f}s')

    # Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'exon_dim': exon_dim,
        'n_cont': N_CONT,
        'selected_features_cont': SELECTED_FEATS_CONT,
        'selected_features_discrete': SELECTED_FEATS_DISCRETE,
        'selected_features': SELECTED_FEATS,
        'imputer_statistics': imputer.statistics_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'expert_role': 'High-Precision Mammalian Specialist (d <= 0.65)',
    }
    torch.save(checkpoint, OUTPUT_CHECKPOINT)
    logger.info(f'Saved Expert M3 checkpoint -> {OUTPUT_CHECKPOINT}')


if __name__ == '__main__':
    train_expert_m3()
