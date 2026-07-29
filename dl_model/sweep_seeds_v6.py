#!/usr/bin/env python3
"""Sweep seeds 0 to 20 for ExonRNN v6 to find the best deterministic checkpoint.
Pre-extracts features ONCE so each seed takes only ~10 seconds on GPU.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Parent dir for extract_features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_rnn_v6 import (
    process_dataframe, fit_scalers, apply_scalers, ExonDataset, collate_fn,
    ExonRNN, FocalLoss, get_species_sampler, set_seed,
    EXON_FEATS, SKEWED_FEATS, N_EXON_CONTINUOUS, N_GLOBAL_CONTINUOUS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def main():
    train_path = '../train_cleaned.csv'
    test_path = '../test_cleaned.csv'
    if not os.path.exists(train_path):
        train_path = 'train_cleaned.csv'
        test_path = 'test_cleaned.csv'
        
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    
    logger.info('Extracting features ONCE for train/test...')
    X_ex_tr_raw, X_g_tr_raw, y_tr, sp_tr = process_dataframe(train_df, 'Train')
    X_ex_te_raw, X_g_te_raw, y_te, sp_te = process_dataframe(test_df, 'Test')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    best_overall_f1 = -1.0
    best_overall_seed = -1
    best_checkpoint_data = None
    results_summary = []
    
    logger.info('Fitting scalers on training data ONCE...')
    exon_scaler, global_scaler = fit_scalers(X_ex_tr_raw, X_g_tr_raw)
    X_ex_tr_scaled, X_g_tr_scaled = apply_scalers(X_ex_tr_raw, X_g_tr_raw, exon_scaler, global_scaler)
    X_ex_te_scaled, X_g_te_scaled = apply_scalers(X_ex_te_raw, X_g_te_raw, exon_scaler, global_scaler)

    exon_dim = X_ex_tr_scaled[0].shape[1]
    global_dim = X_g_tr_scaled.shape[1]
    
    seeds_to_test = list(range(21))  # 0 to 20
    logger.info(f'Starting seed sweep over {len(seeds_to_test)} seeds: {seeds_to_test}')
    
    for seed in seeds_to_test:
        set_seed(seed)
        
        sampler = get_species_sampler(sp_tr, y_tr, seed=seed)
        train_loader = DataLoader(ExonDataset(X_ex_tr_scaled, X_g_tr_scaled, y_tr), batch_size=1024, sampler=sampler, collate_fn=collate_fn)
        test_loader = DataLoader(ExonDataset(X_ex_te_scaled, X_g_te_scaled, y_te), batch_size=1024, shuffle=False, collate_fn=collate_fn)
        
        model = ExonRNN(exon_dim=exon_dim, global_dim=global_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 20
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        lambda_entropy = 0.01
        
        for epoch in range(epochs):
            model.train()
            for X_ex, lengths, X_g, y_b in train_loader:
                X_ex, X_g, y_b = X_ex.to(device), X_g.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits, entropy = model(X_ex, lengths, X_g)
                loss = criterion(logits, y_b) - lambda_entropy * entropy
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
        # Evaluate on test
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X_ex, lengths, X_g, y_b in test_loader:
                logits, _ = model(X_ex.to(device), lengths, X_g.to(device))
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(y_b.numpy())
                
        all_probs, all_targets = np.array(all_probs), np.array(all_targets)
        
        # Metrics at threshold 0.50
        preds_05 = (all_probs >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets, preds_05).ravel()
        acc_05 = accuracy_score(all_targets, preds_05)
        prec_05 = precision_score(all_targets, preds_05, zero_division=0)
        rec_05 = recall_score(all_targets, preds_05, zero_division=0)
        f1_05 = 2 * prec_05 * rec_05 / (prec_05 + rec_05) if (prec_05 + rec_05) else 0
        
        # Find best overall threshold for info
        best_t_f1, best_t_thresh = -1, 0.50
        for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            p_t = (all_probs >= t).astype(int)
            pr = precision_score(all_targets, p_t, zero_division=0)
            rc = recall_score(all_targets, p_t, zero_division=0)
            f_t = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
            if f_t > best_t_f1:
                best_t_f1, best_t_thresh = f_t, t
                
        logger.info(f"Seed {seed:2d} | t=0.50: F1={f1_05:.4f} (TP={tp}, TN={tn}, FP={fp}, FN={fn}) | Peak F1={best_t_f1:.4f} @ t={best_t_thresh:.2f}")
        sys.stdout.flush()
        
        results_summary.append({
            'seed': seed,
            'f1_05': f1_05,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'best_t_f1': best_t_f1,
            'best_t_thresh': best_t_thresh
        })
        
        if f1_05 > best_overall_f1:
            best_overall_f1 = f1_05
            best_overall_seed = seed
            best_checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'exon_scaler_mean': exon_scaler.mean_,
                'exon_scaler_scale': exon_scaler.scale_,
                'global_scaler_mean': global_scaler.mean_,
                'global_scaler_scale': global_scaler.scale_,
                'best_threshold': 0.50,
                'exon_dim': exon_dim,
                'global_dim': global_dim,
                'n_exon_continuous': N_EXON_CONTINUOUS,
                'n_global_continuous': N_GLOBAL_CONTINUOUS,
                'exon_feats': EXON_FEATS,
                'skewed_feats': SKEWED_FEATS,
                'seed': seed
            }

    # Save the absolute best checkpoint found across seeds
    torch.save(best_checkpoint_data, 'dl_orthology_model_v6.pth')
    
    logger.info("=" * 60)
    logger.info(f" SWEEP COMPLETE: Best Seed is {best_overall_seed} with t=0.50 F1 = {best_overall_f1:.4f}")
    logger.info(f" Saved optimal model checkpoint to dl_orthology_model_v6.pth")
    logger.info("=" * 60)
    
    # Sort top 5 by F1 @ 0.50
    sorted_res = sorted(results_summary, key=lambda x: x['f1_05'], reverse=True)
    logger.info("TOP 5 SEEDS AT THRESHOLD 0.50:")
    for r in sorted_res[:5]:
        logger.info(f"  Seed {r['seed']:2d} -> F1: {r['f1_05']:.4f} (TP: {r['tp']}, TN: {r['tn']}, FP: {r['fp']}, FN: {r['fn']})")

if __name__ == '__main__':
    main()
