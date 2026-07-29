#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

DL_MODEL_DIR = "/mnt/c/work/profiler_testing/dl_model"
USED_SCRIPT_DIR = "/mnt/c/work/profiler_testing/training_data_pipeline"

sys.path.insert(0, DL_MODEL_DIR)
from train_rnn_v7 import (
    ExonDataset, ExonRNN, apply_scalers, collate_fn, process_dataframe, EXON_FEATS
)

def load_mappings():
    ensg_enst_map = {}
    with open(f"{USED_SCRIPT_DIR}/ensg_to_enst.tsv", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ensg_enst_map[parts[0]] = parts[1]

    para_to_ortho = {}
    with open(f"{USED_SCRIPT_DIR}/ortholog_paralog_pair_negatives.txt", "r") as f:
        for line in f:
            parts = line.strip().split('-')
            if len(parts) == 2:
                o_ensg, p_ensg = parts[0].strip(), parts[1].strip()
                o_enst, p_enst = ensg_enst_map.get(o_ensg), ensg_enst_map.get(p_ensg)
                if o_enst and p_enst:
                    para_to_ortho[p_enst] = o_enst
    return para_to_ortho

def compute_relative_features(group):
    # If the group only has 1 candidate or all NaNs, handle safely
    group = group.copy()
    
    d4v = group['d4v'].values
    kmer = group['kmer'].values
    
    group['candidate_count'] = len(group)
    
    # Ranks (handle NaNs by giving them worst rank)
    group['d4v_rank'] = group['d4v'].rank(method='min', ascending=True, na_option='bottom')
    group['kmer_rank'] = group['kmer'].rank(method='min', ascending=False, na_option='bottom')
    
    # Disagreement
    group['d4v_kmer_disagreement'] = group['d4v_rank'] - group['kmer_rank']
    
    # Deltas
    min_d4v = np.nanmin(d4v) if not np.isnan(d4v).all() else np.nan
    max_kmer = np.nanmax(kmer) if not np.isnan(kmer).all() else np.nan
    
    group['delta_d4v_best'] = d4v - min_d4v
    group['delta_kmer_best'] = max_kmer - kmer
    
    # Z-scores (using MAD)
    med_d4v = np.nanmedian(d4v)
    mad_d4v = np.nanmedian(np.abs(d4v - med_d4v))
    group['z_d4v'] = (d4v - med_d4v) / (mad_d4v + 1e-6)
    
    med_kmer = np.nanmedian(kmer)
    mad_kmer = np.nanmedian(np.abs(kmer - med_kmer))
    group['z_kmer'] = (kmer - med_kmer) / (mad_kmer + 1e-6)
    
    # Pareto frontier flag
    # A candidate is Pareto optimal if no other candidate has BOTH lower d4v AND higher kmer
    frontier = []
    for i in range(len(group)):
        d_i = d4v[i]
        k_i = kmer[i]
        if np.isnan(d_i) or np.isnan(k_i):
            frontier.append(0)
            continue
            
        dominated = False
        for j in range(len(group)):
            if i == j: continue
            d_j = d4v[j]
            k_j = kmer[j]
            if np.isnan(d_j) or np.isnan(k_j): continue
            
            # If j is strictly better than i on both axes
            if (d_j <= d_i and k_j >= k_i) and (d_j < d_i or k_j > k_i):
                dominated = True
                break
        frontier.append(0 if dominated else 1)
        
    group['frontier_flag'] = frontier
    
    return group

def generate_v8_dataset(csv_name, model, exon_scaler, global_scaler, para_to_ortho, device):
    print(f"\nProcessing {csv_name}...")
    df = pd.read_csv(f"{DL_MODEL_DIR}/{csv_name}")
    
    print("1. Extracting base features...")
    x_ex, x_g, y, species_list, _ = process_dataframe(df, csv_name)
    
    # Extract raw d4v and kmer for the relative features BEFORE scaling
    d4v_idx = EXON_FEATS.index('d4v')
    kmer_idx = EXON_FEATS.index('kmer4_cosine')
    
    raw_d4v, raw_kmer = [], []
    for ex in x_ex:
        raw_d4v.append(np.nanmean(ex[:, d4v_idx]))
        raw_kmer.append(np.nanmean(ex[:, kmer_idx]))
        
    # Get group IDs (we must exactly match the groupby order used in train_rnn_v7)
    grouped = df.groupby(['transcript_id', 'species'], sort=False)
    tids = [tid for (tid, sp), grp in grouped]
    
    print("2. Running V7 inference to get baseline scores...")
    x_ex_scaled, x_g_scaled = apply_scalers(x_ex, x_g, exon_scaler, global_scaler)
    loader = DataLoader(ExonDataset(x_ex_scaled, x_g_scaled, y), batch_size=256, shuffle=False, collate_fn=collate_fn)
    
    scores = []
    with torch.no_grad():
        for batch_x_ex, lengths, batch_x_g, _ in loader:
            logits, _ = model(batch_x_ex.to(device), lengths, batch_x_g.to(device))
            scores.extend(logits.squeeze(-1).cpu().numpy()) # RAW logits, not sigmoid
            
    print("3. Building Candidate Sets...")
    rows = []
    for i in range(len(tids)):
        tid = tids[i]
        sp = species_list[i]
        label = y[i]
        
        if label == 1.0:
            set_id = f"{tid}_{sp}"
        else:
            ortho_tid = para_to_ortho.get(tid)
            if ortho_tid is None:
                continue # Skip unmapped paralogs
            set_id = f"{ortho_tid}_{sp}"
            
        rows.append({
            'candidate_set_id': set_id,
            'transcript_id': tid,
            'species': sp,
            'label': label,
            'v7_logit': scores[i],
            'd4v': raw_d4v[i],
            'kmer': raw_kmer[i]
        })
        
    set_df = pd.DataFrame(rows)
    print(f"   Created {len(set_df)} valid candidate rows.")
    
    print("4. Computing relative evolutionary features...")
    set_df = set_df.groupby('candidate_set_id', group_keys=False).apply(compute_relative_features)
    
    # Save the flattened V8 dataset
    out_name = f"v8_features_{csv_name}"
    set_df.to_csv(f"{DL_MODEL_DIR}/{out_name}", index=False)
    print(f"Saved {out_name}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading V7 Model...")
    
    checkpoint = torch.load("/mnt/c/work/profiler_testing/dl_orthology_model_v7.pth", map_location=device, weights_only=False)
    model = ExonRNN(exon_dim=checkpoint["exon_dim"], global_dim=checkpoint["global_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    exon_scaler = StandardScaler()
    exon_scaler.mean_ = checkpoint["exon_scaler_mean"]
    exon_scaler.scale_ = checkpoint["exon_scaler_scale"]
    
    global_scaler = StandardScaler()
    global_scaler.mean_ = checkpoint["global_scaler_mean"]
    global_scaler.scale_ = checkpoint["global_scaler_scale"]
    
    para_to_ortho = load_mappings()
    
    generate_v8_dataset("test_cleaned.csv", model, exon_scaler, global_scaler, para_to_ortho, device)
    generate_v8_dataset("train_cleaned.csv", model, exon_scaler, global_scaler, para_to_ortho, device)

if __name__ == "__main__":
    main()
