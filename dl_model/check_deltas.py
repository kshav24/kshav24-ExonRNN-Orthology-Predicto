import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '/mnt/c/work/profiler_testing/dl_model')
from train_rnn_v6 import process_dataframe, EXON_FEATS

print("Loading test data...")
df = pd.read_csv('/mnt/c/work/profiler_testing/dl_model/test_cleaned.csv')
# Keep it relatively fast by sampling if necessary, but 10,000 should be quick
df = df.sample(n=min(len(df), 10000), random_state=42)

print("Extracting features (this takes a few seconds)...")
X_ex_list, _, y, _ = process_dataframe(df, 'Test Sample')

# Calculate mean feature values per transcript
X_flat = []
for ex_seq in X_ex_list:
    cont_feats = ex_seq[:, :len(EXON_FEATS)]
    X_flat.append(np.nanmean(cont_feats, axis=0))
X = np.nan_to_num(np.array(X_flat), nan=0.0)

d4v_idx = EXON_FEATS.index('d4v')
kmer_idx = EXON_FEATS.index('kmer4_cosine')

orthologs = X[y == 1.0]
paralogs = X[y == 0.0]

print(f"\nFound {len(orthologs)} Orthologs and {len(paralogs)} Paralogs in sample.")

bins = [0, 0.5, 1.5, 10.0]
labels = ["Recent (e.g. Mouse)", "Intermediate (e.g. Dog/Cow)", "Deep Divergence (e.g. Platypus)"]

print("\n--- Feature Separability by Evolutionary Depth ---")
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    
    # Filter Orthologs in this distance band
    ortho_band = orthologs[(orthologs[:, d4v_idx] >= low) & (orthologs[:, d4v_idx] < high)]
    # Filter Paralogs in this distance band
    para_band = paralogs[(paralogs[:, d4v_idx] >= low) & (paralogs[:, d4v_idx] < high)]
    
    if len(ortho_band) == 0 or len(para_band) == 0:
        continue
        
    print(f"\n[{labels[i]}] (d4v between {low} and {high})")
    print(f"  Orthologs in band : {len(ortho_band)}")
    print(f"  Paralogs in band  : {len(para_band)}")
    
    mean_ortho_d4v = ortho_band[:, d4v_idx].mean()
    mean_para_d4v = para_band[:, d4v_idx].mean()
    
    mean_ortho_kmer = ortho_band[:, kmer_idx].mean()
    mean_para_kmer = para_band[:, kmer_idx].mean()
    
    print(f"  -> Avg Ortholog d4v: {mean_ortho_d4v:.3f} | Avg Paralog d4v: {mean_para_d4v:.3f}")
    print(f"  -> Avg Ortholog Kmer: {mean_ortho_kmer:.3f} | Avg Paralog Kmer: {mean_para_kmer:.3f}")
