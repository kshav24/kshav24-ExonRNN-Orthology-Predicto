import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '/mnt/c/work/profiler_testing/dl_model')
from train_rnn_v6 import process_dataframe, EXON_FEATS

print("Loading data...")
df = pd.read_csv('/mnt/c/work/profiler_testing/dl_model/test_cleaned.csv')
df = df.sample(n=min(len(df), 5000), random_state=42)

X_ex_list, _, _, _ = process_dataframe(df, 'test')

X_flat = []
for ex_seq in X_ex_list:
    cont_feats = ex_seq[:, :len(EXON_FEATS)]
    X_flat.append(np.nanmean(cont_feats, axis=0))
X = np.nan_to_num(np.array(X_flat), nan=0.0)

targets = ['kmer4_cosine', 'transition_ratio', 'four_fold_id', 'ka', 'ks', 'kaks_ratio', 'd4v']
indices = [EXON_FEATS.index(f) for f in targets]

print("\n--- Pearson Correlation Matrix (Distance Features) ---\n")
print(f"{'':>20}" + "".join([f"{f:>18}" for f in targets]))

corr = np.corrcoef(X[:, indices], rowvar=False)
for i, f1 in enumerate(targets):
    row = f"{f1:>20}"
    for j in range(len(targets)):
        row += f"{corr[i,j]:18.3f}"
    print(row)
