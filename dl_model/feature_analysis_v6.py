import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.insert(0, '/mnt/c/work/profiler_testing/dl_model')
from train_rnn_v6 import process_dataframe, EXON_FEATS

print("Loading test dataset for quick analysis...")
df = pd.read_csv('/mnt/c/work/profiler_testing/dl_model/test_cleaned.csv')
# Sample heavily for speed so we can get the answer fast
df = df.sample(n=min(len(df), 5000), random_state=42)

print("Extracting V6 Features (this takes a moment)...")
X_ex_list, _, y_list, _ = process_dataframe(df, 'test')

# Average the exon features across the transcript for global importance
X_flat = []
for ex_seq in X_ex_list:
    # Just take the continuous features
    cont_feats = ex_seq[:, :len(EXON_FEATS)]
    X_flat.append(np.nanmean(cont_feats, axis=0))

X = np.array(X_flat)
y = np.array(y_list)

# Handle NaNs from mean pool
X = np.nan_to_num(X, nan=0.0)

print("\n=== COLLINEARITY ANALYSIS (Pearson |r| > 0.8) ===")
corr_matrix = np.corrcoef(X, rowvar=False)
found_collinear = False
for i in range(len(EXON_FEATS)):
    for j in range(i+1, len(EXON_FEATS)):
        r = corr_matrix[i, j]
        if abs(r) > 0.8:
            print(f"HIGH COLLINEARITY: {EXON_FEATS[i]:>25}  <-->  {EXON_FEATS[j]:<25} (r = {r:.3f})")
            found_collinear = True
            
if not found_collinear:
    print("No highly collinear features found (|r| > 0.8).")

print("\n=== FEATURE IMPORTANCE (RandomForest Gini Importance) ===")
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
clf.fit(X, y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(len(EXON_FEATS)):
    print(f"{i+1:2d}. {importances[indices[i]]:.4f} weight | {EXON_FEATS[indices[i]]}")
