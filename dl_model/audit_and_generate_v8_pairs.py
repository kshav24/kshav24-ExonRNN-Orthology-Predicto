#!/usr/bin/env python3
"""Audit and Pairwise Feature Generator for V8 Ranking Model.

This script performs a rigorous data integrity audit on the candidate pairs
formed by linking existing cleaned CSVs with ensg_to_enst.tsv and
ortholog_paralog_pair_negatives.txt.

It verifies:
1. Mapping completeness and coverage
2. Group integrity (1 positive + competing paralogues per locus/species)
3. Exclusion accounting (missing features, unmapped IDs, label mismatches)
4. Pairwise ranking feature computation (ranks, deltas, z-scores, Pareto flags)
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

DL_MODEL_DIR = "/mnt/c/work/profiler_testing/dl_model"
USED_SCRIPT_DIR = "/mnt/c/work/profiler_testing/training_data_pipeline"

TEST_CSV = os.path.join(DL_MODEL_DIR, "test_cleaned.csv")
TRAIN_CSV = os.path.join(DL_MODEL_DIR, "train_cleaned.csv")
ENSG_ENST_TSV = os.path.join(USED_SCRIPT_DIR, "ensg_to_enst.tsv")
ORTHO_PARA_PAIRS = os.path.join(USED_SCRIPT_DIR, "ortholog_paralog_pair_negatives.txt")

sys.path.insert(0, DL_MODEL_DIR)
from train_rnn_v7 import process_dataframe, EXON_FEATS


def load_mappings():
    print("1. Loading identifier mappings...")
    ensg_enst_map = {}
    with open(ENSG_ENST_TSV, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ensg_enst_map[parts[0]] = parts[1]

    para_to_ortho = defaultdict(set)
    total_pair_lines = 0
    skipped_pair_lines = 0

    with open(ORTHO_PARA_PAIRS, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total_pair_lines += 1
            parts = line.split("-")
            if len(parts) != 2:
                skipped_pair_lines += 1
                continue
            o_ensg, p_ensg = parts[0].strip(), parts[1].strip()
            o_enst = ensg_enst_map.get(o_ensg)
            p_enst = ensg_enst_map.get(p_ensg)

            if o_enst and p_enst:
                para_to_ortho[p_enst].add(o_enst)
            else:
                skipped_pair_lines += 1

    print(f"   ENSG -> ENST entries: {len(ensg_enst_map):,}")
    print(f"   Paralog -> Ortholog ENST relations: {sum(len(v) for v in para_to_ortho.values()):,}")
    print(f"   Skipped/unmapped pair lines: {skipped_pair_lines:,} / {total_pair_lines:,}")
    return para_to_ortho


def audit_and_process_split(csv_path, split_name, para_to_ortho):
    print(f"\n========================================================")
    print(f" AUDITING AND PROCESSING SPLIT: {split_name}")
    print(f" File: {csv_path}")
    print(f"========================================================")

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    total_raw_rows = len(df)
    print(f"1. Raw input rows: {total_raw_rows:,}")

    # Extract exon features using process_dataframe
    print("2. Extracting per-transcript sequence features...")
    x_ex, x_g, y, species_list, depths = process_dataframe(df, split_name)

    d4v_idx = EXON_FEATS.index("d4v")
    kmer_idx = EXON_FEATS.index("kmer4_cosine")

    # Reconstruct grouped dictionary safely
    grouped = df.groupby(["transcript_id", "species"], sort=False)
    group_keys = [(tid, sp) for (tid, sp), _ in grouped]
    assert len(group_keys) == len(x_ex), "Mismatch between grouped keys and process_dataframe output!"

    feat_map = {}
    for (tid, sp), ex, label in zip(group_keys, x_ex, y):
        d4v_val = np.nanmean(ex[:, d4v_idx])
        kmer_val = np.nanmean(ex[:, kmer_idx])
        feat_map[(tid, sp)] = {
            "d4v": d4v_val,
            "kmer": kmer_val,
            "label": float(label),
        }

    print(f"   Stored unique (transcript_id, species) feature vectors: {len(feat_map):,}")

    # 3. Assign Candidate Set IDs
    print("3. Building Candidate Pairs/Sets...")
    candidate_rows = []
    unmapped_negatives = 0
    multi_ortho_negatives = 0

    for (tid, sp), fdict in feat_map.items():
        lbl = fdict["label"]
        if lbl == 1.0:
            # Positive Ortholog row defines its own set_id for this locus + species
            set_id = f"{tid}_{sp}"
            candidate_rows.append({
                "candidate_set_id": set_id,
                "transcript_id": tid,
                "species": sp,
                "label": 1.0,
                "d4v": fdict["d4v"],
                "kmer": fdict["kmer"],
                "is_ortholog": 1,
            })
        elif lbl == 0.0:
            # Negative Paralog row maps back to its target ortholog(s)
            target_orthos = para_to_ortho.get(tid)
            if not target_orthos:
                unmapped_negatives += 1
                continue
            
            if len(target_orthos) > 1:
                multi_ortho_negatives += 1

            for o_tid in target_orthos:
                set_id = f"{o_tid}_{sp}"
                candidate_rows.append({
                    "candidate_set_id": set_id,
                    "transcript_id": tid,
                    "species": sp,
                    "label": 0.0,
                    "d4v": fdict["d4v"],
                    "kmer": fdict["kmer"],
                    "is_ortholog": 0,
                })

    cand_df = pd.DataFrame(candidate_rows)

    # 4. Rigorous Audit Metrics
    print("\n--- AUDIT REPORT ---")
    print(f" Total rows in feature map: {len(feat_map):,}")
    print(f" Mapped candidate records: {len(cand_df):,}")
    print(f" Unmapped negative paralogs excluded: {unmapped_negatives:,}")
    print(f" Negatives mapping to >1 ortholog in pair file: {multi_ortho_negatives:,}")

    # Group statistics
    set_groups = cand_df.groupby("candidate_set_id")
    total_sets = set_groups.ngroups
    
    pos_counts = set_groups["is_ortholog"].sum()
    total_counts = set_groups.size()
    neg_counts = total_counts - pos_counts

    sets_with_0_pos = (pos_counts == 0).sum()
    sets_with_1_pos = (pos_counts == 1).sum()
    sets_with_gt1_pos = (pos_counts > 1).sum()
    sets_with_0_neg = (neg_counts == 0).sum()

    print(f" Total unique candidate sets: {total_sets:,}")
    print(f"   - Sets with exactly 1 positive ortholog: {sets_with_1_pos:,} ({sets_with_1_pos/total_sets:.1%})")
    print(f"   - Sets with 0 positive orthologs (orphan paralogs): {sets_with_0_pos:,}")
    print(f"   - Sets with >1 positive orthologs: {sets_with_gt1_pos:,}")
    print(f"   - Sets with 0 negative paralogs (singletons): {sets_with_0_neg:,}")
    print(f" Set size distribution (total items per set):")
    print(f"   - Min: {total_counts.min()}, Median: {total_counts.median()}, Max: {total_counts.max()}")

    # Filter down to valid decision pairs (sets that have at least 1 positive and at least 1 negative)
    valid_set_ids = pos_counts[(pos_counts >= 1) & (neg_counts >= 1)].index
    valid_df = cand_df[cand_df["candidate_set_id"].isin(valid_set_ids)].copy()

    print(f"\n Valid Pairwise Candidate Sets (1+ Ortho & 1+ Para): {len(valid_set_ids):,}")
    print(f" Total rows in valid pairwise sets: {len(valid_df):,}")

    # 5. Compute Within-Group Relative Pairwise Features
    print("\n5. Computing Within-Set Relative Features...")

    def compute_pairwise_features(group):
        group = group.copy()
        d4v = group["d4v"].values
        kmer = group["kmer"].values

        group["candidate_count"] = len(group)

        # Ranks (handling NaNs)
        group["d4v_rank"] = group["d4v"].rank(method="min", ascending=True, na_option="bottom")
        group["kmer_rank"] = group["kmer"].rank(method="min", ascending=False, na_option="bottom")
        group["d4v_kmer_disagreement"] = group["d4v_rank"] - group["kmer_rank"]

        # Deltas to best in set
        min_d4v = np.nanmin(d4v) if not np.isnan(d4v).all() else np.nan
        max_kmer = np.nanmax(kmer) if not np.isnan(kmer).all() else np.nan

        group["delta_d4v_best"] = d4v - min_d4v
        group["delta_kmer_best"] = max_kmer - kmer

        # Robust Z-scores using median and MAD
        med_d4v = np.nanmedian(d4v)
        mad_d4v = np.nanmedian(np.abs(d4v - med_d4v))
        group["z_d4v"] = (d4v - med_d4v) / (mad_d4v + 1e-6)

        med_kmer = np.nanmedian(kmer)
        mad_kmer = np.nanmedian(np.abs(kmer - med_kmer))
        group["z_kmer"] = (kmer - med_kmer) / (mad_kmer + 1e-6)

        # Pareto frontier flag
        frontier = []
        for i in range(len(group)):
            d_i, k_i = d4v[i], kmer[i]
            if np.isnan(d_i) or np.isnan(k_i):
                frontier.append(0)
                continue
            dominated = False
            for j in range(len(group)):
                if i == j:
                    continue
                d_j, k_j = d4v[j], kmer[j]
                if np.isnan(d_j) or np.isnan(k_j):
                    continue
                if (d_j <= d_i and k_j >= k_i) and (d_j < d_i or k_j > k_i):
                    dominated = True
                    break
            frontier.append(0 if dominated else 1)

        group["frontier_flag"] = frontier
        return group

    final_df = valid_df.groupby("candidate_set_id", group_keys=False).apply(compute_pairwise_features)

    out_csv = os.path.join(DL_MODEL_DIR, f"v8_pairwise_{split_name.lower()}.csv")
    final_df.to_csv(out_csv, index=False)
    print(f" Saved pairwise dataset to: {out_csv}")

    return final_df


def main():
    para_to_ortho = load_mappings()
    audit_and_process_split(TEST_CSV, "Test", para_to_ortho)
    audit_and_process_split(TRAIN_CSV, "Train", para_to_ortho)
    print("\n========================================================")
    print(" AUDIT AND DATASET CREATION COMPLETE.")
    print("========================================================")


if __name__ == "__main__":
    main()
