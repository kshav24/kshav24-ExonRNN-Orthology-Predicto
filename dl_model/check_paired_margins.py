#!/usr/bin/env python3
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

DL_MODEL_DIR = "/mnt/c/work/profiler_testing/dl_model"
USED_SCRIPT_DIR = "/mnt/c/work/profiler_testing/training_data_pipeline"

TEST_CSV = f"{DL_MODEL_DIR}/test_cleaned.csv"
ENSG_ENST_TSV = f"{USED_SCRIPT_DIR}/ensg_to_enst.tsv"
ORTHO_PARA_PAIRS = f"{USED_SCRIPT_DIR}/ortholog_paralog_pair_negatives.txt"

sys.path.insert(0, DL_MODEL_DIR)
from train_rnn_v6 import process_group, EXON_FEATS


def find_locus_columns(df):
    """Return plausible target-locus columns if they exist in the CSV."""
    exact_candidates = [
        "target_locus_id",
        "locus_id",
        "target_gene_id",
        "target_ensg",
        "target_transcript_id",
        "target_enst",
        "ortholog_id",
        "ortholog_ensg",
        "ortholog_enst",
        "target_chrom",
        "target_chr",
        "chromosome",
        "chrom",
        "target_start",
        "target_end",
        "start",
        "end",
    ]

    present = [col for col in exact_candidates if col in df.columns]

    if present:
        preferred_groups = [
            ["target_locus_id"],
            ["locus_id"],
            ["target_gene_id"],
            ["target_ensg"],
            ["target_transcript_id"],
            ["target_enst"],
            ["ortholog_id"],
            ["ortholog_ensg"],
            ["ortholog_enst"],
            ["target_chrom", "target_start", "target_end"],
            ["target_chr", "target_start", "target_end"],
            ["chromosome", "start", "end"],
            ["chrom", "start", "end"],
        ]
        for cols in preferred_groups:
            if all(col in df.columns for col in cols):
                return cols

    return []


def load_ensg_to_enst(path):
    mapping = {}

    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            ensg, enst = parts[0], parts[1]

            if ensg in mapping and mapping[ensg] != enst:
                print(
                    f"WARNING: ENSG {ensg} maps to multiple ENSTs; "
                    f"keeping latest ({enst}) at line {line_no}",
                    file=sys.stderr,
                )

            mapping[ensg] = enst

    return mapping


def load_paralog_ortholog_pairs(path, ensg_enst_map):
    pairs = defaultdict(set)
    skipped = 0

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("-")
            if len(parts) != 2:
                skipped += 1
                continue

            ensg_orth, ensg_para = parts[0].strip(), parts[1].strip()
            enst_orth = ensg_enst_map.get(ensg_orth)
            enst_para = ensg_enst_map.get(ensg_para)

            if enst_orth is None or enst_para is None:
                skipped += 1
                continue

            pairs[enst_para].add(enst_orth)

    return pairs, skipped


def safe_group_mean(X_exon, feature_index):
    """Mean feature value across exons, preserving invalid/missing values."""
    if X_exon is None:
        return np.nan

    X_exon = np.asarray(X_exon)

    if X_exon.ndim != 2 or X_exon.shape[0] == 0:
        return np.nan

    if X_exon.shape[1] <= feature_index:
        return np.nan

    values = X_exon[:, feature_index].astype(float, copy=False)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return np.nan

    return float(finite.mean())


def make_locus_key(row, locus_cols):
    """Build a hashable target-locus key from selected raw CSV columns."""
    if not locus_cols:
        return None

    values = tuple(row[col] for col in locus_cols)

    if any(pd.isna(value) for value in values):
        return None

    return values


def fmt_pct(numerator, denominator):
    if denominator == 0:
        return "NA"
    return f"{numerator / denominator:.1%}"


def print_band_summary(name, band, lower=None, upper=None):
    if upper is None:
        description = f"Ortholog d4v >= {lower}"
    else:
        description = f"Ortholog d4v in [{lower}, {upper})"

    print(f"\n[{name}] ({description})")
    print(f"  Complete triplets : {len(band)}")

    if len(band) == 0:
        return

    print(
        f"  Median m_d4v     : {band['m_d4v'].median():.4f} "
        f"(mean: {band['m_d4v'].mean():.4f})"
    )
    print(
        f"  Median m_kmer    : {band['m_kmer'].median():.4f} "
        f"(mean: {band['m_kmer'].mean():.4f})"
    )

    d4v_positive = (band["m_d4v"] > 0).sum()
    kmer_positive = (band["m_kmer"] > 0).sum()
    both_positive = ((band["m_d4v"] > 0) & (band["m_kmer"] > 0)).sum()

    d4v_ties = np.isclose(band["m_d4v"], 0.0).sum()
    kmer_ties = np.isclose(band["m_kmer"], 0.0).sum()

    print(f"  Fraction m_d4v > 0 : {d4v_positive / len(band):.1%}")
    print(f"  Fraction m_kmer > 0: {kmer_positive / len(band):.1%}")
    print(f"  Fraction both > 0  : {both_positive / len(band):.1%}")
    print(f"  Exact/tolerance d4v ties : {d4v_ties / len(band):.1%}")
    print(f"  Exact/tolerance kmer ties: {kmer_ties / len(band):.1%}")


def main():
    for path in [TEST_CSV, ENSG_ENST_TSV, ORTHO_PARA_PAIRS]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    if "d4v" not in EXON_FEATS:
        raise ValueError("'d4v' is absent from EXON_FEATS.")

    if "kmer4_cosine" not in EXON_FEATS:
        raise ValueError("'kmer4_cosine' is absent from EXON_FEATS.")

    d4v_idx = EXON_FEATS.index("d4v")
    kmer_idx = EXON_FEATS.index("kmer4_cosine")

    print("1. Loading mappings...")
    ensg_enst_map = load_ensg_to_enst(ENSG_ENST_TSV)
    pairs_ortho_para, skipped_pair_lines = load_paralog_ortholog_pairs(
        ORTHO_PARA_PAIRS,
        ensg_enst_map,
    )

    total_relations = sum(len(orthos) for orthos in pairs_ortho_para.values())
    multi_orth_paras = sum(
        len(orthos) > 1 for orthos in pairs_ortho_para.values()
    )

    print(f"   ENSG -> ENST mappings: {len(ensg_enst_map)}")
    print(f"   Unique paralogue ENSTs: {len(pairs_ortho_para)}")
    print(f"   Total para -> ortho relations: {total_relations}")
    print(f"   Paralogue ENSTs with >1 orthologue mapping: {multi_orth_paras}")
    print(f"   Skipped malformed/unmapped pair-file lines: {skipped_pair_lines}")

    print("\n2. Loading test data...")
    df = pd.read_csv(TEST_CSV)

    required_cols = {"transcript_id", "species"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"test_cleaned.csv lacks required columns: {missing_cols}")

    print(f"   Raw rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    locus_cols = find_locus_columns(df)
    if locus_cols:
        print(f"   Target-locus columns used: {locus_cols}")
        group_cols = ["transcript_id", "species", *locus_cols]
    else:
        print(
            "   WARNING: No target-locus column detected. "
            "Pairing will be by transcript_id + species only."
        )
        print(
            "   This is valid only if each transcript/species group represents "
            "one unique CESAR target locus."
        )
        group_cols = ["transcript_id", "species"]

    print("\n3. Building feature-extraction groups...")
    grouped = df.groupby(group_cols, sort=True, dropna=False)
    group_data_list = []

    for key, grp in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        tid = key[0]
        sp = key[1]
        locus_key = key[2:] if locus_cols else None

        group_data_list.append(
            {
                "tid": tid,
                "species": sp,
                "locus_key": locus_key,
                "group_df": grp,
            }
        )

    print(f"   Transcript/species/locus groups: {len(group_data_list):,}")

    worker_inputs = [
        (entry["tid"], entry["species"], entry["group_df"])
        for entry in group_data_list
    ]

    workers = min(8, max(1, cpu_count() - 1))
    print(f"   ProcessPoolExecutor workers: {workers}")
    print("   Extracting features...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_group, worker_inputs, chunksize=100))

    if len(results) != len(group_data_list):
        raise RuntimeError(
            f"Feature extraction returned {len(results)} results for "
            f"{len(group_data_list)} input groups."
        )

    print("   Feature extraction complete.")

    feat_dict = {}
    invalid_groups = 0
    species_mismatches = 0

    for entry, result in zip(group_data_list, results):
        tid = entry["tid"]
        sp = entry["species"]
        locus_key = entry["locus_key"]

        if result is None or len(result) != 4:
            invalid_groups += 1
            continue

        X_exon, X_global, label, out_sp = result

        if str(out_sp) != str(sp):
            species_mismatches += 1
            continue

        d4v = safe_group_mean(X_exon, d4v_idx)
        kmer = safe_group_mean(X_exon, kmer_idx)

        key = (tid, sp, locus_key)

        if key in feat_dict:
            raise RuntimeError(
                f"Duplicate feature key produced: {key}. "
                "Check grouping columns."
            )

        feat_dict[key] = {
            "d4v": d4v,
            "kmer": kmer,
            "label": float(label) if pd.notna(label) else np.nan,
        }

    print(f"   Usable feature groups stored: {len(feat_dict):,}")
    print(f"   Invalid worker results excluded: {invalid_groups}")
    print(f"   Species mismatches excluded: {species_mismatches}")

    print("\n4. Calculating paired margins...")

    available_species_loci = defaultdict(set)
    for tid, sp, locus_key in feat_dict:
        available_species_loci[tid].add((sp, locus_key))

    margins = []
    counters = defaultdict(int)

    for para_tid, ortho_tids in pairs_ortho_para.items():
        para_targets = available_species_loci.get(para_tid, set())

        for ortho_tid in ortho_tids:
            for sp, locus_key in para_targets:
                counters["candidate_pairs"] += 1

                para_key = (para_tid, sp, locus_key)
                ortho_key = (ortho_tid, sp, locus_key)

                if ortho_key not in feat_dict:
                    counters["missing_matched_orthologue"] += 1
                    continue

                counters["matched_pairs"] += 1

                para_feats = feat_dict[para_key]
                ortho_feats = feat_dict[ortho_key]

                if not (
                    np.isclose(ortho_feats["label"], 1.0)
                    and np.isclose(para_feats["label"], 0.0)
                ):
                    counters["label_mismatch"] += 1
                    continue

                if not (
                    np.isfinite(ortho_feats["d4v"])
                    and np.isfinite(para_feats["d4v"])
                    and np.isfinite(ortho_feats["kmer"])
                    and np.isfinite(para_feats["kmer"])
                ):
                    counters["missing_feature"] += 1
                    continue

                margins.append(
                    {
                        "species": sp,
                        "locus_key": locus_key,
                        "ortho_tid": ortho_tid,
                        "para_tid": para_tid,
                        "ortho_d4v": ortho_feats["d4v"],
                        "para_d4v": para_feats["d4v"],
                        "ortho_kmer": ortho_feats["kmer"],
                        "para_kmer": para_feats["kmer"],
                        "m_d4v": para_feats["d4v"] - ortho_feats["d4v"],
                        "m_kmer": ortho_feats["kmer"] - para_feats["kmer"],
                    }
                )

    margins_df = pd.DataFrame(margins)

    print(f"   Candidate para/ortho/species-locus comparisons: {counters['candidate_pairs']:,}")
    print(f"   Matched transcript/species/locus comparisons: {counters['matched_pairs']:,}")
    print(f"   Excluded: matched orthologue absent: {counters['missing_matched_orthologue']:,}")
    print(f"   Excluded: unexpected labels: {counters['label_mismatch']:,}")
    print(f"   Excluded: missing/non-finite d4v or k-mer: {counters['missing_feature']:,}")
    print(f"   Complete valid triplets: {len(margins_df):,}")

    if margins_df.empty:
        print("\nNo complete valid triplets were found. Stopping.")
        return

    print("\n5. Auditing d4v cap behaviour...")
    n_exact_10 = np.isclose(margins_df["ortho_d4v"], 10.0).sum()
    n_above_10 = (margins_df["ortho_d4v"] > 10.0).sum()
    n_below_0 = (margins_df["ortho_d4v"] < 0.0).sum()

    print(f"   Orthologue d4v exactly 10.0: {n_exact_10:,}")
    print(f"   Orthologue d4v > 10.0: {n_above_10:,}")
    print(f"   Orthologue d4v < 0.0: {n_below_0:,}")
    print(
        f"   Orthologue d4v range: "
        f"{margins_df['ortho_d4v'].min():.4f} to "
        f"{margins_df['ortho_d4v'].max():.4f}"
    )

    print("\n--- Paired Margins by Orthologue d4v ---")
    print("Positive m_d4v means the orthologue has lower d4v.")
    print("Positive m_kmer means the orthologue has higher k-mer similarity.")

    recent = margins_df[
        (margins_df["ortho_d4v"] >= 0.0)
        & (margins_df["ortho_d4v"] < 0.5)
    ]
    intermediate = margins_df[
        (margins_df["ortho_d4v"] >= 0.5)
        & (margins_df["ortho_d4v"] < 1.5)
    ]
    deep_resolved = margins_df[
        (margins_df["ortho_d4v"] >= 1.5)
        & (margins_df["ortho_d4v"] < 10.0)
    ]
    saturated_or_extreme = margins_df[
        margins_df["ortho_d4v"] >= 10.0
    ]

    print_band_summary("Recent", recent, 0.0, 0.5)
    print_band_summary("Intermediate", intermediate, 0.5, 1.5)
    print_band_summary("Deep-resolved", deep_resolved, 1.5, 10.0)
    print_band_summary(
        "Capped/saturated-or-extreme",
        saturated_or_extreme,
        lower=10.0,
        upper=None,
    )

    output_csv = f"{DL_MODEL_DIR}/paired_margins_test.csv"
    margins_df.to_csv(output_csv, index=False)

    print(f"\n6. Wrote complete per-triplet margins to:\n   {output_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
