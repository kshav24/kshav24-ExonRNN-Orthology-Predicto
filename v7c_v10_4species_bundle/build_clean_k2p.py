#!/usr/bin/env python3
"""CLEAN rebuild from RAW CSVs (no cache trust).

Fixes two cache flaws:
  1. Label: majority-vote (mean>0.5) instead of first-row (order-dependent,
     skewed the cache to 94.7% ortholog).
  2. Clock: K2P distance computed per (transcript_id, species) group.

Reuses the exact per-exon feature extraction from build_v8_dataset_cache.py
(extract_features from extract_features_minimal.py) so features are identical,
but recomputes labels + clock from the raw sequences.

Outputs to clean_k2p/:
  train_clean.npz          (train_cleaned.csv, 4 placental species)
  val_<sp>_clean.npz       (6 validation species)
"""

import concurrent.futures
import logging
import os
import sys
import time
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/c/work/profiler_testing/dl_model")
from extract_features_minimal import FEATURE_ORDER, extract_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE = "/mnt/c/work/profiler_testing/dl_model"
RAW_TRAIN = "/mnt/c/work/profiler_testing/train_cleaned.csv"
RAW_VAL = "/mnt/c/work/profiler_testing/ensembl_validation/raw_validation_data_{sp}.csv"
DST = f"{BASE}/clean_k2p"
os.makedirs(DST, exist_ok=True)

ALL_RAW_FEATS = list(FEATURE_ORDER)
FEAT_NAMES = ALL_RAW_FEATS + [
    "is_ks_near_zero",
    "is_ks_saturated",
    "is_kaks_var_missing",
    "delta_exon",
    "total_exons",
]

# ---- K2P clock (same as recompute_dclock_k2p.py) ----
_MAP = np.full(256, 4, dtype=np.int8)
_MAP[ord("A")] = 0
_MAP[ord("C")] = 1
_MAP[ord("G")] = 2
_MAP[ord("T")] = 3


def seq_codes(s):
    return _MAP[np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)]


def k2p(r, q):
    valid = (r != 4) & (q != 4)
    S = int(valid.sum())
    if S < 3:
        return 0.0
    diff = valid & (r != q)
    is_trans = diff & (
        ((r == 0) & (q == 2))
        | ((r == 2) & (q == 0))
        | ((r == 1) & (q == 3))
        | ((r == 3) & (q == 1))
    )
    n_trans = int(is_trans.sum())
    n_transv = int(diff.sum()) - n_trans
    p = n_trans / (2.0 / 3.0 * S)
    qq = n_transv / (1.0 / 3.0 * S)
    if 2 * p + qq >= 1.0 or 2 * qq >= 1.0:
        return 2.5
    return float(min(-0.5 * np.log(1 - 2 * p - qq) - 0.5 * np.log(1 - 2 * qq), 2.5))


def process_transcript_group(group_data):
    """Fresh per-exon features + K2P clock + exact label for one (tid, sp, label) group."""
    tid, sp, expected_label, group = group_data
    if "exon_number" in group.columns:
        group = group.sort_values("exon_number")
    total_exons = (
        int(group["exon_number"].max())
        if "exon_number" in group.columns
        else len(group)
    )

    exon_feats_seq = []
    prev_exon_number = 0
    full_ref, full_query = [], []

    for _, row in group.iterrows():
        ref = str(row["ref_seq"]) if pd.notnull(row["ref_seq"]) else ""
        query = str(row["query_seq"]) if pd.notnull(row["query_seq"]) else ""
        full_ref.append(ref)
        full_query.append(query)
        current_exon_number = int(row.get("exon_number", 1))
        feats = extract_features(
            ref, query, exon_number=current_exon_number, total_exons=total_exons
        )
        delta_exon = float(current_exon_number - prev_exon_number)
        prev_exon_number = current_exon_number

        raw_feat = [float(feats.get(k, np.nan)) for k in ALL_RAW_FEATS]
        ks_val = feats.get("ks", np.nan)
        is_ks_near_zero = 1.0 if (not np.isnan(ks_val) and ks_val <= 1e-5) else 0.0
        is_ks_saturated = 1.0 if (not np.isnan(ks_val) and ks_val >= 9.9) else 0.0
        is_kaks_var_missing = (
            1.0 if np.isnan(feats.get("kaks_window_variance", np.nan)) else 0.0
        )
        raw_feat.extend([is_ks_near_zero, is_ks_saturated, is_kaks_var_missing])
        raw_feat.extend([delta_exon, float(total_exons)])
        exon_feats_seq.append(raw_feat)

    # K2P clock on concatenated ref/query for THIS (transcript, species, label) group
    d_clock = k2p(seq_codes("".join(full_ref)), seq_codes("".join(full_query)))
    label = int(expected_label)

    return (
        np.array(exon_feats_seq, dtype=np.float32),
        float(d_clock),
        int(label),
        str(sp),
        str(tid),
    )


def build(csv_path, output_npz_path, name):
    t0 = time.time()
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        grouped = df.groupby(["transcript_id", "species", "label"])
        group_data_list = [(tid, sp, lbl, grp) for (tid, sp, lbl), grp in grouped]
    else:
        grouped = df.groupby(["transcript_id", "species"])
        group_data_list = [(tid, sp, 0, grp) for (tid, sp), grp in grouped]
    logger.info(f"{name}: {len(group_data_list)} (transcript,species,label) groups")

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as ex:
        results = list(ex.map(process_transcript_group, group_data_list, chunksize=100))

    X_exons, d_clocks, y, species, transcript_ids = zip(*results, strict=True)
    np.savez_compressed(
        output_npz_path,
        X_exons=np.array(X_exons, dtype=object),
        d_clocks=np.array(d_clocks, dtype=np.float32),
        y=np.array(y, dtype=np.int32),
        species=np.array(species, dtype=object),
        transcript_ids=np.array(transcript_ids, dtype=object),
        feature_names=np.array(FEAT_NAMES),
    )
    y = np.array(y)
    logger.info(
        f"{name}: done in {time.time() - t0:.1f}s  orth={int(y.sum())} para={int(len(y) - y.sum())} "
        f"({y.mean() * 100:.1f}% orth)  clock mean={np.array(d_clocks).mean():.4f} -> {output_npz_path}"
    )


def main():
    build(RAW_TRAIN, f"{DST}/train_clean.npz", "train")
    for sp in ["rn6", "phyCat1", "myoLuc2", "bisBis1", "monDom5", "ornAna2"]:
        build(RAW_VAL.format(sp=sp), f"{DST}/val_{sp}_clean.npz", f"val_{sp}")
    logger.info("=== CLEAN K2P build done ===")


if __name__ == "__main__":
    main()
