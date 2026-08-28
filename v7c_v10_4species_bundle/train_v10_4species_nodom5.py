#!/usr/bin/env python3
"""Train v10 on ONLY the 4 placental species (RAW_TRAIN), removing monDom5.

Sister script to train_v7c_4species_nodom5.py — same 4-species training data,
same evaluation harness, but trains the v10 model instead of v7c. This gives a
directly comparable v10 vs v7c comparison on identical 4-species training data.

Notes:
  - Training: Training_data/train.csv ONLY (4 species: mm10, bosTau8, equCab2,
    canFam3). monDom5 training data is excluded.
  - K2P fix wired in (v10 DOES use the external clock via FiLM).
  - Evaluation: standard 6 validation NPZs (rn6, phyCat1, myoLuc2, bisBis1,
    monDom5, ornAna2) via the shared tac.evaluate harness.
  - Checkpoint: dl_v10_4species_nodom5.pth (NEW name).

Usage:
  python3 train_v10_4species_nodom5.py
"""

import concurrent.futures
import importlib.util as _ilu
import logging
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/mnt/c/work/profiler_testing/dl_model")

# --- Wire in the FIXED textbook K2P clock (v10 uses the clock) --------------
_K2P_FIX_PATH = "/mnt/c/work/profiler_testing/k_2p_fix.py"
_spec = _ilu.spec_from_file_location("k_2p_fix", _K2P_FIX_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load k_2p_fix from {_K2P_FIX_PATH}")
_k2p_fix = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_k2p_fix)
k2p_textbook = _k2p_fix.k2p_textbook

import build_clean_k2p as bck  # noqa: E402
import train_all_clean as tac  # noqa: E402
import train_v10 as v10  # noqa: E402

bck.k2p = k2p_textbook  # monkey-patch: fixed formula now used at call time
from build_clean_k2p import FEAT_NAMES, process_transcript_group  # noqa: E402

# ------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_TRAIN = "/mnt/c/work/profiler_testing/Training_data/train.csv"


def build_train_inmemory(csv_paths):
    """Build (X_exons[39], d_clocks, y, species) in-memory from one or more raw CSVs."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    grouped = df.groupby(["transcript_id", "species"])
    group_data_list = []
    for _key, idx in grouped.indices.items():
        grp = df.loc[idx]
        first = grp.iloc[0]
        # majority-vote label computed here, passed in
        lbl = int(grp["label"].mean() > 0.5)
        group_data_list.append((first["transcript_id"], first["species"], lbl, grp))
    logger.info(f"raw train: {len(group_data_list)} (transcript,species) groups")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as ex:
        results = list(ex.map(process_transcript_group, group_data_list, chunksize=100))
    X_exons, d_clocks, y, species, _tids = zip(*results, strict=True)
    return (
        np.array(X_exons, dtype=object),
        np.array(d_clocks, dtype=np.float32),
        np.array(y, dtype=np.int32),
        [str(s) for s in species],
    )


def main():
    # Sanity check: confirm the patched k2p is in effect before any work.
    assert bck.k2p is k2p_textbook, "K2P fix not wired in!"

    # 4-species training ONLY (monDom5 excluded)
    X_exons, d_tr, y_tr, sp_tr = build_train_inmemory(RAW_TRAIN)
    feat_names = list(FEAT_NAMES)
    logger.info(f"training species: {sorted(set(sp_tr))}")

    # Macro medians from the raw training set (fill NaNs in macro features)
    macro_idx = [feat_names.index(f) for f in v10.MACRO_FEATS]
    allm = np.vstack([X_exons[i][:, macro_idx] for i in range(len(X_exons))])
    macro_medians = np.nan_to_num(np.nanmedian(allm, axis=0), nan=0.0)

    # 39 -> 22 reduction + v_macros (same as v10.load_split)
    X_tr, vm_tr = v10._reorder_and_skew(X_exons, feat_names, macro_medians)
    y_tr = y_tr.astype(np.float32)
    del X_exons, allm

    # Fit imputer/scaler strictly on the raw training set
    imputer, scaler = v10.fit_scalers(X_tr)
    X_tr = v10.apply_scalers(X_tr, imputer, scaler)

    d_mean = float(np.mean(d_tr))
    d_std = float(np.std(d_tr))
    if d_std < 1e-8:
        d_std = 1.0
    logger.info(
        f"RAW train n={len(y_tr)} orth={int(y_tr.sum())} ({y_tr.mean() * 100:.1f}%) "
        f"d_mean={d_mean:.4f} d_std={d_std:.4f}"
    )

    # Eval: standard 6 clean_k2p validation NPZs
    eval_data = {}
    for sp in tac.SPECIES:
        p = f"{tac.CLEAN}/val_{sp}_clean.npz"
        b = v10.load_split(p, p, macro_medians)
        X_eval = v10.apply_scalers(b.X_exons, imputer, scaler)
        eval_data[sp] = (X_eval, b.y, b.d, b.v_macros)

    # Train v10 on 4 species (no monDom5)
    logger.info("===== Training v10 on 4 SPECIES [no monDom5] =====")
    model = tac.TRAINERS["v10"](X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std)
    tac.MODEL["v10"] = model

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "d_mean": d_mean,
            "d_std": d_std,
            "imputer_statistics": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "macro_medians": macro_medians,
        },
        f"{tac.BASE}/dl_v10_4species_nodom5.pth",
    )
    logger.info("Saved checkpoint -> dl_v10_4species_nodom5.pth")

    rows = tac.evaluate("v10", d_mean, d_std, eval_data)
    tac.print_metrics(rows, "v10 4-species (no monDom5)")


if __name__ == "__main__":
    main()
