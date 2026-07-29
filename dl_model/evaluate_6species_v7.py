#!/usr/bin/env python3
import logging
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/c/work/profiler_testing/dl_model")

from train_rnn_v7 import (
    ExonDataset,
    ExonRNN,
    apply_scalers,
    collate_fn,
    process_dataframe,
)

MODEL_PATH = "/mnt/c/work/profiler_testing/dl_orthology_model_v7.pth"
VALIDATION_DIR = "/mnt/c/work/profiler_testing/ensembl_validation"

SPECIES_LIST = [
    "bisBis1",
    "monDom5",
    "myoLuc2",
    "ornAna2",
    "phyCat1",
    "rn6",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_model_and_scalers(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = ExonRNN(
        exon_dim=checkpoint["exon_dim"],
        global_dim=checkpoint["global_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    exon_scaler = StandardScaler()
    exon_scaler.mean_ = checkpoint["exon_scaler_mean"]
    exon_scaler.scale_ = checkpoint["exon_scaler_scale"]

    global_scaler = StandardScaler()
    global_scaler.mean_ = checkpoint["global_scaler_mean"]
    global_scaler.scale_ = checkpoint["global_scaler_scale"]

    threshold = float(checkpoint.get("best_threshold", 0.35))
    return model, exon_scaler, global_scaler, threshold


def evaluate_species(
    species,
    model,
    exon_scaler,
    global_scaler,
    threshold,
    device,
):
    csv_path = f"{VALIDATION_DIR}/raw_validation_data_{species}.csv"

    if not __import__("os").path.exists(csv_path):
        logger.error("Missing file: %s", csv_path)
        return None

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        logger.error("%s has no label column; skipping.", species)
        return None

    # FIXED: train_rnn_v7 process_dataframe returns 5 elements (depth is the 5th)
    x_ex, x_g, y, _, _ = process_dataframe(df, species)
    x_ex, x_g = apply_scalers(x_ex, x_g, exon_scaler, global_scaler)

    loader = DataLoader(
        ExonDataset(x_ex, x_g, y),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_x_ex, lengths, batch_x_g, batch_y in loader:
            logits, _ = model(
                batch_x_ex.to(device),
                lengths,
                batch_x_g.to(device),
            )
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    probs = np.asarray(all_probs)
    y_true = np.asarray(all_targets).astype(int)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    logger.info("=" * 68)
    logger.info("V7 evaluation: %s", species)
    logger.info("=" * 68)
    logger.info(
        "n=%d threshold=%.2f TP=%d TN=%d FP=%d FN=%d",
        len(y_true), threshold, tp, tn, fp, fn,
    )
    logger.info(
        "accuracy=%.4f precision=%.4f recall=%.4f F1=%.4f",
        accuracy, precision, recall, f1,
    )

    return {
        "species": species,
        "n": len(y_true),
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Loading V7 checkpoint: %s", MODEL_PATH)

    model, exon_scaler, global_scaler, threshold = load_model_and_scalers(device)
    results = []

    for species in SPECIES_LIST:
        try:
            result = evaluate_species(
                species,
                model,
                exon_scaler,
                global_scaler,
                threshold,
                device,
            )
            if result is not None:
                results.append(result)
        except Exception:
            logger.exception("Failed to evaluate %s", species)

    if results:
        summary = pd.DataFrame(results)
        output_path = (
            "/mnt/c/work/profiler_testing/dl_model/"
            "v7_6species_evaluation_summary.csv"
        )
        summary.to_csv(output_path, index=False)

        logger.info("=" * 68)
        logger.info("V7 six-species summary")
        logger.info("=" * 68)
        logger.info("\n%s", summary.to_string(index=False))
        logger.info("Saved summary: %s", output_path)


if __name__ == "__main__":
    main()
