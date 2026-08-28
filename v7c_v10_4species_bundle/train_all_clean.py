#!/usr/bin/env python3
"""Unified clean-data training harness for v7c, v8, m3, m4, v9, v10.

All six models are trained on the SAME clean_k2p data rebuilt from RAW CSVs
(39 -> 22 features via v10.load_split, majority-vote labels, per-species K2P
clock), with the SAME imputer/scaler (fit on training) and the SAME K2P clock.
The ONLY differences are each model's architecture, loss, and d handling:

  v7c : no external d (uses veto-feature mean as proxy)      Focal
  v8  : raw d                                                CE + 0.10*MSE + 0.01*(-H)
  m3  : raw d                                                ASL + 0.10*NLL + 0.01*(-H)
  m4  : raw d + v_macro                                      CE + 0.10*NLL + 0.01*(-H)
  v9  : standardized d                                       Focal + proto + 0.01*(-H)
  v10 : standardized d + v_macro                             Focal + proto + 0.01*(-H)

Priority 2 : train on clean training data only (4 placentals).
Priority 3 : train on clean training data + 50% monDom5 + 50% ornAna2.
             (eval for monDom5/ornAna2 uses the held-out 50% halves)

Usage:
  python3 train_all_clean.py --priority 2
  python3 train_all_clean.py --priority 3
  python3 train_all_clean.py --priority 2 --models v9 v10
"""

import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/c/work/profiler_testing/dl_model")
import train_expert_m3 as m3  # noqa: E402
import train_expert_m4 as m4  # noqa: E402
import train_rnn_v7c_platypus as v7c  # noqa: E402
import train_rnn_v8 as v8  # noqa: E402
import train_rnn_v9 as v9  # noqa: E402
import train_v10 as v10  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE = "/mnt/c/work/profiler_testing/dl_model"
CLEAN = f"{BASE}/clean_k2p"
SPECIES = ["rn6", "phyCat1", "myoLuc2", "bisBis1", "monDom5", "ornAna2"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
VETO_IDX = v10.EXON_FEATS.index("max_kaks_window")  # 12


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(priority):
    """Load + preprocess clean data for a given priority.

    Returns (X_tr, y_tr, sp_tr, d_tr, vm_tr, imputer, scaler, d_mean, d_std,
    eval_data) where eval_data is {species: (X, y, d, vm)}.
    """
    if priority == 2:
        train_paths = [f"{CLEAN}/train_clean.npz"]
        eval_paths = {sp: f"{CLEAN}/val_{sp}_clean.npz" for sp in SPECIES}
    elif priority == 4:
        # NEW: 4 placental species + full ornAna1 (platypus), NO monDom5 / ornAna2.
        # Evaluate on all 6 held-out eval species.
        train_paths = [
            f"{CLEAN}/train_clean.npz",
            f"{CLEAN}/ornAna1_all.npz",
        ]
        eval_paths = {sp: f"{CLEAN}/val_{sp}_clean.npz" for sp in SPECIES}
    else:
        train_paths = [
            f"{CLEAN}/train_clean.npz",
            f"{CLEAN}/val_monDom5_trhalf.npz",
            f"{CLEAN}/val_ornAna2_trhalf.npz",
        ]
        eval_paths = {
            "rn6": f"{CLEAN}/val_rn6_clean.npz",
            "phyCat1": f"{CLEAN}/val_phyCat1_clean.npz",
            "myoLuc2": f"{CLEAN}/val_myoLuc2_clean.npz",
            "bisBis1": f"{CLEAN}/val_bisBis1_clean.npz",
            "monDom5": f"{CLEAN}/val_monDom5_evhalf.npz",
            "ornAna2": f"{CLEAN}/val_ornAna2_evhalf.npz",
        }

    # Macro medians from training data (fill NaNs in macro features)
    raw_caches = [np.load(p, allow_pickle=True) for p in train_paths]
    feat_names = list(raw_caches[0]["feature_names"])
    macro_medians = v10._compute_macro_medians(raw_caches, feat_names)

    # Load + concatenate training
    X_tr, y_tr, sp_tr, d_tr, vm_tr = [], [], [], [], []
    for p in train_paths:
        b = v10.load_split(p, p, macro_medians)
        X_tr += b.X_exons
        y_tr.append(b.y)
        sp_tr += b.species
        d_tr.append(b.d)
        vm_tr.append(b.v_macros)
    X_tr, y_tr = X_tr, np.concatenate(y_tr)
    d_tr, vm_tr = np.concatenate(d_tr), np.concatenate(vm_tr)

    # Fit imputer/scaler strictly on training
    imputer, scaler = v10.fit_scalers(X_tr)
    X_tr = v10.apply_scalers(X_tr, imputer, scaler)

    d_mean = float(np.mean(d_tr))
    d_std = float(np.std(d_tr))
    if d_std < 1e-8:
        d_std = 1.0
    logger.info(
        f"priority={priority} train n={len(y_tr)} orth={int(y_tr.sum())} "
        f"({y_tr.mean() * 100:.1f}%) d_mean={d_mean:.4f} d_std={d_std:.4f}"
    )

    # Load eval (apply training scalers)
    eval_data = {}
    for sp, p in eval_paths.items():
        b = v10.load_split(p, p, macro_medians)
        X_eval = v10.apply_scalers(b.X_exons, imputer, scaler)
        eval_data[sp] = (X_eval, b.y, b.d, b.v_macros)
    return X_tr, y_tr, sp_tr, d_tr, vm_tr, imputer, scaler, d_mean, d_std, eval_data


def _chunk_probs(model, forward_fn, X, d, vm, d_mean, d_std, use_std_d, chunk=512):
    """Run forward over chunks, return ortholog probabilities."""
    probs = []
    for i in range(0, len(X), chunk):
        ex = X[i : i + chunk]
        padded = pad_sequence([torch.from_numpy(x) for x in ex], batch_first=True)
        lengths = torch.tensor([x.shape[0] for x in ex])
        d_b = torch.tensor(d[i : i + chunk], dtype=torch.float32)
        vm_b = torch.stack([torch.from_numpy(v) for v in vm[i : i + chunk]])
        padded, lengths, d_b, vm_b = (
            padded.to(DEVICE),
            lengths.to(DEVICE),
            d_b.to(DEVICE),
            vm_b.to(DEVICE),
        )
        probs.append(
            forward_fn(padded, lengths, d_b, vm_b, d_mean, d_std).cpu().numpy()
        )
    return np.concatenate(probs)


def _forward_v7c(padded, lengths, d_b, vm_b, d_mean, d_std):
    logits, _, _ = MODEL["v7c"](padded, lengths)
    return torch.sigmoid(logits)


def _forward_v8(padded, lengths, d_b, vm_b, d_mean, d_std):
    logits, _, _, _ = MODEL["v8"](padded, d_b, lengths)
    return torch.softmax(logits, dim=-1)[:, 1]


def _forward_m3(padded, lengths, d_b, vm_b, d_mean, d_std):
    logits, _, _, _ = MODEL["m3"](padded, d_b, lengths)
    return torch.softmax(logits, dim=-1)[:, 1]


def _forward_m4(padded, lengths, d_b, vm_b, d_mean, d_std):
    logits, _, _, _ = MODEL["m4"](padded, d_b, lengths, vm_b)
    return torch.softmax(logits, dim=-1)[:, 1]


def _forward_v9(padded, lengths, d_b, vm_b, d_mean, d_std):
    dz = (d_b - d_mean) / d_std
    logits, _, _ = MODEL["v9"](padded, lengths, dz)
    return torch.sigmoid(logits)


def _forward_v10(padded, lengths, d_b, vm_b, d_mean, d_std):
    dz = (d_b - d_mean) / d_std
    logits, _, _ = MODEL["v10"](padded, lengths, dz, vm_b)
    return torch.sigmoid(logits)


FORWARD = {
    "v7c": _forward_v7c,
    "v8": _forward_v8,
    "m3": _forward_m3,
    "m4": _forward_m4,
    "v9": _forward_v9,
    "v10": _forward_v10,
}

MODEL = {}


# ---------------------------------------------------------------------------
# Per-model training
# ---------------------------------------------------------------------------
def train_v7c(X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std):
    v10.set_seed(SEED)
    model = v7c.ExonRNN_V7C(exon_dim=22, veto_idx=VETO_IDX).to(DEVICE)
    ds = v7c.ExonDatasetV7C(X_tr, y_tr)
    sampler = v10.get_sampler(sp_tr, y_tr, seed=SEED)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=v7c.collate_fn_v7c
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(loader), epochs=EPOCHS
    )
    crit = v7c.FocalLoss()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for X_b, lengths, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            logits, _, _ = model(X_b, lengths)
            loss = crit(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        logger.info(f"v7c ep{ep + 1}/{EPOCHS} loss={tot / len(loader):.4f}")
    return model


def train_v8(X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std):
    v10.set_seed(SEED)
    model = v8.ExonRNN_V8(exon_dim=22).to(DEVICE)
    ds = v8.ExonDatasetV8(X_tr, d_tr, y_tr)
    sampler = v10.get_sampler(sp_tr, y_tr, seed=SEED)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=v8.collate_fn_v8
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(loader), epochs=EPOCHS
    )
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for X_b, d_b, lengths, y_b in loader:
            X_b, d_b, y_b = X_b.to(DEVICE), d_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            logits, pred_d, _, entropy = model(
                X_b, d_b, lengths, labels=y_b, lambda_adv=0.10
            )
            loss = ce(logits, y_b.long()) + 0.10 * mse(pred_d, d_b) + 0.01 * (-entropy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        logger.info(f"v8 ep{ep + 1}/{EPOCHS} loss={tot / len(loader):.4f}")
    return model


def train_m3(X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std):
    v10.set_seed(SEED)
    model = m3.ExonRNN_ExpertM3(exon_dim=22).to(DEVICE)
    ds = m3.ExonDataset(X_tr, d_tr, y_tr)
    sampler = v10.get_sampler(sp_tr, y_tr, seed=SEED)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=m3.collate_fn
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(loader), epochs=EPOCHS
    )
    asl = m3.AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip_margin=0.05)
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for X_b, d_b, lengths, y_b in loader:
            X_b, d_b, y_b = X_b.to(DEVICE), d_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            logits, dom, _, entropy = model(
                X_b, d_b, lengths, labels=y_b, lambda_adv=0.10
            )
            mu = dom[:, 0]
            logvar = dom[:, 1].clamp(min=-5.0, max=5.0)
            nll = (0.5 * torch.exp(-logvar) * (d_b - mu) ** 2 + 0.5 * logvar).mean()
            loss = asl(logits, y_b) + 0.10 * nll + 0.01 * (-entropy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        logger.info(f"m3 ep{ep + 1}/{EPOCHS} loss={tot / len(loader):.4f}")
    return model


def train_m4(X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std):
    v10.set_seed(SEED)
    model = m4.ExonRNN_ExpertM4(exon_dim=22).to(DEVICE)
    ds = m4.ExonDatasetM4(X_tr, d_tr, y_tr, vm_tr)
    sampler = v10.get_sampler(sp_tr, y_tr, seed=SEED)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=m4.collate_fn_m4
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(loader), epochs=EPOCHS
    )
    ce = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for X_b, d_b, lengths, vm_b, y_b in loader:
            X_b, d_b, vm_b, y_b = (
                X_b.to(DEVICE),
                d_b.to(DEVICE),
                vm_b.to(DEVICE),
                y_b.to(DEVICE),
            )
            opt.zero_grad()
            logits, dom, _, entropy = model(
                X_b, d_b, lengths, vm_b, labels=y_b, lambda_adv=0.10
            )
            mu = dom[:, 0]
            logvar = dom[:, 1].clamp(min=-5.0, max=5.0)
            nll = (0.5 * torch.exp(-logvar) * (d_b - mu) ** 2 + 0.5 * logvar).mean()
            loss = ce(logits, y_b.long()) + 0.10 * nll + 0.01 * (-entropy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        logger.info(f"m4 ep{ep + 1}/{EPOCHS} loss={tot / len(loader):.4f}")
    return model


def _train_proto_model(
    model, X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std, use_struct
):
    """Shared Focal + prototype + entropy training for v9 (no struct) / v10 (struct)."""
    ds = v10.ExonDataset(X_tr, y_tr, (d_tr - d_mean) / d_std, vm_tr)
    sampler = v10.get_sampler(sp_tr, y_tr, seed=SEED)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=v10.collate_fn
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(loader), epochs=EPOCHS
    )
    crit = v10.FocalLoss()
    lam_ent, lam_proto, hidden = 0.01, 0.5, 256
    c_orth = torch.zeros(hidden, device=DEVICE)
    c_para = torch.zeros(hidden, device=DEVICE)
    ema = 0.99
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for X_b, lengths, y_b, d_b, vm_b in loader:
            X_b, y_b, d_b, vm_b = (
                X_b.to(DEVICE),
                y_b.to(DEVICE),
                d_b.to(DEVICE),
                vm_b.to(DEVICE),
            )
            opt.zero_grad()
            if use_struct:
                logits, attended, entropy = model(X_b, lengths, d_b, vm_b)
            else:
                logits, attended, entropy = model(X_b, lengths, d_b)
            loss_focal = crit(logits, y_b)
            with torch.no_grad():
                om, pm = y_b == 1, y_b == 0
                if om.any():
                    c_orth = ema * c_orth + (1 - ema) * attended[om].mean(dim=0)
                if pm.any():
                    c_para = ema * c_para + (1 - ema) * attended[pm].mean(dim=0)
            c_on = nn.functional.normalize(c_orth.unsqueeze(0), dim=1).expand(
                attended.size(0), -1
            )
            c_pn = nn.functional.normalize(c_para.unsqueeze(0), dim=1).expand(
                attended.size(0), -1
            )
            h_n = nn.functional.normalize(attended, dim=1)
            logits_proto = torch.stack(
                [(h_n * c_pn).sum(dim=1), (h_n * c_on).sum(dim=1)], dim=1
            )
            loss_proto = nn.functional.cross_entropy(logits_proto / 0.05, y_b.long())
            loss_orth = torch.relu((c_on[0] * c_pn[0]).sum())
            loss = (
                loss_focal
                + lam_ent * (-entropy)
                + lam_proto * loss_proto
                + 0.5 * loss_orth
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        logger.info(f"proto-model ep{ep + 1}/{EPOCHS} loss={tot / len(loader):.4f}")


def train_v9(X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std):
    v10.set_seed(SEED)
    model = v9.ExonRNNSeq(exon_dim=22, veto_idx=VETO_IDX).to(DEVICE)
    _train_proto_model(
        model, X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std, use_struct=False
    )
    return model


def train_v10(X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std):
    v10.set_seed(SEED)
    model = v10.ExonRNNV10(exon_dim=22, veto_idx=VETO_IDX, use_struct=True).to(DEVICE)
    _train_proto_model(
        model, X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std, use_struct=True
    )
    return model


TRAINERS = {
    "v7c": train_v7c,
    "v8": train_v8,
    "m3": train_m3,
    "m4": train_m4,
    "v9": train_v9,
    "v10": train_v10,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(name, d_mean, d_std, eval_data):
    model = MODEL[name]
    model.eval()
    rows = {}
    with torch.no_grad():
        for sp in SPECIES:
            X, y, d, vm = eval_data[sp]
            probs = _chunk_probs(
                model, FORWARD[name], X, d, vm, d_mean, d_std, use_std_d=None
            )
            y = y.astype(int)
            try:
                auc = float(roc_auc_score(y, probs))
            except Exception:
                auc = float("nan")
            rows[sp] = (probs, y, auc)
    return rows


def print_metrics(rows, tag):
    print("\n" + "=" * 108)
    print(f" {tag}  (threshold rows: TP TN FP FN  F1  precision  recall  AUC)")
    print("=" * 108)
    for sp in SPECIES:
        probs, y, auc = rows[sp]
        marker = "  << LONO" if sp == "ornAna2" else ""
        for thr in (0.4, 0.5):
            pred = (probs >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            m = marker if thr == 0.4 else ""
            print(
                f"  {sp:9s} {thr:4.1f} {tp:6d} {tn:6d} {fp:6d} {fn:6d} "
                f"{f1:7.4f} {prec:7.4f} {rec:7.4f} {auc:7.4f}{m}"
            )
        print(
            f"    ({sp}: n_orth={int((y == 1).sum())} n_para={int((y == 0).sum())} n={len(y)})"
        )
        print("  " + "-" * 104)


def save_checkpoint(name, d_mean, d_std, imputer, scaler, macro_medians, priority):
    torch.save(
        {
            "model_state_dict": MODEL[name].state_dict(),
            "exon_dim": 22,
            "veto_idx": VETO_IDX,
            "d_mean": d_mean,
            "d_std": d_std,
            "imputer_statistics": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "n_cont": v10.N_CONT,
            "macro_medians": macro_medians,
            "seed": SEED,
            "priority": priority,
            "source": "clean_k2p (RAW CSVs, majority labels, per-species K2P)",
        },
        f"{BASE}/dl_{name}_clean_p{priority}.pth",
    )
    logger.info(f"Saved checkpoint -> dl_{name}_clean_p{priority}.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument(
        "--models", nargs="+", default=["v7c", "v8", "m3", "m4", "v9", "v10"]
    )
    args = ap.parse_args()

    (
        X_tr,
        y_tr,
        sp_tr,
        d_tr,
        vm_tr,
        imputer,
        scaler,
        d_mean,
        d_std,
        eval_data,
    ) = prepare_data(args.priority)
    raw_caches = [np.load(f"{CLEAN}/train_clean.npz", allow_pickle=True)]
    macro_medians = v10._compute_macro_medians(
        raw_caches, list(raw_caches[0]["feature_names"])
    )

    for name in args.models:
        logger.info(f"===== Training {name} (priority {args.priority}) =====")
        MODEL[name] = TRAINERS[name](X_tr, y_tr, sp_tr, d_tr, vm_tr, d_mean, d_std)
        save_checkpoint(
            name, d_mean, d_std, imputer, scaler, macro_medians, args.priority
        )
        rows = evaluate(name, d_mean, d_std, eval_data)
        print_metrics(rows, f"{name} clean priority-{args.priority}")
        # free GPU memory between models
        del MODEL[name]
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
