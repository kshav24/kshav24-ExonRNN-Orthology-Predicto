# v7c + v10 — 4-Species Bundle

Self-contained code bundle for training **v7c** and **v10** on ONLY the 4
placental species (mm10, bosTau8, equCab2, canFam3), with monDom5 excluded from
training. Both evaluate on the 6 held-out LONO species.

## Runners

| Script | Trains | Uses K2P clock? |
|--------|--------|-----------------|
| `train_v7c_4species_nodom5.py` | v7c | No (computes internal d) |
| `train_v10_4species_nodom5.py` | v10 | Yes (FiLM on external clock) |

```bash
python3 train_v7c_4species_nodom5.py
python3 train_v10_4species_nodom5.py
```

## Code dependencies (all included here)

- `k_2p_fix.py` — corrected textbook Kimura-2-parameter clock (`k2p_textbook`)
- `build_clean_k2p.py` — feature extraction + `process_transcript_group` + `k2p()`
- `train_all_clean.py` — unified training/eval harness (TRAINERS, evaluate, etc.)
- `train_v10.py` — v10 model + shared data utilities (dataset, scalers, load_split)
- `train_rnn_v7c_platypus.py` — v7c model
- `train_expert_m3.py`, `train_expert_m4.py` — m3/m4 models (imported by harness)
- `train_rnn_v8.py`, `train_rnn_v9.py` — v8/v9 models (imported by harness)

## Data files required (NOT included — too large for git)

The scripts reference these via absolute paths. Make sure they exist:

| Path | Purpose | Size |
|------|---------|------|
| `/mnt/c/work/profiler_testing/Training_data/train.csv` | 4-species training data | ~207 MB |
| `.../dl_model/clean_k2p/val_rn6_clean.npz` | eval: rn6 | — |
| `.../dl_model/clean_k2p/val_phyCat1_clean.npz` | eval: phyCat1 | — |
| `.../dl_model/clean_k2p/val_myoLuc2_clean.npz` | eval: myoLuc2 | — |
| `.../dl_model/clean_k2p/val_bisBis1_clean.npz` | eval: bisBis1 | — |
| `.../dl_model/clean_k2p/val_monDom5_clean.npz` | eval: monDom5 | — |
| `.../dl_model/clean_k2p/val_ornAna2_clean.npz` | eval: ornAna2 (LONO) | — |

Total data ≈ 274 MB (intentionally excluded from git).

## External packages

- `numpy`, `pandas`, `torch`, `scikit-learn`

## Note on paths

The runner scripts use hardcoded absolute paths (e.g. `/mnt/c/work/profiler_testing/...`)
for the K2P fix, data CSV, and validation NPZs. If you move this bundle to another
machine, update those paths in:
- `train_v7c_4species_nodom5.py` (`_K2P_FIX_PATH`, `RAW_TRAIN`)
- `train_v10_4species_nodom5.py` (`_K2P_FIX_PATH`, `RAW_TRAIN`)
- `train_all_clean.py` (`BASE`, `CLEAN`)
