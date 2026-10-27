# SynthHeart

Privacy-safe synthetic patient data for cardiovascular risk prediction. We train the bootcamp's TabDDPM on real patients and release synthetic ones instead. A scorecard proves they are realistic, useful and safe.

## Run order

| # | Notebook | Where | Time | Produces |
|---|---|---|---|---|
| 01 | `01_data_prep_and_baseline.ipynb` | anywhere (CPU) | ~1 min | `data/cardio/` splits + metadata, `results/baseline_metrics.json` |
| 02 | `02_train_tabddpm_and_generate.ipynb` | **AI Factory GPU** | smoke: minutes · full: **~4–5 h** | `runs/<mode>/` model + synthetic CSVs, `results/02_generation_summary_<mode>.json` |
| 03 | scorecard (next) | GPU or CPU | – | fidelity · utility · privacy |

All notebooks share `synthheart_common.py`: column order, cleaning/filter rules and paths.

## Setup (from the repo root)

```bash
uv sync --dev --group tabular-data     # installs midst-toolkit, torch, xgboost, sdv, opacus
```

## Running the long TabDDPM job (notebook 02)

1. Open the notebook and run it with `RUN_MODE = "smoke"` first, to confirm the environment works.
2. Set `RUN_MODE = "full"`, save, and launch it in the background so it survives a closed browser tab:

```bash
cd implementations/synthheart
mkdir -p logs
nohup uv run jupyter nbconvert --to notebook --execute 02_train_tabddpm_and_generate.ipynb \
      --output 02_train_tabddpm_and_generate.executed.ipynb \
      --ExecutePreprocessor.timeout=-1 > logs/02_full_run.log 2>&1 &
tail -f logs/02_full_run.log
```

## Data split (fixed seed 42)

| Split | Rows | Used for |
|---|---|---|
| train | 41,135 | the only data TabDDPM learns from |
| holdout | 13,712 | privacy metrics (closer to train than to holdout?) |
| test | 13,712 | locked final exam for every model |

Cleaning drops 1,441 of 70,000 rows (2.1%) that break medical-range rules. The same rules filter the synthetic rows.

## Real baseline (TRTR, locked test set)

| Model | AUC | Accuracy |
|---|---|---|
| XGBoost | 0.8014 | 0.7381 |
| Logistic regression | 0.7913 | 0.7277 |

**Target:** XGBoost trained only on synthetic patients (TSTR) within 0.02 AUC of 0.8014.

## Git hygiene

`runs/`, `logs/` and every CSV are git-ignored. Only code, metadata JSON and `results/*.json` are committed. Push to this fork only (`origin`), never to VectorInstitute.
