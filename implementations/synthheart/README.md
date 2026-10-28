# SynthHeart

Privacy-safe synthetic patient data for cardiovascular risk prediction. We train the bootcamp's TabDDPM on real patients and release synthetic ones instead. A scorecard proves they are realistic, useful and safe.

## Run order

| # | Notebook | Where | Time | Produces |
|---|---|---|---|---|
| 01 | `01_data_prep_and_baseline.ipynb` | anywhere (CPU) | ~1 min | `data/cardio/` splits + metadata, `results/baseline_metrics.json` |
| 02 | `02_train_tabddpm_and_generate.ipynb` | **AI Factory GPU** | smoke: minutes · full: **~15–30 min on an H200** | `runs/<mode>/` model + synthetic CSVs, `results/02_generation_summary_<mode>.json` |
| 03 | scorecard (next) | GPU or CPU | – | fidelity · utility · privacy |

All notebooks share `synthheart_common.py`: column order, cleaning/filter rules and paths.

## Setup (from the repo root)

```bash
uv sync --dev --group tabular-data     # installs midst-toolkit, torch, xgboost, sdv, opacus
```

## Running the long TabDDPM job (notebook 02)

The run mode is chosen **when you launch**, so you never edit the notebook. It defaults to `smoke`.

**One-time kernel setup (important).** On the AI Factory, the default `python3` Jupyter kernel is the system conda Python, which does **not** have midst-toolkit (you'd get `ModuleNotFoundError: No module named 'midst_toolkit'`). Register the repo environment as a kernel once:
```bash
uv run python -m ipykernel install --user --name synthheart --display-name "SynthHeart (.venv)"
```
Always pass `--ExecutePreprocessor.kernel_name=synthheart` to nbconvert, and pick **SynthHeart (.venv)** as the kernel when running notebooks in JupyterLab.

1. Smoke test first (minutes):
```bash
cd implementations/synthheart
mkdir -p logs
uv run jupyter nbconvert --to notebook --execute 02_train_tabddpm_and_generate.ipynb \
      --output 02_smoke.executed.ipynb --ExecutePreprocessor.kernel_name=synthheart --ExecutePreprocessor.timeout=-1
```
2. Full run, in the background so it survives a closed browser tab:
```bash
SYNTHHEART_RUN_MODE=full nohup uv run jupyter nbconvert --to notebook --execute 02_train_tabddpm_and_generate.ipynb \
      --output 02_train_tabddpm_and_generate.executed.ipynb \
      --ExecutePreprocessor.kernel_name=synthheart \
      --ExecutePreprocessor.timeout=-1 > logs/02_full_run.log 2>&1 &
```
3. Watch progress (bar, step count, ETA, loss):
```bash
uv run python progress.py --watch          # refreshes every 20 s; Ctrl+C stops watching, not training
uv run python progress.py                  # one snapshot
```
`progress.py` reads `logs/02_<mode>_progress.log`. Training updates every 500 steps. Sampling has no step updates, so it shows elapsed time instead. If something crashes, the error is in `logs/02_full_run.log`.

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
