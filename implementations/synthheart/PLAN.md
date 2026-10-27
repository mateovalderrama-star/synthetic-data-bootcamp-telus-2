# SynthHeart · team plan

**Team:** Harshit · Mateo · Kiersten
**Goal for the build days:** a synthetic cardiovascular patient table that is realistic, useful and safe, plus a dashboard that proves it.
**Rule of thumb:** the TabDDPM full run takes **~4–5 h**, so it must start **as early as possible today**. Everything else is built in parallel against the smoke-run output.

---

## Build phases (whole project)

| Phase | What | Status |
|---|---|---|
| **1 · Data & baseline** | Clean, split 60/20/20, metadata, real XGBoost baseline (AUC 0.801) | ✅ done: `01_data_prep_and_baseline.ipynb` |
| **2 · Generate** | Train TabDDPM, sample 1× and 2×, filter | ⏳ today: `02_train_tabddpm_and_generate.ipynb` (smoke → full) |
| **3 · Evaluate** | Scorecard: bootcamp quality + privacy metrics, TSTR vs TRTR, risk curves, importance agreement | 🔨 today (build), finalize when the full run lands |
| **4 · Improve** | One tuning round if a target is missed. Stretch: CTGAN comparison, DP-SGD privacy–utility curve, small-clinic test | next build day |
| **5 · Showcase** | Streamlit dashboard (predict · compare · scorecard), slides, 3-minute story | 🔨 skeleton today, finish last day |

**Acceptance targets:**
- TSTR XGBoost AUC ≥ **0.781** (within 0.02 of 0.801)
- Membership-inference AUC ≤ **0.55**
- α-precision ≥ **0.90**
- ≥ **98%** of synthetic rows pass the rule filter
- DCR score ≈ **0.75**

---

## Who owns what (no overlapping files)

| Owner | Owns these files / folders | Today's mission |
|---|---|---|
| **Harshit** | `01_*.ipynb`, `02_*.ipynb`, `synthheart_common.py`, `runs/` on the AI Factory | Get TabDDPM training: environment → smoke → **launch the full run** → monitor |
| **Mateo** | `03_scorecard.ipynb` (+ `scorecard_utils.py` if needed) | Build the scorecard against the **smoke** synthetic data, so it's ready the moment the full run finishes |
| **Kiersten** | `dashboard/` (`app.py`, assets) | Build the Streamlit dashboard skeleton with the **real-data baseline model** and placeholder scorecard; swap in synthetic outputs later |

**Shared file rule:** `synthheart_common.py` (rules, column order, paths) is owned by Harshit. Need a change? Message the owner. Don't edit it in parallel. Import it, never copy it.

---

## Hand-off contracts (agree once, then work independently)

These are the only things that cross between people:

| Produced by | File | Used by |
|---|---|---|
| 01 (Harshit) | `data/cardio/cardio.csv`, `cardio_holdout.csv`, `cardio_test.csv` (12 columns, `sh.ALL_COLS` order) | 02, 03, dashboard |
| 01 (Harshit) | `results/baseline_metrics.json` → `trtr_baseline.xgboost.test_auc` | 03, dashboard |
| 02 (Harshit) | `runs/<mode>/synth_scale_1/cardio/_final/cardio_synthetic_filtered.csv` (same 12 columns) | 03, dashboard |
| 02 (Harshit) | `results/02_generation_summary_<mode>.json` | 03, dashboard |
| 03 (Mateo) | `results/03_scorecard_<mode>.json` (keys: `fidelity`, `utility`, `privacy`, `relationships`) | dashboard |
| 03 (Mateo) | `runs/<mode>/models/risk_model_synthetic.json` (XGBoost `save_model`) | dashboard |

`<mode>` is `smoke` today and `full` once the long run finishes. Build everything against `smoke`; switching to `full` is a one-line change.

---

## Today, step by step

### Block 1 · Now (everyone, ~20 min)
1. Everyone: `git pull origin main` on the AI Factory, then from the repo root `uv sync --dev --group tabular-data`.
2. Everyone: read this file, `README.md` and the cell titles of notebooks 01 and 02 (5 min).
3. Agree the hand-off contracts above (5 min).

### Block 2 · Start the clock (Harshit, ~45 min) · *critical path*
4. Run **notebook 01** on the AI Factory (≈1 min). Check: 68,559 clean rows, XGBoost AUC ≈ 0.80.
5. Run **notebook 02** with `RUN_MODE = "smoke"` (minutes). Check: model saved, synthetic CSVs written, filter pass rate printed, no errors.
6. **Commit + push the smoke summary** (`results/02_generation_summary_smoke.json`) so Mateo and Kiersten can see the paths work.
7. Set `RUN_MODE = "full"` and **launch in the background** (command in `README.md`). Note the start time. Expect results in ~4–5 h.

### Block 3 · Build in parallel while TabDDPM trains (~4 h)

**Harshit: monitor + prepare**
- Check `logs/02_full_run.log` every ~30 min (loss going down, no crash).
- Decide the **BP rounding question**: 97% of real systolic readings are multiples of 10. Round synthetic BP to tens, or accept and explain?
- Draft the "Data & generation" part of the showcase slides (data issues, cleaning, the split, the baseline).

**Mateo: `03_scorecard.ipynb`**, built against `runs/smoke/...`
1. Load real train / holdout / test and the synthetic filtered CSV.
2. **Bootcamp quality metrics** (copy from `tabular_data/evaluation/quality/quality_evaluation_pipeline.ipynb`): α-precision/β-recall, KS/TVD, CIO, correlation matrix difference, F1 difference with `LABEL_COLUMN="cardio"`, `TASK_TYPE="binclass"`.
3. **Bootcamp privacy metrics** (from `tabular_data/evaluation/privacy/`): DCR, median DCR, NNDR, hitting rate, ε-identifiability, plus the membership-inference attack.
4. **Our additions:**
   - TSTR vs TRTR XGBoost AUC on the locked test set (the headline).
   - Risk curves real vs synthetic (% cardio by BP band, age band, cholesterol).
   - The BP × age 3×3 table.
   - Permutation-importance agreement.
5. Save `results/03_scorecard_<mode>.json` and `runs/<mode>/models/risk_model_synthetic.json`.

**Kiersten: `dashboard/app.py`** (Streamlit)
1. **Tab 1 · Predict:** inputs for the 11 features → risk %. Use the real-baseline XGBoost for now, retrained in the app from `cardio.csv`. Swap to `risk_model_synthetic.json` later. Add a "demo, not a medical device" note.
2. **Tab 2 · Compare:** real vs synthetic histograms / bar charts per column, reading the smoke CSV.
3. **Tab 3 · Scorecard:** read `results/03_scorecard_<mode>.json`, with traffic-light status against the targets above. Use a placeholder JSON until Mateo's lands.
4. Run with `uv run streamlit run implementations/synthheart/dashboard/app.py`. Add `streamlit` with `uv add` only after telling the team, since it changes `pyproject.toml` / `uv.lock`.

### Block 4 · When the full run finishes (~end of day)
8. Harshit: check `results/02_generation_summary_full.json` (filter pass rate, quick TSTR AUC). Push it.
9. Mateo: switch `RUN_MODE` to `full`, run the scorecard, push `results/03_scorecard_full.json`.
10. Kiersten: point the dashboard at `full`. Check all three tabs.
11. Everyone (15 min): **which targets are met, which missed?** That decides tomorrow's Phase 4 work.

> If the full run isn't done by end of day, it keeps running overnight (it's a background job). Do steps 8–11 first thing tomorrow.

---

## Git rules (fork only)

- Push **directly to `origin main`** of this fork. **Never** open PRs or push to `VectorInstitute/synthetic-data-bootcamp`. On any GitHub PR page, the base repository must say `mateovalderrama-star/...`.
- `git pull origin main` **before you start** and **before every push**.
- Only commit files you own (table above). Commit small and often.
- Never commit `runs/`, `logs/`, model files or CSVs (already git-ignored). Results go in `results/*.json` and `*.png`.
- Don't commit `uv.lock` / `pyproject.toml` changes without telling the team.

---

## If something breaks

| Problem | First thing to try |
|---|---|
| `uv sync` fails on the AI Factory | Ask the Vector facilitators. The bootcamp's own tabular notebooks use the same group |
| Notebook 02 smoke fails | Read the error cell. Try the bootcamp's Berka notebook (`tabular_data/single_table/training/`) to see if it's the environment or our data |
| Full run crashes mid-way | Check `logs/02_full_run.log`. Re-launch. Consider `iterations=10000` to halve the time |
| Synthetic pass rate < 98% | Note which rules fail (in the summary JSON). Usually BP. Discuss tuning tomorrow |
| TSTR AUC far below 0.78 | Check the synthetic `cardio` balance and the risk curves in the scorecard; tuning in Phase 4 |
