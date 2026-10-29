"""SynthHeart · 04 · Training-steps ablation: does a less-trained TabDDPM leak fewer exact matches?

Re-runs notebook 02 at fewer training iterations (10k, 5k; the full run used 20k), scores every run with notebook 03,
and compares them. Both notebooks run unchanged: one extra cell after 02's settings cell overrides the iteration count.

Run on the AI Factory, from implementations/synthheart (~30 min on the H200):

    mkdir -p logs
    nohup uv run python 04_steps_ablation.py > logs/04_steps_ablation.log 2>&1 &

Outputs: runs/iter<N>/ (git-ignored), results/02_generation_summary_iter<N>.json, results/03_scorecard_iter<N>.json,
results/03_scorecard_full.json (re-scored with the seeded notebook 03) and results/04_steps_ablation.json (comparison).
Set SYNTHHEART_ABLATION_ITERS=10000,5000,2000 to try other counts. A step whose output already exists is skipped,
so re-launching after a crash resumes; delete results/03_scorecard_*.json to force re-scoring.
"""
import json
import os
import sys
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthheart_common as sh

ITERATIONS = [int(i) for i in os.environ.get("SYNTHHEART_ABLATION_ITERS", "10000,5000").split(",")]


def execute(notebook: str, mode: str, override: str | None = None) -> None:
    nb = nbformat.read(HERE / notebook, as_version=4)
    if override:
        assert "DIFFUSION_CONFIG = dict(" in nb.cells[2].source, "notebook 02's settings cell moved; update the index"
        nb.cells.insert(3, nbformat.v4.new_code_cell(override))
    # The kernel inherits this environment. 02 starts from "full" and the override cell renames the run.
    os.environ["SYNTHHEART_RUN_MODE"] = "full" if override else mode
    print(f"running {notebook} ({mode}) ...", flush=True)
    NotebookClient(nb, kernel_name="synthheart", timeout=None, resources={"metadata": {"path": str(HERE)}}).execute()
    sh.LOGS_DIR.mkdir(exist_ok=True)
    nbformat.write(nb, sh.LOGS_DIR / f"{Path(notebook).stem}_{mode}.executed.ipynb")


modes = ["full"]
for n in ITERATIONS:
    mode = f"iter{n}"
    modes.append(mode)
    if not (sh.RESULTS_DIR / f"02_generation_summary_{mode}.json").exists():
        # 02 starts from its "full" config (2,000 diffusion timesteps, batch 4,096, same seed); only iterations change.
        execute("02_train_tabddpm_and_generate.ipynb", mode,
                f'RUN_MODE = "{mode}"\nDIFFUSION_CONFIG["iterations"] = {n}\nprint(RUN_MODE, DIFFUSION_CONFIG)')
for mode in modes:
    if mode == "full" or not (sh.RESULTS_DIR / f"03_scorecard_{mode}.json").exists():
        execute("03_scorecard.ipynb", mode)  # full is always re-scored so every run uses the same seeded scorecard

rows = {}
for mode in modes:
    gen = json.loads((sh.RESULTS_DIR / f"02_generation_summary_{mode}.json").read_text())
    card = json.loads((sh.RESULTS_DIR / f"03_scorecard_{mode}.json").read_text())
    fid, util, priv = card["fidelity"], card["utility"], card["privacy"]
    copies, ap = priv["exact_copies"], fid["alpha_precision"]
    rows[mode] = {
        "iterations": gen["diffusion_config"]["iterations"],
        "train_minutes": gen["train_minutes"],
        # Leakage signal: train-sample vs holdout are the same size, so their gap is matches beyond chance.
        "exact_copies_train": copies["train"],
        "exact_copies_train_sample": copies["train_sample_holdout_size"],
        "exact_copies_holdout": copies["holdout"],
        "exact_copies_excess": copies["train_sample_holdout_size"] - copies["holdout"],
        "hitting_rate_train": round(priv["hitting_rate"]["train"], 4),
        "hitting_rate_holdout": round(priv["hitting_rate"]["holdout"], 4),
        "dcr_score": round(priv["dcr"]["dcr_score"], 4),
        "mia_auc": priv["mia"]["auc"],
        "alpha_precision_OC": round(ap["delta_precision_alpha_OC"], 4),
        "beta_coverage_OC": round(ap["delta_coverage_beta_OC"], 4),
        "beta_coverage_naive": round(ap["delta_coverage_beta_naive"], 4),
        "authenticity_OC": round(ap["authenticity_OC"], 4),
        "ks_tvd_columns_differing": fid["ks_tvd"]["num sigs"],
        "correlation_gap": round(fid["correlation"]["corr_mat_diff"], 4),
        "tstr_auc": util["tstr_auc"]["synth_scale_1"],
        "targets_passed": f'{sum(c["pass"] for c in card["acceptance"].values())}/{len(card["acceptance"])}',
    }

table = pd.DataFrame(rows)
print(table.to_string())
out = sh.RESULTS_DIR / "04_steps_ablation.json"
out.write_text(json.dumps(rows, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("wrote", out)
