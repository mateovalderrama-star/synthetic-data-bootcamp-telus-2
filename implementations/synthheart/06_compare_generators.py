"""SynthHeart · 06 · TabDDPM vs NeMo Safe Synthesizer: score every run with the same scorecard, then compare.

Three steps. Each is skipped when its output already exists, so re-launching resumes.

  1. Re-run notebook 02 (TabDDPM, full settings) as run "tabddpm" under the GPU monitor, so its time, memory and
     energy are measured on the same node and by the same instrument as the NSS runs (~10 min on the H200).
  2. Run the scorecard (03b, unchanged) on "tabddpm" and on every NSS run made by 05_nss_generate.py.
     For NSS, `train` is replaced by the rows NSS actually trained on (runs/<name>/train_seen.csv).
  3. Build the side-by-side tables and figures.

Run on the AI Factory, from implementations/synthheart:

    nohup uv run python 06_compare_generators.py > logs/06_compare.log 2>&1 &

    uv run python 06_compare_generators.py --modes full nss_default   # pick the runs to compare (max 4 in figures)
    uv run python 06_compare_generators.py --no-run                   # only rebuild tables + figures (CPU, anywhere)

Outputs, all under results/comparison/ (committed; this is the material for the slide deck):
  comparison.md / comparison.json      every metric side by side
  *.png                                headline metrics, runtime, distributions, correlation gaps, risk curves,
                                       privacy vs utility
  figures/<run>/*.png                  every figure from that run's scorecard notebook
  reports/03b_<run>.html               the executed scorecard notebook, self-contained
  gpu_<run>.json                       GPU memory / utilisation / energy trace
  nss_native/                          NSS's own evaluation report (written by 05)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthheart_common as sh  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402


RESULTS = sh.RESULTS_DIR
OUT = RESULTS / "comparison"
RUNS = sh.PROJECT / "runs"

# Chart colours. A generator keeps its colour in every figure; real data is always neutral grey.
INK, INK2, MUTED, GRID, AXIS, SURFACE, REAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#a8a69f"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
FIXED = {"tabddpm": SLOTS[0], "full": SLOTS[0], "nss_default": SLOTS[1], "nss_dp8": SLOTS[2]}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "savefig.bbox": "tight", "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False,
})


def label(mode: str) -> str:
    return {"tabddpm": "TabDDPM", "full": "TabDDPM (bootcamp run)"}.get(mode, mode.replace("nss_", "NSS ").replace("dp", "DP ε="))


def colors(modes: list[str]) -> dict[str, str]:
    free = [c for c in SLOTS if c not in {FIXED.get(m) for m in modes}]
    return {m: FIXED.get(m) or free.pop(0) for m in modes}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- steps 1 + 2: run the notebooks

def execute(notebook: str, mode: str, env_mode: str, inject: tuple[int, str, str] | None = None) -> None:
    """Run a notebook for `mode`, optionally inserting one cell, and keep everything it showed.

    inject = (index for the new cell, text the cell before it must contain, source of the new cell)
    """
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    nb = nbformat.read(HERE / notebook, as_version=4)
    if inject:
        index, anchor, source = inject
        assert anchor in nb.cells[index - 1].source, f"{notebook} changed: cell {index - 1} no longer has {anchor!r}"
        nb.cells.insert(index, nbformat.v4.new_code_cell(source))
    os.environ["SYNTHHEART_RUN_MODE"] = env_mode  # the kernel inherits this
    stem = f"{Path(notebook).stem.split('_')[0]}_{mode}"
    print(f"running {notebook} for {mode} ...", flush=True)
    try:
        NotebookClient(nb, kernel_name="synthheart", timeout=None, resources={"metadata": {"path": str(HERE)}}).execute()
    finally:  # also on failure, so the error is readable in the saved notebook
        sh.LOGS_DIR.mkdir(exist_ok=True)
        nbformat.write(nb, sh.LOGS_DIR / f"{stem}.executed.ipynb")
        (OUT / "reports").mkdir(parents=True, exist_ok=True)
        (OUT / "reports" / f"{stem}.html").write_text(HTMLExporter().from_notebook_node(nb)[0], encoding="utf-8")
        figs, n = OUT / "figures" / mode, 0
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                png = output.get("data", {}).get("image/png")
                if png:
                    n += 1
                    figs.mkdir(parents=True, exist_ok=True)
                    (figs / f"{stem}_fig{n:02d}.png").write_bytes(base64.b64decode(png))


def run_notebooks(modes: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "tabddpm" in modes and not (RESULTS / "02_generation_summary_tabddpm.json").exists():
        # Same trick as 04_steps_ablation.py: notebook 02 starts from its "full" settings, one extra cell renames
        # the run, so results/*_full.json and runs/full are left untouched.
        with GpuMonitor(OUT / "gpu_tabddpm.json"):
            execute("02_train_tabddpm_and_generate.ipynb", "tabddpm", "full",
                    (3, "DIFFUSION_CONFIG = dict(", 'RUN_MODE = "tabddpm"\nprint(RUN_MODE, DIFFUSION_CONFIG)'))
    for mode in modes:
        if (RESULTS / f"03b_scorecard_{mode}.json").exists():
            continue
        if not (RESULTS / f"02_generation_summary_{mode}.json").exists():
            print(f"skipping {mode}: no results/02_generation_summary_{mode}.json (generation has not finished)")
            continue
        seen = RUNS / mode / "train_seen.csv"
        inject = (5, "train = pd.read_csv", "# This generator held some train rows out for its own report: score "
                  f'against the rows it saw.\ntrain = pd.read_csv(r"{seen}")[sh.ALL_COLS]\nprint("train rows seen '
                  'by the generator:", len(train))') if seen.exists() else None
        execute("03b_scorecard_size_matched.ipynb", mode, mode, inject)


# ---------------------------------------------------------------- step 3: tables

def collect(mode: str) -> dict | None:
    """One flat row of numbers for a run, from its generation summary, scorecard and GPU trace."""
    gen, card = load(RESULTS / f"02_generation_summary_{mode}.json"), load(RESULTS / f"03b_scorecard_{mode}.json")
    if not (gen and card):
        return None
    gpu = load(OUT / f"gpu_{mode}.json") or {}
    fid, util, priv, rel = card["fidelity"], card["utility"], card["privacy"], card["relationships"]
    ap, copies = fid["alpha_precision"], priv["exact_copies"]
    scale = max(gen["sample_minutes"], key=lambda s: gen["filter"][s]["generated"])  # the biggest sampling call
    rows_per_s = gen["filter"][scale]["generated"] / max(gen["sample_minutes"][scale] * 60, 1e-9)
    ckpt = RUNS / mode / "models" / f"None_{sh.TABLE}_ckpt.pkl"
    return {
        "generator": gen.get("generator", "tabddpm"),
        "train_rows_seen": card["rows"]["train"],
        "synthetic_rows_scored": card["rows"]["synth_scale_1"],
        "rule_pass_rate": gen["filter"]["synth_scale_1"]["pass_rate"],
        "alpha_precision": ap["delta_precision_alpha_OC"],
        "beta_recall": ap["delta_coverage_beta_OC"],
        "authenticity": ap["authenticity_OC"],
        "ks_tvd_avg": fid["ks_tvd"]["avg stat"],
        "columns_differing": fid["ks_tvd"]["num sigs"],
        "ci_overlap": fid["ci_overlap"]["avg overlap"],
        "correlation_gap": fid["correlation"]["corr_mat_diff"],
        "mutual_information_gap": fid["mutual_information"]["mutual_inf_diff"],
        "trtr_auc": util["trtr_auc"],
        "tstr_auc": util["tstr_auc"]["synth_scale_1"],
        "tstr_auc_2x": util["tstr_auc"]["synth_scale_2"],
        "tstr_auc_gap": util["auc_gap"]["synth_scale_1"],
        "f1_difference_holdout": util["f1_difference"]["mean_f1_difference_holdout"],
        "mia_auc": priv["mia"]["auc"],
        "mia_tpr_at_10pct_fpr": priv["mia"]["tpr_at_10pct_fpr"],
        "dcr_score": priv["dcr"]["dcr_score"],
        "dcr_ideal": priv["dcr"]["ideal"],
        "median_dcr": priv["median_dcr"]["median_dcr_score"],
        "nndr": priv["nndr"]["mean_nndr"],
        "hitting_rate_train_sample": priv["hitting_rate"]["train_sample_holdout_size"],
        "hitting_rate_holdout": priv["hitting_rate"]["holdout"],
        "epsilon_identifiability": priv["epsilon_identifiability"]["epsilon_identifiability_risk"],
        "exact_copies_train_sample": copies["train_sample_holdout_size_mean"],
        "exact_copies_holdout": copies["holdout"],
        "exact_copies_ratio": copies["train_sample_holdout_size_mean"] / max(copies["holdout"], 1),
        "bp_age_max_diff": rel["bp_age_table"]["max_abs_diff"],
        "importance_spearman": rel["importance"]["spearman"],
        "targets_passed": f'{sum(c["pass"] for c in card["acceptance"].values())}/{len(card["acceptance"])}',
        "train_minutes": gen["train_minutes"],
        "generate_rows_per_second": rows_per_s,
        "generate_minutes_per_train_size": gen["train_rows"] / rows_per_s / 60,
        "total_minutes": gpu.get("total_minutes"),
        "peak_gpu_gb": gpu.get("peak_mem_gb"),
        "gpu_energy_wh": gpu.get("energy_wh"),
        "mean_gpu_util_pct": gpu.get("mean_util_pct"),
        "model_mb": ckpt.stat().st_size / 1e6 if ckpt.exists() else gen.get("adapter_mb"),
    }


# (section, row label, key, number format, what good looks like)
TABLE = [
    ("Fidelity", "α-precision (synthetic rows look real)", "alpha_precision", ".3f", "≥ 0.90"),
    ("Fidelity", "β-recall (real variety is covered)", "beta_recall", ".3f", "higher"),
    ("Fidelity", "Columns whose distribution differs (KS / TVD)", "columns_differing", ".0f", "≤ 1"),
    ("Fidelity", "Mean KS / TVD distance", "ks_tvd_avg", ".4f", "lower"),
    ("Fidelity", "Confidence-interval overlap", "ci_overlap", ".3f", "higher"),
    ("Fidelity", "Correlation-matrix gap", "correlation_gap", ".3f", "≤ 0.10"),
    ("Fidelity", "Mutual-information gap", "mutual_information_gap", ".3f", "lower"),
    ("Fidelity", "Rows passing the medical rules", "rule_pass_rate", ".2%", "≥ 98%"),
    ("Utility", "TSTR AUC, 1× synthetic (real test set)", "tstr_auc", ".4f", "≥ real − 0.02"),
    ("Utility", "TSTR AUC, 2× synthetic", "tstr_auc_2x", ".4f", "higher"),
    ("Utility", "Real-data baseline AUC (TRTR)", "trtr_auc", ".4f", "reference"),
    ("Utility", "AUC lost vs real data", "tstr_auc_gap", "+.4f", "≤ 0.02"),
    ("Utility", "Mean F1 difference (holdout)", "f1_difference_holdout", "+.4f", "near 0"),
    ("Privacy", "Membership-inference AUC", "mia_auc", ".4f", "≤ 0.55 (0.50 = coin flip)"),
    ("Privacy", "MIA true-positive rate at 10% false positives", "mia_tpr_at_10pct_fpr", ".3f", "≈ 0.10"),
    ("Privacy", "DCR score", "dcr_score", ".4f", "≈ ideal"),
    ("Privacy", "DCR ideal (train share)", "dcr_ideal", ".4f", "reference"),
    ("Privacy", "Exact copies: train sample ÷ holdout", "exact_copies_ratio", ".2f", "≤ 1.2"),
    ("Privacy", "Hitting rate, train sample", "hitting_rate_train_sample", ".3f", "≈ holdout"),
    ("Privacy", "Hitting rate, holdout", "hitting_rate_holdout", ".3f", "reference"),
    ("Privacy", "ε-identifiability risk", "epsilon_identifiability", ".3f", "lower"),
    ("Privacy", "NNDR", "nndr", ".3f", "higher"),
    ("Relationships", "BP × age risk table, max difference", "bp_age_max_diff", ".3f", "lower"),
    ("Relationships", "Feature-importance agreement (Spearman)", "importance_spearman", ".3f", "higher"),
    ("Overall", "Acceptance targets passed", "targets_passed", "", "8/8"),
    ("Runtime", "Training (min)", "train_minutes", ".1f", "lower"),
    ("Runtime", "Generating one train-size table (min)", "generate_minutes_per_train_size", ".2f", "lower"),
    ("Runtime", "Generation speed (rows / s)", "generate_rows_per_second", ",.0f", "higher"),
    ("Runtime", "Whole run, wall clock (min)", "total_minutes", ".1f", "lower"),
    ("Runtime", "Peak GPU memory (GB)", "peak_gpu_gb", ".1f", "lower"),
    ("Runtime", "GPU energy (Wh)", "gpu_energy_wh", ".0f", "lower"),
    ("Runtime", "Mean GPU utilisation (%)", "mean_gpu_util_pct", ".0f", "-"),
    ("Runtime", "Trained model on disk (MB)", "model_mb", ".0f", "lower"),
    ("Data", "Train rows the generator saw", "train_rows_seen", ",.0f", "-"),
    ("Data", "Synthetic rows scored (1×)", "synthetic_rows_scored", ",.0f", "-"),
]


def fmt(value, spec: str) -> str:
    if value is None:
        return "n/a"
    return format(value, spec) if spec else str(value)


def flatten(obj, prefix: str = "") -> dict:
    """Nested dict -> {"a.b": scalar}. Used for NSS's own summary, whose layout we do not control."""
    if isinstance(obj, dict):
        return {k2: v2 for k, v in obj.items() for k2, v2 in flatten(v, f"{prefix}{k}.").items()}
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return {prefix[:-1]: obj}
    return {}


def write_tables(rows: dict[str, dict]) -> None:
    modes = list(rows)
    lines = ["# TabDDPM vs NeMo Safe Synthesizer · measured results", "",
             "Built by `06_compare_generators.py`. Same train split, same medical-rule filter, same scorecard (03b),",
             "same locked real test set for every column. Method notes and caveats: `../../NSS_COMPARISON.md`.", ""]
    section = None
    for sec, name, key, spec, good in TABLE:
        if sec != section:
            section = sec
            lines += ["", f"## {sec}", "", "| Metric | " + " | ".join(label(m) for m in modes) + " | Good is |",
                      "|---|" + "---:|" * len(modes) + "---|"]
        lines.append(f"| {name} | " + " | ".join(fmt(rows[m].get(key), spec) for m in modes) + f" | {good} |")

    native = {m: flatten((load(RESULTS / f"02_generation_summary_{m}.json") or {}).get("nss_summary", {})) for m in modes}
    native = {m: v for m, v in native.items() if v}
    if native:
        keys = sorted({k for v in native.values() for k in v})
        lines += ["", "## NSS's own report (as NSS computes it, on its own 5% holdout)", "",
                  "Scores are out of 10. Full report: `nss_native/<run>_evaluation_report.html`.", "",
                  "| NSS summary field | " + " | ".join(label(m) for m in native) + " |", "|---|" + "---:|" * len(native)]
        lines += [f"| `{k}` | " + " | ".join(fmt(native[m].get(k), ".4g") for m in native) + " |" for k in keys]

    lines += ["", "## Reading notes", "",
              "- **Train rows seen.** NSS keeps 5% of its input (max 2,000 rows) out of training for its own privacy "
              "tests. The scorecard counts only the rows a generator trained on as `train`, so privacy metrics are exact.",
              "- **Peak GPU memory** is the whole device. NSS's generation engine (vLLM) reserves about 80% of the card "
              "up front, so its peak shows a reservation, not a need. NVIDIA lists an A100 as the minimum.",
              "- **Generation time** is the time of the sampling call divided by the rows it returned, scaled to one "
              "train-size table. It includes loading the model.",
              "- **One run per generator, so no error bars.** Treat gaps of a few thousandths of AUC as ties until "
              "the runs are repeated with other seeds."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps({"labels": {m: label(m) for m in modes}, "runs": rows}, indent=2))
    print("\n".join(lines))


# ---------------------------------------------------------------- step 3: figures

def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def legend(fig, entries: dict[str, str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in entries.values()]
    fig.legend(handles, list(entries), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(entries))


def grouped_bars(ax, categories: list, series: dict[str, list[float]], palette: dict[str, str]) -> None:
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        x = np.arange(len(categories)) - 0.4 + width * (i + 0.5)
        ax.bar(x, values, width * 0.88, color=palette[name])  # 0.88 leaves a surface gap between neighbours
    ax.set_xticks(range(len(categories)), [str(c) for c in categories])
    ax.grid(axis="x", visible=False)


def fig_headline(rows: dict[str, dict], pal: dict[str, str]) -> None:
    first = next(iter(rows.values()))
    panels = [  # title, key, subtitle, (reference value, its label)
        ("TSTR AUC", "tstr_auc", "utility · higher is better", (first["trtr_auc"], "real data")),
        ("α-precision", "alpha_precision", "fidelity · higher is better", (0.90, "target")),
        ("β-recall", "beta_recall", "coverage · higher is better", None),
        ("Correlation gap", "correlation_gap", "fidelity · lower is better", (0.10, "limit")),
        ("Membership-inference AUC", "mia_auc", "privacy · lower is safer", (0.50, "coin flip")),
        ("DCR score", "dcr_score", "privacy · closer to ideal is safer", (first["dcr_ideal"], "ideal")),
        ("Exact copies vs holdout", "exact_copies_ratio", "privacy · train ÷ holdout, lower is safer", (1.2, "limit")),
        ("Rows passing medical rules", "rule_pass_rate", "validity · higher is better", (0.98, "target")),
    ]
    modes = list(rows)
    fig, axes = plt.subplots(2, 4, figsize=(15, 1.6 + 0.9 * len(modes)))
    for ax, (title, key, sub, ref) in zip(axes.flat, panels):
        values = [rows[m][key] for m in modes]
        y = np.arange(len(modes))[::-1]
        span = [*values, *([ref[0]] if ref else [])]
        pad = max(max(span) - min(span), 0.02 * abs(max(span)), 1e-6)
        ax.set_xlim(min(span) - 0.25 * pad, max(span) + 0.75 * pad)
        if ref:
            ax.axvline(ref[0], color=MUTED, lw=1)
            ax.annotate(f"{ref[1]} {ref[0]:.3g}", (ref[0], 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", va="top", fontsize=8, color=INK2)
        ax.scatter(values, y, s=110, c=[pal[m] for m in modes], edgecolors=SURFACE, linewidths=2, zorder=3)
        for v, yy in zip(values, y):
            ax.annotate(f"{v:.3f}", (v, yy), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(y, [label(m) for m in modes])
        ax.set_ylim(-0.6, len(modes) - 0.2)
        ax.set_title(f"{title}\n", pad=2)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    fig.tight_layout(w_pad=2.5, h_pad=2)
    save(fig, "headline_metrics.png")


def fig_runtime(rows: dict[str, dict], pal: dict[str, str]) -> None:
    panels = [("Training", "train_minutes", "minutes"), ("Generating one train-size table", "generate_minutes_per_train_size", "minutes"),
              ("Peak GPU memory", "peak_gpu_gb", "GB, whole device"), ("GPU energy, whole run", "gpu_energy_wh", "Wh"),
              ("Trained model on disk", "model_mb", "MB")]
    panels = [p for p in panels if all(rows[m].get(p[1]) is not None for m in rows)]
    modes = list(rows)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 1.2 + 0.55 * len(modes)), squeeze=False)
    for ax, (title, key, unit) in zip(axes.flat, panels):
        values = [rows[m][key] for m in modes]
        y = np.arange(len(modes))[::-1]
        ax.barh(y, values, height=0.38, color=[pal[m] for m in modes])
        for v, yy in zip(values, y):
            ax.annotate(f"{v:,.2f}" if v < 10 else f"{v:,.0f}", (v, yy), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=9)
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_yticks(y, [label(m) for m in modes])
        ax.set_ylim(-0.6, len(modes) - 0.4)
        ax.set_title(f"{title}\n", pad=2)
        ax.text(0, 1.03, unit, transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(w_pad=2.5)
    save(fig, "runtime.png")


def fig_privacy_utility(rows: dict[str, dict], pal: dict[str, str]) -> None:
    first = next(iter(rows.values()))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs, ys = [r["mia_auc"] for r in rows.values()], [r["tstr_auc"] for r in rows.values()]
    ax.axhline(first["trtr_auc"], color=MUTED, lw=1)
    ax.axvline(0.5, color=MUTED, lw=1)
    xpad, ypad = max(max(xs) - 0.5, 0.01), max(first["trtr_auc"] - min(ys), 0.005)
    ax.set_xlim(0.5 - 0.35 * xpad, max(xs) + 0.9 * xpad)
    ax.set_ylim(min(ys) - 0.5 * ypad, first["trtr_auc"] + 0.5 * ypad)
    ax.annotate("real-data baseline", (1, first["trtr_auc"]), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.annotate("coin flip", (0.5, 0), xycoords=("data", "axes fraction"), xytext=(4, 4), textcoords="offset points",
                fontsize=8, color=INK2)
    for m, r in rows.items():
        ax.scatter(r["mia_auc"], r["tstr_auc"], s=130, color=pal[m], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(label(m), (r["mia_auc"], r["tstr_auc"]), xytext=(10, 0), textcoords="offset points", va="center")
    ax.set_xlabel("Membership-inference AUC (lower = safer)")
    ax.set_ylabel("TSTR AUC on real test set (higher = more useful)")
    ax.set_title("Privacy vs utility")
    save(fig, "privacy_utility.png")


def fig_correlation(modes: list[str]) -> None:
    cards = {m: load(RESULTS / f"03b_scorecard_{m}.json") for m in modes}
    gaps = {}
    for m, card in cards.items():
        mats = card["fidelity"]["correlation_matrices"]
        gaps[m] = pd.DataFrame(mats["real"]) - pd.DataFrame(mats["synthetic"])
    lim = max(0.02, max(float(g.abs().max().max()) for g in gaps.values()))
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
    fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes) + 1, 4.8), squeeze=False)
    for i, (ax, (m, g)) in enumerate(zip(axes.flat, gaps.items())):
        im = ax.imshow(g.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(g.columns)), g.columns, rotation=90)
        ax.set_yticks(range(len(g.index)), g.index if i == 0 else [])
        gap = cards[m]["fidelity"]["correlation"]["corr_mat_diff"]
        ax.set_title(f"{label(m)}\ncorrelation gap {gap:.3f}", fontsize=10)
        ax.grid(visible=False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="real − synthetic association")
    save(fig, "correlation_gaps.png")


def fig_risk_curves(modes: list[str], pal: dict[str, str]) -> None:
    cards = {m: load(RESULTS / f"03b_scorecard_{m}.json") for m in modes}
    order = {"systolic BP": ["<120", "120-139", ">=140"], "age": ["<50", "50-59", ">=60"],
             "cholesterol": ["normal", "above normal", "well above"]}
    palette = {"Real patients": REAL, **{label(m): pal[m] for m in modes}}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (factor, cats) in zip(axes, order.items()):
        curves = {m: cards[m]["relationships"]["risk_curves"][factor] for m in modes}
        series = {"Real patients": [curves[modes[0]]["real"][c] for c in cats],
                  **{label(m): [curves[m]["synthetic"].get(c, 0) for c in cats] for m in modes}}
        grouped_bars(ax, cats, series, palette)
        ax.set_title(f"Share with heart disease, by {factor}")
        ax.set_ylim(0, 1)
    legend(fig, palette)
    fig.tight_layout()
    save(fig, "risk_curves.png")


def fig_distributions(modes: list[str], pal: dict[str, str]) -> None:
    real_path = sh.DATA_DIR / f"{sh.TABLE}.csv"
    paths = {m: RUNS / m / "synth_scale_1" / sh.TABLE / "_final" / f"{sh.TABLE}_synthetic_filtered.csv" for m in modes}
    missing = [str(p) for p in [real_path, *paths.values()] if not p.exists()]
    if missing:
        print("skipping distributions.png, these CSVs are not on this machine:", *missing, sep="\n  ")
        return
    real, synth = pd.read_csv(real_path), {m: pd.read_csv(p) for m, p in paths.items()}
    palette = {"Real patients": REAL, **{label(m): pal[m] for m in modes}}
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, col in zip(axes.flat, sh.ALL_COLS):
        if col in sh.NUM_COLS:
            lo, hi = real[col].quantile([0.005, 0.995])
            edges = np.linspace(lo, hi, 41)
            ax.stairs(np.histogram(real[col], edges, density=True)[0], edges, fill=True, color=REAL, alpha=0.45)
            for m, df in synth.items():
                ax.stairs(np.histogram(df[col], edges, density=True)[0], edges, color=pal[m], lw=2)
            ax.set_yticks([])
        else:
            cats = sorted(real[col].unique())
            share = lambda df: df[col].value_counts(normalize=True).reindex(cats, fill_value=0).tolist()  # noqa: E731
            grouped_bars(ax, cats, {"Real patients": share(real), **{label(m): share(df) for m, df in synth.items()}}, palette)
        ax.set_title(col)
    legend(fig, palette)
    fig.tight_layout()
    save(fig, "distributions.png")


def compare(modes: list[str]) -> None:
    rows = {m: r for m in modes if (r := collect(m))}
    if not rows:
        print("nothing to compare yet: no run has both a generation summary and a 03b scorecard")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    write_tables(rows)
    shown = list(rows)[:4]  # the palette has four validated colours; the tables carry every run
    if len(rows) > 4:
        print("figures show the first four runs only:", shown, "(use --modes to choose)")
    pal, sub = colors(shown), {m: rows[m] for m in shown}
    fig_headline(sub, pal)
    fig_runtime(sub, pal)
    fig_privacy_utility(sub, pal)
    fig_correlation(shown)
    fig_risk_curves(shown, pal)
    fig_distributions(shown, pal)


def default_modes() -> list[str]:
    nss = sorted(p.stem.removeprefix("02_generation_summary_") for p in RESULTS.glob("02_generation_summary_nss_*.json"))
    nss = [m for m in nss if m != "nss_smoke"]
    return ["tabddpm", *sorted(nss, key=lambda m: (m not in FIXED, m))]  # nss_default, nss_dp8, then the rest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", nargs="+", help="runs to compare (default: tabddpm + every nss_* run except nss_smoke)")
    ap.add_argument("--no-run", action="store_true", help="do not execute notebooks, only rebuild tables and figures")
    args = ap.parse_args()
    chosen = args.modes or default_modes()
    print("comparing:", chosen)
    if not args.no_run:
        run_notebooks(chosen)
    compare(chosen)
