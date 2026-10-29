"""Build the SynthHeart web app: one self-contained page per design in designs/ (no server needed).

Embeds the synthetic-trained XGBoost risk model (compacted), the scorecard results, the shared components
(base.css, panels.html, app.js) and the predictor into each design, so predictions run in the browser and
nothing a nurse types leaves the page. dist/index.html is the design picker.

    uv run python implementations/synthheart/webapp/build_webapp.py                  # model from runs/full/models
    uv run python implementations/synthheart/webapp/build_webapp.py --model path.json
    uv run python implementations/synthheart/webapp/build_webapp.py --model x.json --test-model   # dev only, adds a warning banner
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import synthheart_common as sh  # noqa: E402


MAIN_DESIGN = "care"  # the team picked Design B (Care Dashboard)
FEATURES = [*sh.NUM_COLS, *[c for c in sh.CAT_COLS if c != sh.TARGET], "bmi"]  # order of sh.xy()


def compact_model(path: Path) -> dict:
    """XGBoost JSON model -> the small structure predictor.js walks."""
    learner = json.loads(path.read_text())["learner"]
    if learner["objective"]["name"] != "binary:logistic":
        raise ValueError(f"expected a binary:logistic model, got {learner['objective']['name']}")
    if learner["feature_names"] != FEATURES:
        raise ValueError(f"model features {learner['feature_names']} != expected {FEATURES}")
    base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))
    trees = []
    for t in learner["gradient_booster"]["model"]["trees"]:
        left, right, hess = t["left_children"], t["right_children"], t["sum_hessian"]
        mean = [0.0] * len(left)
        # Node mean = hessian-weighted mean of its leaves, which is what XGBoost's approximate
        # contributions use. Children always have larger ids than their parent, so go backwards.
        for n in reversed(range(len(left))):
            if left[n] == -1:
                mean[n] = t["split_conditions"][n]
            else:
                mean[n] = (mean[left[n]] * hess[left[n]] + mean[right[n]] * hess[right[n]]) / hess[n]
        trees.append({"f": t["split_indices"], "t": t["split_conditions"], "l": left, "r": right,
                      "m": [round(v, 7) for v in mean]})
    return {"features": FEATURES, "base_margin": math.log(base_score / (1 - base_score)), "trees": trees}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    # The committed copy in webapp/models/ (synthetic-trained, safe to share) wins; runs/ exists only on the AI Factory.
    default_model = next((p for p in [HERE / "models" / "risk_model_synthetic.json",
                                      sh.PROJECT / "runs" / "full" / "models" / "risk_model_synthetic.json"] if p.exists()),
                         HERE / "models" / "risk_model_synthetic.json")
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--test-model", action="store_true", help="model is a dev stand-in: show a warning banner")
    ap.add_argument("--out-dir", type=Path, default=HERE / "dist")
    args = ap.parse_args()
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}\nCopy risk_model_synthetic.json from the AI Factory "
                 "(runs/full/models/) or pass --model.")

    # Prefer the size-matched scorecard (03b) when it exists; fall back to notebook 03's.
    scorecard_path = next(p for p in [sh.RESULTS_DIR / "03b_scorecard_full.json",
                                      sh.RESULTS_DIR / "03_scorecard_full.json"] if p.exists())
    sc = load_json(scorecard_path)
    # Where most training patients fall (1st-99th percentile): the form warns outside it, where the model has
    # seen few patients. Only these aggregates leave the real data, never rows.
    train_csv = sh.DATA_DIR / f"{sh.TABLE}.csv"
    typical = None
    if train_csv.exists():
        q = pd.read_csv(train_csv)[sh.NUM_COLS].quantile([0.01, 0.99])
        typical = {c: [round(float(q.loc[0.01, c]), 1), round(float(q.loc[0.99, c]), 1)] for c in sh.NUM_COLS}
    data = {
        "model": compact_model(args.model),
        "test_model": args.test_model,
        "built": date.today().isoformat(),
        "scorecard_file": scorecard_path.name,
        "ranges": sh.RANGES,
        "typical": typical,
        "acceptance": sc["acceptance"],
        "rows": sc["rows"],
        "fidelity": {k: sc["fidelity"][k] for k in ["alpha_precision", "ks_tvd", "ci_overlap", "correlation",
                                                    "cardio_rate", "correlation_matrices"]},
        "utility": {k: sc["utility"][k] for k in ["trtr_auc", "tstr_auc", "tstr_accuracy", "auc_gap"]},
        "privacy": sc["privacy"],
        "relationships": sc["relationships"],
        "generation": load_json(sh.RESULTS_DIR / "02_generation_summary_full.json"),
        "baseline": load_json(sh.RESULTS_DIR / "baseline_metrics.json"),
    }
    parts = {
        "/*__BASE_CSS__*/": (HERE / "base.css").read_text(encoding="utf-8"),
        "<!--__PANELS__-->": (HERE / "panels.html").read_text(encoding="utf-8"),
        "/*__PREDICTOR__*/": (HERE / "predictor.js").read_text(encoding="utf-8"),
        "/*__APP__*/": (HERE / "app.js").read_text(encoding="utf-8"),
        "/*__DATA__*/": "window.SYNTHHEART = " + json.dumps(data, separators=(",", ":")) + ";",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for design in sorted((HERE / "designs").glob("*.html")):
        html = design.read_text(encoding="utf-8")
        for marker, content in parts.items():
            html = html.replace(marker, content)
        out = args.out_dir / design.name
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size / 1e6:.2f} MB)")
    # The chosen design is the app's front door; the others stay reachable from designs.html.
    (args.out_dir / "index.html").write_text((args.out_dir / f"{MAIN_DESIGN}.html").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"index.html = {MAIN_DESIGN}.html")
    print(f"model {args.model.name}{' (TEST MODEL)' if args.test_model else ''} | scorecard {scorecard_path.name}")


if __name__ == "__main__":
    main()
