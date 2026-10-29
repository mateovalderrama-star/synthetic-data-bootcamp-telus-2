"""Check that predictor.js gives the same risk and explanations as XGBoost in Python.

Scores every row of the locked test set both ways (JavaScript via Node) and compares.

    uv run python implementations/synthheart/webapp/verify_predictor.py --model path/to/risk_model_synthetic.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import synthheart_common as sh  # noqa: E402
from build_webapp import FEATURES, compact_model  # noqa: E402

NODE_SCRIPT = """
const fs = require("fs");
const P = require(process.argv[2]);
const model = P.prepare(JSON.parse(fs.readFileSync(process.argv[3], "utf8")));
const rows = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const out = rows.map((r) => { const p = P.predict(model, r); return [p.risk, p.bias, ...model.features.map((f) => p.contributions[f])]; });
fs.writeFileSync(process.argv[5], JSON.stringify(out));
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=HERE / "models" / "risk_model_synthetic.json")
    args = ap.parse_args()

    X, _ = sh.xy(pd.read_csv(sh.DATA_DIR / f"{sh.TABLE}_test.csv"))
    X = X[FEATURES]
    booster = xgb.Booster(model_file=str(args.model))
    dm = xgb.DMatrix(X)
    py_risk = booster.predict(dm)
    py_contrib = booster.predict(dm, pred_contribs=True, approx_contribs=True)  # last column = bias

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.json").write_text(json.dumps(compact_model(args.model)))
        (tmp / "rows.json").write_text(X.to_json(orient="records"))
        (tmp / "check.js").write_text(NODE_SCRIPT)
        subprocess.run(["node", str(tmp / "check.js"), str(HERE / "predictor.js"), str(tmp / "model.json"),
                        str(tmp / "rows.json"), str(tmp / "out.json")], check=True)
        js = np.array(json.loads((tmp / "out.json").read_text()))

    risk_diff = np.abs(js[:, 0] - py_risk)
    contrib_diff = np.abs(js[:, 2:] - py_contrib[:, :-1])
    bias_diff = np.abs(js[:, 1] - py_contrib[:, -1])
    print(f"rows checked:                 {len(X):,}")
    print(f"max |risk JS - risk Python|:  {risk_diff.max():.2e}")
    print(f"max |contribution diff|:      {contrib_diff.max():.2e}")
    print(f"max |bias diff|:              {bias_diff.max():.2e}")
    ok = risk_diff.max() < 1e-4 and contrib_diff.max() < 1e-3 and bias_diff.max() < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
