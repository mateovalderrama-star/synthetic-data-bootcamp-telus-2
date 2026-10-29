"""SynthHeart showcase dashboard: predict · compare · scorecard.

    uv run --with streamlit streamlit run implementations/synthheart/dashboard/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synthheart_common as sh

XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           eval_metric="logloss", random_state=sh.SEED, n_jobs=-1)  # identical to notebooks 01 and 03

st.set_page_config(page_title="SynthHeart", layout="wide")
modes = sorted(p.stem.removeprefix("03_scorecard_") for p in sh.RESULTS_DIR.glob("03_scorecard_*.json")) or ["full"]
mode = st.sidebar.selectbox("Run", modes, index=modes.index("full") if "full" in modes else 0)
RUN_DIR = sh.PROJECT / "runs" / mode
st.title("SynthHeart · privacy-safe synthetic heart patients")


@st.cache_resource
def load_model(mode: str) -> tuple[XGBClassifier, str]:
    path = sh.PROJECT / "runs" / mode / "models" / "risk_model_synthetic.json"
    model = XGBClassifier()
    if path.exists():
        model.load_model(path)
        return model, "trained **only on synthetic patients**"
    X, y = sh.xy(pd.read_csv(sh.DATA_DIR / f"{sh.TABLE}.csv"))
    return XGBClassifier(**XGB).fit(X, y), "**real-data baseline** (synthetic model not found; run notebook 03)"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


predict, compare, scorecard = st.tabs(["1 · Predict", "2 · Compare", "3 · Scorecard"])

with predict:
    st.warning("Demo only, not a medical device. Do not use for clinical decisions.")
    model, source = load_model(mode)
    st.caption(f"Risk model: XGBoost {source}.")
    c1, c2, c3 = st.columns(3)
    patient = {
        "age": c1.slider("Age (years)", 30, 70, 52),
        "height": c1.number_input("Height (cm)", *sh.RANGES["height"], 168),
        "weight": c1.number_input("Weight (kg)", *sh.RANGES["weight"], 72),
        "ap_hi": c2.number_input("Systolic BP (mmHg)", *sh.RANGES["ap_hi"], 120),
        "ap_lo": c2.number_input("Diastolic BP (mmHg)", *sh.RANGES["ap_lo"], 80),
        "gender": ["Woman", "Man"].index(c2.radio("Sex", ["Woman", "Man"], horizontal=True)),
        "cholesterol": ["Normal", "Above normal", "Well above normal"].index(
            c3.selectbox("Cholesterol", ["Normal", "Above normal", "Well above normal"])),
        "gluc": ["Normal", "Above normal", "Well above normal"].index(
            c3.selectbox("Glucose", ["Normal", "Above normal", "Well above normal"])),
        "smoke": int(c3.checkbox("Smokes")),
        "alco": int(c3.checkbox("Drinks alcohol")),
        "active": int(c3.checkbox("Physically active", value=True)),
        sh.TARGET: 0,  # placeholder so sh.xy can split it off
    }
    X, _ = sh.xy(pd.DataFrame([patient])[sh.ALL_COLS])
    st.metric("Estimated cardiovascular disease risk", f"{model.predict_proba(X)[0, 1]:.0%}")

with compare:
    real = load_csv(sh.DATA_DIR / f"{sh.TABLE}.csv")
    synth = load_csv(RUN_DIR / "synth_scale_1" / sh.TABLE / "_final" / f"{sh.TABLE}_synthetic_filtered.csv")
    if real is None or synth is None:
        st.info(f"Real train split or synthetic CSV for run `{mode}` not found here. Run notebooks 01 and 02 "
                "(the CSVs are git-ignored, so this tab only works where they were generated).")
    else:
        st.caption(f"Real train split ({len(real):,} rows) vs synthetic 1× ({len(synth):,} rows). Shares, not counts.")
        cols = st.columns(3)
        for i, col in enumerate(sh.ALL_COLS):
            with cols[i % 3]:
                st.markdown(f"**{col}**")
                if col in sh.NUM_COLS:
                    edges = np.histogram_bin_edges(real[col].clip(*real[col].quantile([0.005, 0.995])), bins=30)
                    share = lambda s: np.histogram(s.clip(edges[0], edges[-1]), edges)[0] / len(s)
                    df = pd.DataFrame({"real": share(real[col]), "synthetic": share(synth[col])},
                                      index=np.round((edges[:-1] + edges[1:]) / 2, 1))
                else:
                    df = pd.DataFrame({"real": real[col].value_counts(normalize=True),
                                       "synthetic": synth[col].value_counts(normalize=True)}).fillna(0).sort_index()
                st.bar_chart(df, stack=False, height=220)

with scorecard:
    # Prefer 03b (exact copies judged against holdout at equal size); fall back to plain 03.
    path = next((p for p in (sh.RESULTS_DIR / f"03b_scorecard_{mode}.json", sh.RESULTS_DIR / f"03_scorecard_{mode}.json")
                 if p.exists()), sh.RESULTS_DIR / f"03_scorecard_{mode}.json")
    if not path.exists():
        st.info(f"`{path.name}` not found. Run notebook 03 for this run.")
    else:
        card = json.loads(path.read_text())
        st.caption(f"Source: `{path.name}`")
        util, priv = card["utility"], card["privacy"]
        tstr = util["tstr_auc"][card["scored"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TSTR AUC (synthetic-trained)", f"{tstr:.3f}", f"{tstr - util['trtr_auc']:+.3f} vs real")
        m2.metric("Membership attack AUC", f"{priv['mia']['auc']:.3f}", "0.5 = coin flip", delta_color="off")
        m3.metric("DCR score", f"{priv['dcr']['dcr_score']:.3f}", f"ideal {priv['dcr']['ideal']:.2f}", delta_color="off")
        c = priv["exact_copies"]
        same_size = c.get("train_sample_holdout_size_mean", c["train_sample_holdout_size"])
        m4.metric("Exact copies of train", c["train"],
                  f"{same_size:g} vs {c['holdout']} same-size train / holdout", delta_color="off")
        st.dataframe(pd.DataFrame([{"check": k, "status": "🟢 pass" if v["pass"] else "🔴 miss", "value": v["value"],
                                    "target": v["target"]} for k, v in card["acceptance"].items()]),
                     hide_index=True, width="stretch")
    ablation = sh.RESULTS_DIR / "04_steps_ablation.json"
    if ablation.exists():
        st.subheader("Training-steps ablation")
        st.dataframe(pd.DataFrame(json.loads(ablation.read_text())).astype(str), width="stretch")
