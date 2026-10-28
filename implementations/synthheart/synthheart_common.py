"""Shared constants and helpers for the SynthHeart notebooks.

Every notebook imports from here, so the cleaning rules, column order and file paths
are identical for real data (cleaning) and synthetic data (filtering).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TABLE = "cardio"  # table name used by midst-toolkit (file names: cardio.csv, cardio_domain.json, ...)

# Column order matters: midst-toolkit's meta_info.json refers to columns by index.
NUM_COLS = ["age", "height", "weight", "ap_hi", "ap_lo"]
CAT_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
TARGET = "cardio"
ALL_COLS = NUM_COLS + CAT_COLS

# Medical-range rules. Applied to real data in notebook 01 (cleaning) and to synthetic data
# later (filtering), so both sides are judged by exactly the same standard.
RANGES = {
    "age": (18, 100),  # years
    "height": (130, 210),  # cm
    "weight": (35, 200),  # kg
    "ap_hi": (80, 250),  # systolic, mmHg
    "ap_lo": (40, 160),  # diastolic, mmHg
}
CATEGORY_LEVELS = {  # allowed values after encoding (0-based)
    "gender": [0, 1],
    "cholesterol": [0, 1, 2],
    "gluc": [0, 1, 2],
    "smoke": [0, 1],
    "alco": [0, 1],
    "active": [0, 1],
    "cardio": [0, 1],
}


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) until the folder with pyproject.toml is found."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "implementations").exists():
            return p
    raise FileNotFoundError("Could not find the repo root (pyproject.toml). Open the notebook inside the repo.")


REPO = find_repo_root()
RAW_CSV = REPO / "data" / "cardio_train.csv"
PROJECT = REPO / "implementations" / "synthheart"
DATA_DIR = PROJECT / "data" / TABLE  # prepared splits + metadata (CSVs are git-ignored; re-run notebook 01)
RESULTS_DIR = PROJECT / "results"


LOGS_DIR = PROJECT / "logs"


def start_progress_log(run_mode: str, total_steps: int, sample_scales: list[float]):
    """Mirror midst-toolkit's log lines (e.g. "Step 500/20000 ...") plus our own milestones into a
    timestamped file, so `progress.py` can show a live progress bar even when the notebook runs headless.

    Returns a `stage(msg)` function for writing milestone lines.
    """
    import logging

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"02_{run_mode}_progress.log"
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    toolkit = logging.getLogger("midst_toolkit")
    for h in list(toolkit.handlers):  # avoid duplicate file handlers if the cell is re-run
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).name == path.name:
            toolkit.removeHandler(h)
    toolkit.addHandler(handler)

    def stage(msg: str) -> None:
        handler.emit(logging.LogRecord("synthheart", logging.INFO, "", 0, f"STAGE {msg}", None, None))
        handler.flush()

    scales = ",".join(f"{s:g}" for s in sample_scales)
    stage(f"RUN_START mode={run_mode} total_steps={total_steps} scales={scales}")
    print(f"progress log -> {path}  (watch with: uv run python progress.py --mode {run_mode} --watch)")
    return stage


def load_raw() -> pd.DataFrame:
    """Load the Kaggle file exactly as downloaded."""
    return pd.read_csv(RAW_CSV, sep=";")


def to_model_format(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw Kaggle columns -> the 12 columns we model, in a fixed order.

    - drops `id`
    - age: days -> years (1 decimal)
    - gender / cholesterol / gluc: 1-based codes -> 0-based labels
    """
    df = raw.drop(columns=["id"], errors="ignore").copy()
    df["age"] = (df["age"] / 365.25).round(1)
    for c in ["gender", "cholesterol", "gluc"]:
        df[c] = df[c] - 1
    return df[ALL_COLS]


def rule_violations(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule, True where a row breaks that rule."""
    out = {}
    for col, (lo, hi) in RANGES.items():
        out[f"{col} outside {lo}-{hi}"] = ~df[col].between(lo, hi)
    out["ap_hi <= ap_lo"] = df["ap_hi"] <= df["ap_lo"]
    for col, levels in CATEGORY_LEVELS.items():
        out[f"{col} not in {levels}"] = ~df[col].isin(levels)
    return pd.DataFrame(out, index=df.index)


def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (rows that pass every rule, count of failures per rule)."""
    v = rule_violations(df)
    return df[~v.any(axis=1)].copy(), v.sum()


def round_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Diffusion outputs continuous values; snap them back to the real data's types before filtering."""
    out = df.copy()
    out["age"] = out["age"].round(1)
    out["weight"] = out["weight"].round(1)
    for c in ["height", "ap_hi", "ap_lo"]:
        out[c] = out[c].round().astype(int)
    for c in CAT_COLS:
        out[c] = out[c].round().astype(int)
    return out[ALL_COLS]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for the downstream risk model. BMI is derived here, never generated."""
    out = df.copy()
    out["bmi"] = (out["weight"] / (out["height"] / 100) ** 2).round(1)
    return out


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a prepared table into model features and the target."""
    feats = add_features(df)
    return feats.drop(columns=[TARGET]), feats[TARGET].to_numpy()
