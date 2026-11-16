"""SynthHeart · 05 · Generate synthetic patients with NVIDIA NeMo Safe Synthesizer (NSS).

The NSS counterpart of notebook 02. It trains NSS on the same train split TabDDPM used (data/cardio/cardio.csv),
generates 2x the train size, applies the same rounding + medical-range filter, and writes the same hand-off files
as notebook 02, so the scorecard (03b) runs on NSS output unchanged.

Run on the AI Factory with the NSS environment (see setup_nss.sh), from implementations/synthheart:

    .venv-nss/bin/python 05_nss_generate.py --name nss_smoke --set generation.num_records=2000     # quick check
    nohup .venv-nss/bin/python 05_nss_generate.py --name nss_default > logs/05_nss_default.log 2>&1 &

`--set section.key=value` overrides any NSS parameter (repeatable). Examples:
    --set privacy.dp_enabled=true --set privacy.epsilon=8.0 --set training.batch_size=16      differential privacy
    --set training.num_input_records_to_sample=123405                                         train ~3 passes
    --set training.attn_implementation=sdpa                                                   if the Kernels Hub is blocked

Outputs
  runs/<name>/synth_scale_{1,2}/cardio/_final/cardio_synthetic_filtered.csv   hand-off to 03b (git-ignored)
  runs/<name>/train_seen.csv            the rows NSS actually trained on (it holds out 5% for its own report)
  runs/<name>/nss_artifacts/            NSS's own run folder: adapter, logs, splits (git-ignored)
  results/02_generation_summary_<name>.json                 timings, filter pass rate, config (committed)
  results/comparison/gpu_<name>.json                        GPU memory / utilisation / energy (committed)
  results/comparison/nss_native/<name>_*                    NSS's own HTML report, metrics and resolved config

A finished run is skipped; delete results/02_generation_summary_<name>.json to run it again.

    python 05_nss_generate.py --self-test      # checks the clean-up logic; needs pandas only, no GPU, no NSS
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthheart_common as sh  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402


STAGES = ["process_data", "train", "generate", "evaluate", "save_results"]


def apply_overrides(config: dict, pairs: list[str]) -> dict:
    """Apply `section.key=value` strings. Values are read as JSON (true, 8.0, null) and fall back to text."""
    for pair in pairs:
        key, sep, value = pair.partition("=")
        assert sep and "." in key, f"--set needs section.key=value, got {pair!r}"
        *parents, leaf = key.split(".")
        node = config
        for parent in parents:
            node = node.setdefault(parent, {})
        try:
            node[leaf] = json.loads(value)
        except ValueError:
            node[leaf] = value
    return config


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """NSS writes each record as text, so first drop rows that are not numbers, then snap types and apply
    the same medical-range rules notebook 02 applies to TabDDPM output."""
    df = raw.reindex(columns=sh.ALL_COLS).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    parsed = df.dropna()
    kept, fails = sh.apply_rules(sh.round_synthetic(parsed))
    failed = {k: int(v) for k, v in fails[fails > 0].items()}
    if len(parsed) < len(df):
        failed["not a number"] = len(df) - len(parsed)
    return kept, {"generated": len(df), "kept": len(kept), "pass_rate": round(len(kept) / len(df), 4),
                  "rules_failed": failed}


def self_test() -> None:
    cfg = apply_overrides({"data": {"random_state": 42}}, ["privacy.dp_enabled=true", "privacy.epsilon=8.0",
                                                           "training.attn_implementation=sdpa", "data.holdout=0"])
    assert cfg == {"data": {"random_state": 42, "holdout": 0}, "privacy": {"dp_enabled": True, "epsilon": 8.0},
                   "training": {"attn_implementation": "sdpa"}}, cfg
    good = dict(age="52.34", height="168", weight=70.0, ap_hi=120, ap_lo=80, gender=1, cholesterol=0, gluc=0,
                smoke=0, alco=0, active=1, cardio=0)
    raw = pd.DataFrame([good, {**good, "ap_hi": "high"}, {**good, "ap_lo": 130}, {**good, "gluc": 7}])
    kept, report = clean(raw.assign(extra="ignored"))
    assert len(kept) == 1 and list(kept.columns) == sh.ALL_COLS and kept.iloc[0]["age"] == 52.3, kept
    assert report == {"generated": 4, "kept": 1, "pass_rate": 0.25, "rules_failed": {
        "ap_hi <= ap_lo": 1, "gluc not in [0, 1, 2]": 1, "not a number": 1}}, report
    print("05 self-test passed")


def latest(root: Path, pattern: str) -> Path | None:
    """Newest file matching `pattern` under `root` (a re-run leaves older NSS run folders behind)."""
    return max(root.rglob(pattern), key=lambda p: p.stat().st_mtime, default=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", help="run name, e.g. nss_default. Must start with 'nss'")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value", help="override an NSS parameter")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    name = args.name or ""
    assert name.startswith("nss"), "--name must start with 'nss' (06_compare_generators.py finds NSS runs by that prefix)"
    out = sh.RESULTS_DIR / f"02_generation_summary_{name}.json"
    if out.exists():
        print(f"{name}: already done ({out.name} exists). Delete it to run again.")
        return

    train = pd.read_csv(sh.DATA_DIR / f"{sh.TABLE}.csv")[sh.ALL_COLS]
    run_dir = sh.PROJECT / "runs" / name
    native = sh.RESULTS_DIR / "comparison" / "nss_native"
    for d in (run_dir, native):
        d.mkdir(parents=True, exist_ok=True)

    # Everything not listed here is an NSS default (SmolLM3-3B + LoRA, 5% holdout for its own report, no DP).
    # PII replacement is off: this table has no names, addresses or free text, and its column classifier would
    # otherwise call an NVIDIA-hosted LLM endpoint.
    config = apply_overrides({
        "replace_pii": None,
        "data": {"random_state": sh.SEED},
        "generation": {"num_records": 2 * len(train)},  # 1x and 2x the train size, like notebook 02
    }, args.set)
    import yaml  # ships with NSS

    cfg_path = run_dir / "nss_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"{name}: training on {len(train):,} rows with\n{cfg_path.read_text()}", flush=True)

    os.environ.setdefault("NEMO_TELEMETRY_ENABLED", "false")  # no usage telemetry from the AI Factory
    from nemo_safe_synthesizer.config import SafeSynthesizerParameters
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synth = SafeSynthesizer(SafeSynthesizerParameters.from_yaml(str(cfg_path)),
                            save_path=run_dir / "nss_artifacts").with_data_source(train)
    minutes = {}
    with GpuMonitor(sh.RESULTS_DIR / "comparison" / f"gpu_{name}.json") as gpu:
        for stage in STAGES:
            t0 = time.time()
            getattr(synth, stage)()
            minutes[stage] = round((time.time() - t0) / 60, 2)
            gpu.mark(stage)
            print(f"{name}: {stage} done in {minutes[stage]} min", flush=True)
    results = synth.results
    raw = results.synthetic_data
    assert raw is not None and len(raw), "NSS returned no synthetic rows. Check the log above."
    raw.to_csv(run_dir / "nss_synthetic_raw.csv", index=False)

    # Hand-off files, same layout as notebook 02.
    # ponytail: one generation call. 1x is the first len(train) rows of it, 2x is all of it (rows are independent
    # draws). Two separate calls only if 1x and 2x must be independent samples.
    filter_report = {}
    for scale, rows in {"synth_scale_1": raw.iloc[: len(train)], "synth_scale_2": raw}.items():
        kept, filter_report[scale] = clean(rows)
        path = run_dir / scale / sh.TABLE / "_final" / f"{sh.TABLE}_synthetic_filtered.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        kept.to_csv(path, index=False)
        print(f"{scale}: kept {len(kept):,}/{len(rows):,} ({filter_report[scale]['pass_rate']:.1%})")

    # The rows NSS trained on. It holds out part of the input for its own privacy tests, so the scorecard must
    # treat only these rows as "seen by the generator".
    train_rows_seen, seen_path = len(train), latest(run_dir / "nss_artifacts", "dataset/training.csv")
    if seen_path:
        seen = pd.read_csv(seen_path)
        if set(sh.ALL_COLS) <= set(seen.columns) and len(seen) < len(train):
            in_train = seen[sh.ALL_COLS].merge(train.drop_duplicates(), on=sh.ALL_COLS, how="left", indicator=True)
            if (in_train["_merge"] == "both").mean() > 0.999:
                seen[sh.ALL_COLS].to_csv(run_dir / "train_seen.csv", index=False)
                train_rows_seen = len(seen)
    print(f"rows NSS trained on: {train_rows_seen:,} of {len(train):,}")

    # NSS's own report, metrics and fully resolved config, kept next to our results.
    copied = {}
    for pattern in ["evaluation_report.html", "evaluation_metrics.json", "safe-synthesizer-config.json"]:
        src = latest(run_dir / "nss_artifacts", pattern)
        if src:
            copied[pattern] = shutil.copy(src, native / f"{name}_{pattern}")
    if "evaluation_report.html" not in copied and results.evaluation_report_html:
        (native / f"{name}_evaluation_report.html").write_text(results.evaluation_report_html, encoding="utf-8")
    adapter = latest(run_dir / "nss_artifacts", "adapter")
    adapter_mb = sum(f.stat().st_size for f in adapter.rglob("*") if f.is_file()) / 1e6 if adapter else None

    generated = filter_report["synth_scale_2"]["generated"]
    summary = {
        "run_mode": name,
        "generator": "nemo-safe-synthesizer",
        "nss_version": importlib.metadata.version("nemo-safe-synthesizer"),
        "nss_overrides": config,  # the resolved config is in comparison/nss_native/<name>_safe-synthesizer-config.json
        "train_rows": int(len(train)),
        "train_rows_seen": int(train_rows_seen),
        "stage_minutes": minutes,
        "train_minutes": minutes["train"],
        "sample_minutes": {"synth_scale_2": minutes["generate"]},  # one call generated all rows
        "generated_rows_per_second": round(generated / max(minutes["generate"] * 60, 1e-9), 1),
        "adapter_mb": round(adapter_mb, 1) if adapter_mb else None,
        "filter": filter_report,
        "nss_summary": results.summary.model_dump(),  # NSS's own scores and timings
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "nss_summary"}, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
