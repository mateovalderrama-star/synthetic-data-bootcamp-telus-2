"""Live progress bar for notebook 02 (TabDDPM training + generation).

Reads logs/02_<mode>_progress.log, which notebook 02 writes while it runs.

    uv run python progress.py                 # one snapshot of the full run
    uv run python progress.py --watch         # refresh every 20 s (Ctrl+C to stop watching)
    uv run python progress.py --mode smoke    # the smoke run instead
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"
TS = "%Y-%m-%d %H:%M:%S"
STEP_RE = re.compile(r"Step (\d+)/(\d+) MLoss: ([\d.]+) GLoss: ([\d.]+)")


def bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def fmt(td: timedelta | None) -> str:
    if td is None:
        return "--"
    s = int(td.total_seconds())
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def parse(path: Path) -> dict:
    info = {"start": None, "total": None, "scales": [], "steps": [], "stages": {}, "last_line": "", "last_time": None}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " | " not in line:
            continue
        stamp, msg = line.split(" | ", 1)
        try:
            t = datetime.strptime(stamp, TS)
        except ValueError:
            continue
        info["last_line"], info["last_time"] = msg, t
        if msg.startswith("STAGE "):
            name, _, rest = msg[6:].partition(" ")
            info["stages"][name] = (t, rest)
            if name == "RUN_START":
                info["start"] = t
                info["total"] = int(re.search(r"total_steps=(\d+)", rest).group(1))
                info["scales"] = re.search(r"scales=([\d.,]+)", rest).group(1).split(",")
        m = STEP_RE.search(msg)
        if m:
            info["steps"].append((t, int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return info


def render(info: dict, mode: str) -> str:
    now = datetime.now()
    st = info["stages"]
    out = [f"SynthHeart · notebook 02 · {mode} run", ""]
    if info["start"] is None:
        return "Run has not started yet (no RUN_START in the log)."
    elapsed = now - info["start"]

    # --- 1. training ---
    total = info["total"] or 1
    if "TRAIN_DONE" in st:
        t_train = st["TRAIN_DONE"][0] - st.get("TRAIN_START", (info["start"],))[0]
        out.append(f"1 · Training   [{bar(1)}] 100%   done in {fmt(t_train)}")
    elif info["steps"]:
        t, step, tot, ml, gl = info["steps"][-1]
        t0 = st.get("TRAIN_START", (info["start"],))[0]
        rate = (t - t0).total_seconds() / step if step else 0
        eta = timedelta(seconds=rate * (tot - step))
        out.append(f"1 · Training   [{bar(step / tot)}] {step / tot:4.0%}   step {step:,}/{tot:,}   "
                   f"ETA {fmt(eta)}   loss M {ml:.3f} G {gl:.3f}")
        if len(info["steps"]) > 1:
            first = info["steps"][0]
            out.append(f"               loss trend: start {first[3] + first[4]:.3f} → now {ml + gl:.3f}")
    elif "TRAIN_START" in st:
        out.append(f"1 · Training   [{bar(0)}]   0%   started {fmt(now - st['TRAIN_START'][0])} ago "
                   f"(first update after 500 steps)")
    else:
        out.append(f"1 · Training   [{bar(0)}]   0%   waiting (loading data)")

    # --- 2. sampling, one line per scale ---
    train_secs = None
    if "TRAIN_DONE" in st:
        train_secs = (st["TRAIN_DONE"][0] - st.get("TRAIN_START", (info["start"],))[0]).total_seconds()
    for i, sc in enumerate(info["scales"], start=2):
        key = f"synth_scale_{sc}"
        if f"SAMPLE_DONE_{key}" in st:
            dt = st[f"SAMPLE_DONE_{key}"][0] - st[f"SAMPLE_START_{key}"][0]
            out.append(f"{i} · Sample {sc}×   [{bar(1)}] 100%   done in {fmt(dt)}")
        elif f"SAMPLE_START_{key}" in st:
            running = now - st[f"SAMPLE_START_{key}"][0]
            out.append(f"{i} · Sample {sc}×   [{'▒' * 30}]  running {fmt(running)} (no step updates while sampling)")
        else:
            out.append(f"{i} · Sample {sc}×   [{bar(0)}]   0%   waiting")

    # --- 3. filter + summary ---
    n = len(info["scales"]) + 2
    if "RUN_DONE" in st:
        out.append(f"{n} · Filter + summary [{bar(1)}] 100%")
        out += ["", f"✅ FINISHED in {fmt(st['RUN_DONE'][0] - info['start'])}. Next: cat results/02_generation_summary_{mode}.json"]
    elif "FILTER_DONE" in st:
        out.append(f"{n} · Filter + summary [{bar(0.7)}]  writing summary")
    else:
        out.append(f"{n} · Filter + summary [{bar(0)}]   0%   waiting")

    if "RUN_DONE" not in st:
        out += ["", f"elapsed {fmt(elapsed)} · last log line {fmt(now - info['last_time'])} ago:",
                f"  {info['last_line'][:110]}"]
        if (now - info["last_time"]) > timedelta(minutes=20) and "SAMPLE_START" not in " ".join(st):
            out.append("  ⚠ no new log line for 20+ minutes. Check the run: tail logs/02_full_run.log")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    ap.add_argument("--watch", action="store_true", help="refresh every --every seconds")
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()
    path = LOG_DIR / f"02_{args.mode}_progress.log"
    while True:
        text = render(parse(path), args.mode) if path.exists() else f"No progress log yet: {path}"
        if args.watch:
            print("\033[2J\033[H", end="")  # clear screen
        print(text)
        if not args.watch or "FINISHED" in text:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
