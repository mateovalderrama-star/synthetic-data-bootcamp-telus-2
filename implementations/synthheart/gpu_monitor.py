"""Record GPU memory, utilisation and power while a block of code runs, so TabDDPM and NeMo Safe Synthesizer
are timed by the same instrument.

    with GpuMonitor(out_json) as gpu:
        train()
        gpu.mark("train")      # optional: names the stage that just ended
        generate()
        gpu.mark("generate")

It polls `nvidia-smi` for the whole device, so it also sees child processes (the notebook kernel, vLLM).
If someone else is using the same GPU, their memory is counted too: `baseline_mem_gb` shows what was
already in use when the run started.

    python gpu_monitor.py     # self-check of the arithmetic (no GPU needed)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


FIELDS = "memory.used,utilization.gpu,power.draw"


def summarize(samples: list[list[float | None]], every: float, marks: dict[str, float]) -> dict:
    """Turn the samples ([memory MiB, utilisation %, power W] every `every` seconds) into the numbers we report."""
    mem = [s[0] for s in samples if s[0] is not None]
    util = [s[1] for s in samples if s[1] is not None]
    power = [s[2] for s in samples if s[2] is not None]
    out: dict = {"samples": len(samples), "sample_seconds": every}
    if mem:
        out["baseline_mem_gb"] = round(mem[0] / 1024, 2)
        out["peak_mem_gb"] = round(max(mem) / 1024, 2)
    if util:
        out["mean_util_pct"] = round(sum(util) / len(util), 1)
    if power:
        out["energy_wh"] = round(sum(power) * every / 3600, 1)
        out["mean_power_w"] = round(sum(power) / len(power), 1)
    start, stages = 0.0, {}
    for name, end in marks.items():  # marks are stage END times, in order
        part = [s[0] for s in samples[int(start / every) : int(end / every) + 1] if s[0] is not None]
        stages[name] = {"minutes": round((end - start) / 60, 2), "peak_mem_gb": round(max(part) / 1024, 2) if part else None}
        start = end
    if stages:
        out["stages"] = stages
    step = max(1, len(samples) // 300)  # keep the committed JSON small
    out["trace_every_seconds"] = every * step
    out["trace_mem_gb_util_pct"] = [[round(s[0] / 1024, 1) if s[0] is not None else None, s[1]] for s in samples[::step]]
    return out


class GpuMonitor:
    def __init__(self, out: Path, every: int = 2):
        self.out, self.every, self.marks = Path(out), every, {}
        self.gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"

    def __enter__(self) -> "GpuMonitor":
        self.t0 = time.time()
        self.trace = Path(tempfile.mkstemp(suffix=".csv")[1])
        cmd = ["nvidia-smi", f"--id={self.gpu}", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
               "-l", str(self.every), "-f", str(self.trace)]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:  # no nvidia-smi: still record wall-clock time
            self.proc = None
        return self

    def mark(self, stage: str) -> None:
        self.marks[stage] = time.time() - self.t0

    def __exit__(self, exc_type, exc, tb) -> None:
        seconds = time.time() - self.t0
        samples = []
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            for line in self.trace.read_text().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    samples.append([float(p) if p.replace(".", "", 1).isdigit() else None for p in parts])
        self.trace.unlink(missing_ok=True)
        result = {"total_minutes": round(seconds / 60, 2), "gpu_index": self.gpu, "failed": exc_type is not None,
                  **summarize(samples, self.every, self.marks)}
        try:
            result["gpu_name"] = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            result["gpu_name"] = None
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    # 60 s at 1 GiB / 50% / 360 W, then 60 s at 2 GiB / 100% / 360 W, sampled every 2 s.
    fake = [[1024.0, 50.0, 360.0]] * 30 + [[2048.0, 100.0, 360.0]] * 30
    got = summarize(fake, 2, {"train": 60.0, "generate": 120.0})
    assert got["peak_mem_gb"] == 2.0 and got["baseline_mem_gb"] == 1.0, got
    assert got["mean_util_pct"] == 75.0 and got["energy_wh"] == 12.0, got  # 360 W for 2 min = 12 Wh
    assert got["stages"]["train"] == {"minutes": 1.0, "peak_mem_gb": 2.0}, got  # boundary sample belongs to both
    assert got["stages"]["generate"]["minutes"] == 1.0, got
    assert summarize([[None, None, None]], 2, {}) == {
        "samples": 1, "sample_seconds": 2, "trace_every_seconds": 2, "trace_mem_gb_util_pct": [[None, None]]}
    print("gpu_monitor self-check passed")
