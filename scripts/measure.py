# SPDX-License-Identifier: MIT
"""Measure what the project costs to run, so the README states numbers.

    uv run python scripts/measure.py

Prints the page weight a visitor downloads, the time for one prediction in Python and
through the HTTP API, and the time and peak memory to train. Peak memory is read from
the operating system on Linux and macOS, and reported as unavailable elsewhere.
"""

from __future__ import annotations

import gzip
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from homevalue.model import PriceModel
from homevalue.server import create_app

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "bengaluru" / "model.json"
CLIENT = ("app.html", "app.css", "theme.js", "app.js", "favicon.svg")
PAGE = [ROOT / "client" / n for n in CLIENT] + [MODEL.with_suffix(".js")]
# The root index.html only redirects, so it is not counted.
QUERY = {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}


def median_microseconds(call: Callable[[], object], runs: int) -> float:
    """Median wall time of `call()` over `runs`, in microseconds."""
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        call()
        times.append(time.perf_counter_ns() - start)
    return statistics.median(times) / 1000


def training() -> tuple[float, str]:
    """Seconds for one training run in a child process, and its peak memory if known."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "model.json"
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, "-m", "homevalue.train", "--out", str(out)],
            check=True,
            capture_output=True,
        )
        seconds = time.perf_counter() - start
    return seconds, _peak_child_memory()


def _peak_child_memory() -> str:
    if sys.platform == "win32":
        return "peak memory not measured on Windows"
    import resource

    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    mib = peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux
    return f"peak {mib:.0f} MiB"


def main() -> int:
    """Print each measurement on its own line."""
    raw = sum(p.stat().st_size for p in PAGE)
    zipped = sum(len(gzip.compress(p.read_bytes())) for p in PAGE)
    model = PriceModel.from_json(MODEL)
    client = create_app(MODEL).test_client()
    predict_us = median_microseconds(lambda: model.predict("Whitefield", 1200, 2, 2), 20_000)
    api_us = median_microseconds(lambda: client.post("/api/predict", json=QUERY), 2_000)
    seconds, memory = training()
    cores = os.cpu_count() or 1
    print(f"machine        {platform.machine()}, {cores} cores, Python {platform.python_version()}")
    print(
        f"page weight    {raw / 1024:.1f} KiB ({zipped / 1024:.1f} KiB gzipped), {len(PAGE)} files"
    )
    print(f"predict        {predict_us:.1f} us median (Python, in process)")
    print(f"API request    {api_us:.0f} us median (Flask test client, no network)")
    print(f"train          {seconds:.1f} s, {memory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
