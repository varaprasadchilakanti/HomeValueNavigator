"""Measure what the project costs to run, so the README states numbers, not adjectives.

    uv run python scripts/measure.py

Prints the page weight a visitor downloads, the model file's size, the time for one
prediction in Python and through the HTTP API, and the time and peak memory to train.
"""

from __future__ import annotations

import gzip
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from homevalue.model import PriceModel
from homevalue.server import create_app

ROOT = Path(__file__).resolve().parents[1]
PAGE = [ROOT / "client" / n for n in ("app.html", "app.css", "app.js")] + [
    ROOT / "model" / "model.json"
]
QUERY = {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}


def median_microseconds(call: Callable[[], object], runs: int) -> float:
    """Median wall time of `call()` over `runs`, in microseconds."""
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        call()
        times.append(time.perf_counter_ns() - start)
    return statistics.median(times) / 1000


def training() -> tuple[float, float]:
    """Seconds and peak megabytes for one training run, in a child process."""
    start = time.perf_counter()
    out = ROOT / ".measure-model.json"
    subprocess.run(
        [sys.executable, "-m", "homevalue.train", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    seconds = time.perf_counter() - start
    out.unlink()
    peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return seconds, peak_kib / 1024


def main() -> int:
    """Print each measurement on its own line."""
    raw = sum(p.stat().st_size for p in PAGE)
    zipped = sum(len(gzip.compress(p.read_bytes())) for p in PAGE)
    model = PriceModel.from_json(ROOT / "model" / "model.json")
    client = create_app(ROOT / "model" / "model.json", allowed_origin="").test_client()
    predict_us = median_microseconds(lambda: model.predict("Whitefield", 1200, 2, 2), 20_000)
    api_us = median_microseconds(lambda: client.post("/api/predict", json=QUERY), 2_000)
    seconds, peak_mb = training()
    cores = os.cpu_count() or 1
    print(f"machine        {platform.machine()}, {cores} cores, Python {platform.python_version()}")
    print(f"page weight    {raw / 1024:.1f} KiB ({zipped / 1024:.1f} KiB gzipped), 4 files")
    print(f"model file     {(ROOT / 'model' / 'model.json').stat().st_size / 1024:.1f} KiB")
    print(f"predict        {predict_us:.1f} us median (Python, in process)")
    print(f"API request    {api_us:.0f} us median (Flask test client, no network)")
    print(f"train          {seconds:.1f} s, peak {peak_mb:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
