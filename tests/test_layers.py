# SPDX-License-Identifier: MIT
"""Serving needs only the weights: the training stack is never imported to answer a request."""

from __future__ import annotations

import subprocess
import sys

TRAINING_ONLY = ("numpy", "pandas", "scipy", "sklearn")


def test_serving_imports_none_of_the_training_stack() -> None:
    probe = (
        "import sys, homevalue.server; "
        f"print(sorted(m for m in {TRAINING_ONLY!r} if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert loaded == "[]"
