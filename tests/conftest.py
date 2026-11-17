# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from homevalue.model import PriceModel

#: Weights chosen so prices work out by hand: price = 0.05 lakh per sq ft, times 1.5 in
#: Whitefield; bedrooms and bathrooms carry no weight.
SMALL: dict[str, object] = {
    "intercept": math.log(0.05),
    "coefficients": {"log_total_sqft": 1.0, "bath": 0.0, "bhk": 0.0},
    "locations": {"other": 0.0, "Whitefield": math.log(1.5), "Hebbal": 0.0},
    "bounds": {"min_sqft_per_bhk": 300, "max_sqft": 30000.0, "max_bhk": 10, "max_extra_bath": 1},
    "scores": {"held_out": {"median_abs_pct_error": 0.2}},
}


@pytest.fixture
def small() -> dict[str, object]:
    return copy.deepcopy(SMALL)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(SMALL), encoding="utf-8")
    return path


@pytest.fixture
def model(model_path: Path) -> PriceModel:
    return PriceModel.from_json(model_path)
