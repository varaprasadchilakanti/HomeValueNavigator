from __future__ import annotations

import json
from pathlib import Path

import pytest

from homevalue.model import PriceModel

#: A small model with round weights, so expected prices can be worked out by hand.
SMALL = {
    "intercept": 10.0,
    "coefficients": {"total_sqft": 0.05, "bath": 2.0, "bhk": 3.0},
    "locations": {"other": 0.0, "Whitefield": 5.0, "Hebbal": -2.0},
    "scores": {"mae_held_out_lakh": 12.5},
}


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(SMALL), encoding="utf-8")
    return path


@pytest.fixture
def model(model_path: Path) -> PriceModel:
    return PriceModel.from_json(model_path)
