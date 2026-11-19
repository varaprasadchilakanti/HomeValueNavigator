# SPDX-License-Identifier: MIT
"""A second scope is a folder, not a code change: train one from made-up listings and serve it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from homevalue import train
from homevalue.server import create_app

SCOPE = {
    "id": "testville",
    "name": "Testville",
    "region": "a made-up town for this test",
    "currency": "INR",
    "price_unit": "lakh",
    "area_unit": "sq ft",
    "data": {"file": "listings.csv", "kind": "made up", "source": "this test", "license": "MIT"},
}
#: Each locality's price per sq ft, in rupees.
RATES = {"North": 5000, "South": 7000, "East": 9000, "West": 11000}


def made_up_listings(rows: int = 800) -> pd.DataFrame:
    """Listings priced by rate times area, with noise, in the columns the real data has."""
    rng = np.random.default_rng(0)
    place = rng.choice(list(RATES), rows)
    bhk = rng.integers(1, 5, rows)
    sqft = bhk * rng.uniform(400, 900, rows)
    price = sqft * np.vectorize(RATES.get)(place) * rng.lognormal(0, 0.1, rows) / train.LAKH
    return pd.DataFrame(
        {
            "area_type": "Super built-up  Area",
            "availability": "Ready To Move",
            "location": place,
            "size": [f"{b} BHK" for b in bhk],
            "society": None,
            "total_sqft": sqft.round().astype(int).astype(str),
            "bath": bhk,
            "balcony": 1,
            "price": price.round(2),
        }
    )


def test_a_new_scope_trains_and_serves_with_no_code_change(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "testville"
    folder.mkdir(parents=True)
    (folder / "scope.json").write_text(json.dumps(SCOPE), encoding="utf-8")
    made_up_listings().to_csv(folder / "listings.csv", index=False)

    scope = train.read_scope("testville", root=tmp_path)
    trained = train.train(train.clean(pd.read_csv(train.listings(scope, root=tmp_path))))
    out = train.model_path(scope, root=tmp_path)
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(train.export(trained, scope, "")), encoding="utf-8")

    query = {"location": "west", "total_sqft": 1000, "bhk": 2, "bath": 2}
    body = create_app(out).test_client().post("/api/predict", json=query).get_json()
    assert body["location"] == "West"
    assert body["unit"] == "lakh INR"
    assert 90 < body["estimated_price"] < 130  # 1000 sq ft at Rs 11,000, about 110 lakh


def test_a_scope_whose_file_names_another_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "elsewhere"
    folder.mkdir(parents=True)
    (folder / "scope.json").write_text(json.dumps(SCOPE), encoding="utf-8")
    with pytest.raises(ValueError, match="names the scope 'testville'"):
        train.read_scope("elsewhere", root=tmp_path)
