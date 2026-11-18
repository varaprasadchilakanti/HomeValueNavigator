# SPDX-License-Identifier: MIT
"""The page predicts in the browser, so its code must agree with the Python model.

The page's own JavaScript runs under Node through tests/page_parity.js. Without Node the
test is skipped on a workstation and fails in CI, where it must run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from homevalue import train
from homevalue.model import InvalidInputError, PriceModel

HARNESS = Path(__file__).with_name("page_parity.js")
MODEL = train.model_path(train.read_scope(train.DEFAULT_SCOPE))
ESTIMATES = [
    ("Whitefield", 1200, 2, 2),
    ("Whitefield", 600, 2, 3),
    ("Electronic City Phase II", 1000, 2, 2),
    ("Indira Nagar", 3000, 4, 5),
    ("Hebbal", 450, 1, 1),
    ("Sarjapur Road", 1800, 3, 3),
]
TWO = [{"id": "bengaluru", "name": "Bengaluru"}, {"id": "pune", "name": "Pune"}]
#: (the scope list, the address's query, the city the page should open on)
CITIES = [
    (TWO, "", "bengaluru"),
    (TWO, "?city=pune", "pune"),
    (TWO, "?city=delhi", "bengaluru"),
    ([{"id": "../x", "name": "X"}, *TWO], "?city=../x", "bengaluru"),
    ([{"id": "pune"}], "", None),
    ("not a list", "", None),
]
AREAS = ["1200", " 1,200 ", "1200.5", "1e3", "1_200", "12 00", "-5", "1200.", "\u0661\u0662"]


def page_answers() -> dict[str, list[object]]:
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("Node is needed in CI to test the page.")
        pytest.skip("Node is not installed.")
    cities = [[raw, search] for raw, search, _ in CITIES]
    cases = json.dumps({"estimates": ESTIMATES, "areas": AREAS, "cities": cities})
    run = subprocess.run(
        [node, str(HARNESS)], input=cases, check=True, capture_output=True, text=True
    )
    answers: dict[str, list[object]] = json.loads(run.stdout)
    return answers


def test_the_page_prices_as_the_python_model_does() -> None:
    model = PriceModel.from_json(MODEL)
    for case, price in zip(ESTIMATES, page_answers()["estimates"], strict=True):
        assert model.predict(*case) == pytest.approx(price, abs=0.005), case


def test_the_page_reads_an_area_as_the_python_model_does() -> None:
    model = PriceModel.from_json(MODEL)
    for text, read in zip(AREAS, page_answers()["areas"], strict=True):
        try:
            model.predict("Whitefield", text, 1, 1)
            accepted = True
        except InvalidInputError as refused:
            accepted = refused.field != "total_sqft"
        assert accepted == (read is not None), text


def test_the_page_opens_only_a_listed_city_with_a_safe_id() -> None:
    expected = [city for _, _, city in CITIES]
    assert page_answers()["cities"] == expected
