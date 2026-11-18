# SPDX-License-Identifier: MIT
"""The model rebuilds from the data, and the JSON predicts what scikit-learn does."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from homevalue import train
from homevalue.model import PriceModel

COMMITTED = train.OUT


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("1200", 1200.0, id="plain"),
        pytest.param("2100 - 2850", 2475.0, id="range-is-its-midpoint"),
        pytest.param("34.46Sq. Meter", None, id="a-unit-is-dropped"),
        pytest.param("1Grounds", None, id="another-unit"),
    ],
)
def test_area_text_becomes_a_number_or_nothing(raw: str, expected: float | None) -> None:
    assert train.sqft_to_number(raw) == expected


@pytest.fixture(scope="module")
def trained() -> train.Trained:
    return train.train(train.clean(pd.read_csv(train.DATA)))


def test_the_committed_model_is_what_the_data_and_code_produce(trained: train.Trained) -> None:
    rebuilt = flat(json.loads(json.dumps(train.export(trained, ""))))
    committed = flat(json.loads(COMMITTED.read_text(encoding="utf-8")))
    for key, value in rebuilt.items():
        if key.startswith(("data_sha256", "trained_with")):
            continue
        same = pytest.approx(value, rel=1e-9) if isinstance(value, float) else value
        assert committed[key] == same, key


def flat(tree: dict[str, object], prefix: str = "") -> dict[str, object]:
    """{"a": {"b": 1}} -> {"a.b": 1}, so nested numbers compare one by one."""
    out: dict[str, object] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out |= flat(value, f"{prefix}{key}.")
        else:
            out[f"{prefix}{key}"] = value
    return out


def test_the_json_model_predicts_what_scikit_learn_predicts(
    trained: train.Trained, tmp_path: Path
) -> None:
    out = tmp_path / "model.json"
    out.write_text(json.dumps(train.export(trained, "")), encoding="utf-8")
    model = PriceModel.from_json(out)
    sample = trained.rows[trained.rows["location"] != "other"].sample(25, random_state=1)
    locations = sorted(trained.rows["location"].unique())
    x, _ = train.features(sample, locations)
    expected = train.predict(trained.model, x)
    for (_, row), want in zip(sample.iterrows(), expected, strict=True):
        got = model.predict(row["location"], row["total_sqft"], row["bhk"], row["bath"])
        assert got == pytest.approx(want, abs=0.01)


def test_plot_area_listings_are_left_out() -> None:
    raw = pd.read_csv(train.DATA)
    assert (raw["area_type"] == train.PLOT_AREA).sum() > 0  # the rule has something to do
    kept = train.clean(raw)
    assert len(kept) < len(raw)
    assert set(raw.loc[kept.index, "area_type"]) == {
        "Super built-up  Area",
        "Built-up  Area",
        "Carpet  Area",
    }


def test_the_test_rows_are_not_filtered_by_price(trained: train.Trained) -> None:
    """The point of the split-first pipeline: every plausible test row is scored."""
    plausible = len(train.plausible(train.clean(pd.read_csv(train.DATA))))
    held_out = trained.scores["held_out"]["rows"]
    assert held_out + trained.scores["rows_train"] == plausible


def test_the_page_script_carries_the_same_weights_as_the_json() -> None:
    script = COMMITTED.with_suffix(".js").read_text(encoding="utf-8")
    prefix = f"{train.BROWSER_GLOBAL} = "
    assert script.startswith(prefix)
    assert script.endswith(";\n")
    assert json.loads(script[len(prefix) : -2]) == json.loads(COMMITTED.read_text(encoding="utf-8"))
