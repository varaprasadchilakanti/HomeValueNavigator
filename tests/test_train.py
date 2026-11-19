# SPDX-License-Identifier: MIT
"""The model rebuilds from the data, and the JSON predicts what scikit-learn does."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from homevalue import train
from homevalue.model import PriceModel

SCOPE = train.read_scope(train.DEFAULT_SCOPE)
DATA = train.listings(SCOPE)
COMMITTED = train.model_path(SCOPE)


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
    return train.train(train.clean(pd.read_csv(DATA)))


# How far a rebuild may land from the committed model. The fit's last digits depend on
# the order of floating-point sums, the BLAS build and, through uv.lock, the numpy and scipy
# versions each Python gets. Refitting with the columns permuted moves the weights by up to
# 3.3e-3 and the scores by up to 1.0e-3; the tolerances allow three and two times that. A
# weight is a log price, so 0.01 is 1%. A change smaller than that noise passes (a tenfold
# `alpha` moves the weights 3.2e-3); a Huber epsilon of 1.5, or another split, fails.
WEIGHTS = 1e-2
SCORES = 2e-3
# The figures the README and the page state, to the decimals they state them: these
# must come out the same on any machine, or the documents are wrong there.
PUBLISHED = {
    "scores.held_out.median_abs_pct_error": 3,
    "scores.held_out.median_abs_pct_error_95ci": 3,
    "scores.held_out.within_20_pct": 2,
    "scores.held_out.range_coverage": 2,
    "scores.cross_validated.median_abs_pct_error": 3,
    "scores.course_method.median_abs_pct_error": 3,
    "scores.course_method.within_20_pct": 2,
    "scores.course_method.lead_95ci": 3,
    "scores.baseline.median_abs_pct_error": 3,
    "scores.baseline.within_20_pct": 2,
    "scores.baseline.lead_95ci": 3,
}


@pytest.mark.rebuild
def test_the_committed_model_is_what_the_data_and_code_produce(trained: train.Trained) -> None:
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    rebuilt = flat(json.loads(json.dumps(train.export(trained, SCOPE, digest))))
    committed = flat(json.loads(COMMITTED.read_text(encoding="utf-8")))
    assert committed.keys() == rebuilt.keys()
    for key, value in rebuilt.items():
        if key.startswith("trained_with"):
            continue  # versions may differ between machines; the numbers may not
        assert committed[key] == within(key, value), key
    for key, decimals in PUBLISHED.items():
        assert rounded(rebuilt[key], decimals) == rounded(committed[key], decimals), key


def within(key: str, value: object) -> object:
    """`value` as the committed one must equal it: fitted numbers within their noise."""
    if key == "intercept" or key.startswith(("coefficients.", "locations.")):
        return pytest.approx(value, abs=WEIGHTS)
    if key.startswith("scores.") and isinstance(value, float | list):
        return pytest.approx(value, abs=SCORES)
    return pytest.approx(value, rel=1e-9) if isinstance(value, float) else value


def rounded(value: object, decimals: int) -> object:
    if isinstance(value, list):
        return [round(item, decimals) for item in value]
    assert isinstance(value, float)
    return round(value, decimals)


def flat(tree: dict[str, object], prefix: str = "") -> dict[str, object]:
    """{"a": {"b": 1}} -> {"a.b": 1}, so nested numbers compare one by one."""
    out: dict[str, object] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out |= flat(value, f"{prefix}{key}.")
        else:
            out[f"{prefix}{key}"] = value
    return out


@pytest.mark.rebuild
def test_the_json_model_predicts_what_scikit_learn_predicts(
    trained: train.Trained, tmp_path: Path
) -> None:
    out = tmp_path / "model.json"
    out.write_text(json.dumps(train.export(trained, SCOPE, "")), encoding="utf-8")
    model = PriceModel.from_json(out)
    sample = trained.rows[trained.rows["location"] != "other"].sample(25, random_state=1)
    locations = sorted(trained.rows["location"].unique())
    x, _ = train.features(sample, locations)
    expected = train.predict(trained.model, x)
    for (_, row), want in zip(sample.iterrows(), expected, strict=True):
        got = model.predict(row["location"], row["total_sqft"], row["bhk"], row["bath"])
        assert got == pytest.approx(want, abs=0.01)


def test_plot_area_listings_are_left_out() -> None:
    raw = pd.read_csv(DATA)
    assert (raw["area_type"] == train.PLOT_AREA).sum() > 0  # the rule has something to do
    kept = train.clean(raw)
    assert len(kept) < len(raw)
    assert set(raw.loc[kept.index, "area_type"]) == {
        "Super built-up  Area",
        "Built-up  Area",
        "Carpet  Area",
    }


@pytest.mark.rebuild
def test_the_test_rows_are_not_filtered_by_price(trained: train.Trained) -> None:
    """The point of the split-first pipeline: every plausible test row is scored."""
    plausible = len(train.plausible(train.clean(pd.read_csv(DATA))))
    held_out = trained.scores["held_out"]["rows"]
    assert held_out + trained.scores["rows_train"] == plausible


def test_the_page_script_carries_the_same_weights_as_the_json() -> None:
    script = COMMITTED.with_suffix(".js").read_text(encoding="utf-8")
    prefix = f"{train.BROWSER_GLOBAL} = "
    assert script.startswith(prefix)
    assert script.endswith(";\n")
    assert json.loads(script[len(prefix) : -2]) == json.loads(COMMITTED.read_text(encoding="utf-8"))


@pytest.mark.rebuild
def test_the_held_out_figure_lies_inside_its_interval(trained: train.Trained) -> None:
    held_out = trained.scores["held_out"]
    low, high = held_out["median_abs_pct_error_95ci"]
    assert low < held_out["median_abs_pct_error"] < high


def test_the_scope_index_lists_each_trained_model_by_name(tmp_path: Path) -> None:
    for scope_id, name in [("pune", "Pune"), ("bengaluru", "Bengaluru")]:
        (tmp_path / scope_id).mkdir()
        body = {
            "scope": {"id": scope_id, "name": name, "currency": "INR"},
            "locations": {"other": 0.0, "Zeta": 0.1, "alpha": 0.2},
        }
        (tmp_path / scope_id / "model.json").write_text(json.dumps(body), encoding="utf-8")
    names = ["alpha", "Zeta"]  # case-blind order; the pooled "other" is not a place to pick
    assert train.scope_index(tmp_path) == [
        {"id": "bengaluru", "name": "Bengaluru", "locations": names},
        {"id": "pune", "name": "Pune", "locations": names},
    ]


def test_the_committed_scope_index_matches_the_models() -> None:
    models = train.ROOT / "model"
    text = (models / "scopes.js").read_text(encoding="utf-8")
    listed = json.loads(text.removeprefix(f"{train.SCOPES_GLOBAL} = ").removesuffix(";\n"))
    assert listed == train.scope_index(models)
