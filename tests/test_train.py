"""The model is reproducible from the data, and the JSON predicts what scikit-learn does."""

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
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    out = tmp_path_factory.mktemp("model") / "model.json"
    assert train.main(["--out", str(out)]) == 0
    loaded: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
    return loaded


def test_the_committed_model_is_what_the_data_and_code_produce(trained: dict[str, object]) -> None:
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    for key in ("intercept", "coefficients", "locations", "scores", "data_sha256"):
        assert committed[key] == pytest.approx(trained[key]), key


def test_the_json_model_predicts_what_scikit_learn_predicts(tmp_path: Path) -> None:
    df = train.remove_outliers(train.clean(pd.read_csv(train.DATA)))
    x, y = train.features(df)
    fitted, _ = train.fit(x, y)
    out = tmp_path / "model.json"
    out.write_text(json.dumps(train.export(fitted, x, {}, "")), encoding="utf-8")
    model = PriceModel.from_json(out)
    for _, row in df.sample(25, random_state=1).iterrows():
        expected = fitted.predict(x.loc[[row.name]])[0]
        got = model.predict(row["location"], row["total_sqft"], row["bhk"], row["bath"])
        assert got == pytest.approx(round(expected, 2), abs=0.01)
