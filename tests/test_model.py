# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from homevalue.model import InvalidInputError, PriceModel


def test_a_price_is_exp_of_the_weighted_sum(model: PriceModel) -> None:
    assert model.predict("Whitefield", 1000, 2, 2) == 75.0  # 0.05 * 1000 * 1.5
    assert model.predict("Hebbal", 1000, 2, 2) == 50.0


def test_location_matches_whatever_the_case_and_spacing(model: PriceModel) -> None:
    assert model.predict("  whitefield ", 1000, 2, 2) == 75.0


def test_a_stored_name_with_a_double_space_still_matches(
    tmp_path: Path, small: dict[str, object]
) -> None:
    """The data has 'Sarjapur  Road' with two spaces; a caller types one."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps(small | {"locations": {"other": 0.0, "Sarjapur  Road": 0.0}}))
    assert PriceModel.from_json(path).predict("Sarjapur Road", 1000, 2, 2) == 50.0


def test_numbers_may_arrive_as_text_from_a_form(model: PriceModel) -> None:
    assert model.predict("Hebbal", "1000", "2", "2") == 50.0


def test_the_pooled_bucket_is_not_a_place(model: PriceModel) -> None:
    assert "other" not in model.location_names()
    with pytest.raises(InvalidInputError, match="not a listed location"):
        model.predict("other", 1000, 2, 2)


def test_an_unknown_location_is_refused_not_priced_as_none(model: PriceModel) -> None:
    """The 2024 version priced an unknown location as if it had none."""
    with pytest.raises(InvalidInputError, match="not a listed location") as refused:
        model.predict("Atlantis", 1000, 2, 2)
    assert refused.value.field == "location"


@pytest.mark.parametrize(
    ("args", "field", "reason"),
    [
        pytest.param(("Hebbal", True, 2, 2), "total_sqft", "must be a number", id="bool"),
        pytest.param(("Hebbal", float("nan"), 2, 2), "total_sqft", "finite", id="nan"),
        pytest.param(("Hebbal", float("inf"), 2, 2), "total_sqft", "finite", id="infinity"),
        pytest.param(("Hebbal", "=1+1", 2, 2), "total_sqft", "must be a number", id="formula"),
        pytest.param(("Hebbal", " ", 2, 2), "total_sqft", "must be a number", id="blank"),
        pytest.param(("Hebbal", None, 2, 2), "total_sqft", "must be a number", id="missing"),
        pytest.param(("Hebbal", [1000], 2, 2), "total_sqft", "must be a number", id="list"),
        pytest.param(("Hebbal", "-1000", 2, 2), "total_sqft", "at least 300", id="negative"),
        pytest.param(("Hebbal", 10**9, 2, 2), "total_sqft", "at most 30,000", id="huge"),
        pytest.param(("Hebbal", 1000, 2.5, 2), "bhk", "whole number from 1 to 10", id="half-bhk"),
        pytest.param(("Hebbal", 1000, 0, 1), "bhk", "whole number from 1 to 10", id="zero-bhk"),
        pytest.param(("Hebbal", 9000, 11, 2), "bhk", "whole number from 1 to 10", id="11-bhk"),
        pytest.param(("Hebbal", 1000, 2, 4), "bath", "whole number from 1 to 3", id="bath"),
        pytest.param(("", 1000, 2, 2), "location", "listed names", id="empty-location"),
        pytest.param((42, 1000, 2, 2), "location", "listed names", id="numeric-location"),
    ],
)
def test_hostile_input_is_refused_for_its_own_reason(
    model: PriceModel, args: tuple[object, ...], field: str, reason: str
) -> None:
    with pytest.raises(InvalidInputError, match=reason) as refused:
        model.predict(*args)
    assert refused.value.field == field


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("Hebbal", 600, 2, 2), id="smallest-2bhk-area"),
        pytest.param(("Hebbal", 30000, 10, 11), id="largest-area-most-rooms"),
        pytest.param(("Hebbal", "300", 1, 1), id="smallest-home"),
    ],
)
def test_the_edges_of_each_range_are_accepted(model: PriceModel, args: tuple[object, ...]) -> None:
    """The control for the refusals above: the same checks pass at their limits."""
    assert model.predict(*args) > 0


def test_a_model_file_without_its_accuracy_is_refused(
    tmp_path: Path, small: dict[str, object]
) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(small | {"scores": {}}))
    with pytest.raises(KeyError):
        PriceModel.from_json(path)
