from __future__ import annotations

import pytest

from homevalue.model import InvalidInputError, PriceModel


def test_a_price_is_the_weighted_sum(model: PriceModel) -> None:
    # 10 + 0.05 * 1000 + 2 * 2 + 3 * 2 + 5
    assert model.predict("Whitefield", 1000, 2, 2) == 75.0


def test_location_matches_whatever_the_case_and_spacing(model: PriceModel) -> None:
    assert model.predict("  whitefield ", 1000, 2, 2) == model.predict("Whitefield", 1000, 2, 2)


def test_a_stored_name_with_a_double_space_still_matches() -> None:
    """The data has 'Sarjapur  Road' with two spaces; a caller types one."""
    model = PriceModel(10.0, 0.05, 2.0, 3.0, {"other": 0.0, "Sarjapur  Road": 1.0})
    assert model.predict("Sarjapur Road", 1000, 2, 2) == model.predict("Sarjapur  Road", 1000, 2, 2)


def test_numbers_may_arrive_as_text_from_a_form(model: PriceModel) -> None:
    assert model.predict("Hebbal", "1000", "2", "2") == 68.0


def test_an_unknown_location_is_refused_not_priced_as_none(model: PriceModel) -> None:
    """The 2024 version priced an unknown location as if it had none."""
    with pytest.raises(InvalidInputError, match="not a location the model was trained on"):
        model.predict("Atlantis", 1000, 2, 2)
    assert model.predict("Hebbal", 1000, 2, 2) > 0  # the positive control, same path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("sqft", True, id="bool-is-not-a-number"),
        pytest.param("sqft", float("nan"), id="nan"),
        pytest.param("sqft", float("inf"), id="infinity"),
        pytest.param("sqft", "-1000", id="leading-dash"),
        pytest.param("sqft", "=1+1", id="formula"),
        pytest.param("sqft", " ", id="blank"),
        pytest.param("sqft", "1000\n2000", id="newline"),
        pytest.param("sqft", None, id="missing"),
        pytest.param("sqft", [1000], id="list"),
        pytest.param("sqft", 10**9, id="huge"),
        pytest.param("bhk", 2.5, id="fractional-bhk"),
        pytest.param("bhk", 0, id="zero-bhk"),
        pytest.param("bhk", 11, id="too-many-bhk"),
        pytest.param("bath", 4, id="bath-over-bhk-plus-one"),
        pytest.param("location", "", id="empty-location"),
        pytest.param("location", 42, id="numeric-location"),
    ],
)
def test_hostile_input_is_refused_with_a_reason(
    model: PriceModel, field: str, value: object
) -> None:
    args: dict[str, object] = {"location": "Whitefield", "sqft": 1000, "bhk": 2, "bath": 2}
    args[field] = value
    with pytest.raises(InvalidInputError) as refused:
        model.predict(args["location"], args["sqft"], args["bhk"], args["bath"])
    assert str(refused.value)  # every refusal says why


def test_area_below_the_training_floor_is_refused(model: PriceModel) -> None:
    with pytest.raises(InvalidInputError, match="at least 300 per bedroom"):
        model.predict("Whitefield", 599, 2, 2)
    assert model.predict("Whitefield", 600, 2, 2) > 0


def test_a_non_positive_estimate_is_refused_not_shown() -> None:
    cheap = PriceModel(
        intercept=-500.0, sqft=0.01, bath=0.0, bhk=0.0, locations={"other": 0.0, "Hebbal": 0.0}
    )
    with pytest.raises(InvalidInputError, match="no meaningful price"):
        cheap.predict("Hebbal", 1000, 2, 2)


def test_locations_are_listed_sorted(model: PriceModel) -> None:
    assert model.location_names() == ["Hebbal", "Whitefield", "other"]
