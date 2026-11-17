"""The price model at run time: coefficients read from JSON, no scikit-learn, no pickle.

A linear regression predicts with a sum of weights, so serving it needs nothing but the
weights. Loading a pickle runs code and ties the server to one scikit-learn version;
reading JSON does neither. `train.py` writes the file; this module reads it.

Inputs are checked before they reach the sum. A location the model was not trained on is
refused rather than priced as though it had no location, which is what the 2024 version did.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The smallest area per bedroom kept in training; below it the data was treated as an error.
MIN_SQFT_PER_BHK = 300
MAX_SQFT = 30_000
MAX_BHK = 10


class InvalidInputError(ValueError):
    """A request the model cannot answer, with the reason in words."""


@dataclass(frozen=True, slots=True)
class PriceModel:
    """A fitted linear model over area, bathrooms, bedrooms and one-hot location."""

    intercept: float
    sqft: float
    bath: float
    bhk: float
    locations: dict[str, float]
    #: Mean absolute error on the held-out listings, in lakh: how far off an estimate
    #: typically is. Shown beside every price so a number is never read as exact.
    typical_error: float = 0.0
    _index: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index the names once, compared without case or repeated spaces."""
        object.__setattr__(self, "_index", {_key(n): n for n in self.locations})

    @classmethod
    def from_json(cls, path: Path) -> PriceModel:
        """Read the coefficients `train.py` wrote."""
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        weights = data["coefficients"]
        return cls(
            intercept=float(data["intercept"]),
            sqft=float(weights["total_sqft"]),
            bath=float(weights["bath"]),
            bhk=float(weights["bhk"]),
            locations={k: float(v) for k, v in data["locations"].items()},
            typical_error=float(data.get("scores", {}).get("mae_held_out_lakh", 0.0)),
        )

    def location_names(self) -> list[str]:
        """The locations a price can be asked for, sorted."""
        return sorted(self.locations)

    def predict(self, location: object, sqft: object, bhk: object, bath: object) -> float:
        """Price in lakh rupees, rounded to two places, or InvalidInputError saying why not."""
        place = self._location(location)
        area = _number("total_sqft", sqft)
        rooms = _whole("bhk", bhk, 1, MAX_BHK)
        baths = _whole("bath", bath, 1, rooms + 1)
        _check_area(area, rooms)
        price = (
            self.intercept
            + self.sqft * area
            + self.bath * baths
            + self.bhk * rooms
            + self.locations[place]
        )
        if price <= 0:
            raise InvalidInputError(
                "The model gives no meaningful price for these inputs: its estimate is not "
                "positive. They are outside the range it was trained on."
            )
        return round(price, 2)

    def _location(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError("location must be one of the listed location names.")
        name = self._index.get(_key(value))
        if name is not None:
            return name
        raise InvalidInputError(f"{value.strip()!r} is not a location the model was trained on.")


def _key(name: str) -> str:
    return " ".join(name.split()).lower()


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise InvalidInputError(f"{name} must be a number.")
    try:
        number = float(value)
    except ValueError:
        raise InvalidInputError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise InvalidInputError(f"{name} must be a finite number.")
    return number


def _whole(name: str, value: object, low: int, high: int) -> int:
    number = _number(name, value)
    if not number.is_integer() or not low <= number <= high:
        raise InvalidInputError(f"{name} must be a whole number from {low} to {high}.")
    return int(number)


def _check_area(sqft: float, bhk: int) -> None:
    if sqft > MAX_SQFT:
        raise InvalidInputError(f"total_sqft must be at most {MAX_SQFT:,}.")
    if sqft < MIN_SQFT_PER_BHK * bhk:
        raise InvalidInputError(
            f"total_sqft must be at least {MIN_SQFT_PER_BHK} per bedroom "
            f"({MIN_SQFT_PER_BHK * bhk} for {bhk} BHK); smaller listings were removed "
            "from the training data as errors."
        )
