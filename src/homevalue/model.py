# SPDX-License-Identifier: MIT
"""The price model at run time: weights read from JSON, with no scikit-learn and no pickle.

The model is linear in log price, so a prediction is exp of a weighted sum, and serving
it needs only the weights.
Loading a pickle runs code and ties the server to one scikit-learn version; reading JSON
does neither. `train.py` writes the file, including the bounds of the data it learned
from; this module reads it and refuses what falls outside them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The pooled bucket for rare locations: the baseline of the fit, not a place to ask about.
OTHER = "other"
#: Below this area per bedroom, the course treated a listing as a data error.
MIN_SQFT_PER_BHK = 300
#: Listings with more bathrooms than bedrooms plus this were removed in training.
MAX_EXTRA_BATH = 1


class InvalidInputError(ValueError):
    """A request the model cannot answer, with the reason in words.

    `field` names the input at fault, or is None when no single input is.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        """Keep the message and the field it concerns."""
        super().__init__(message)
        self.field = field


@dataclass(frozen=True, slots=True)
class Bounds:
    """The ranges the training data covered."""

    max_sqft: float
    max_bhk: int


@dataclass(frozen=True, slots=True)
class PriceModel:
    """log(price) as a linear function of log area, bathrooms, bedrooms and location."""

    intercept: float
    log_sqft: float
    bath: float
    bhk: float
    locations: dict[str, float]
    bounds: Bounds
    #: Median absolute percentage error on held-out listings: half of the estimates
    #: were within this fraction of the listed price.
    median_error: float
    _index: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index the names once, compared without case or repeated spaces."""
        names = {_key(n): n for n in self.locations if n != OTHER}
        object.__setattr__(self, "_index", names)

    @classmethod
    def from_json(cls, path: Path) -> PriceModel:
        """Read what `train.py` wrote; a missing key is an error, not a default."""
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        weights, bounds = data["coefficients"], data["bounds"]
        return cls(
            intercept=float(data["intercept"]),
            log_sqft=float(weights["log_total_sqft"]),
            bath=float(weights["bath"]),
            bhk=float(weights["bhk"]),
            locations={k: float(v) for k, v in data["locations"].items()},
            bounds=Bounds(float(bounds["max_sqft"]), int(bounds["max_bhk"])),
            median_error=float(data["scores"]["held_out"]["median_abs_pct_error"]),
        )

    def location_names(self) -> list[str]:
        """The locations a price can be asked for, sorted."""
        return sorted(self._index.values())

    def predict(self, location: object, sqft: object, bhk: object, bath: object) -> float:
        """Price in lakh rupees, rounded to two places, or InvalidInputError saying why not."""
        place = self._location(location)
        rooms = _whole("bhk", bhk, 1, self.bounds.max_bhk)
        baths = _whole("bath", bath, 1, rooms + MAX_EXTRA_BATH)
        area = self._area(_number("total_sqft", sqft), rooms)
        log_price = (
            self.intercept
            + self.log_sqft * math.log(area)
            + self.bath * baths
            + self.bhk * rooms
            + self.locations[place]
        )
        return round(math.exp(log_price), 2)

    def _location(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError("location must be one of the listed names.", "location")
        name = self._index.get(_key(value))
        if name is None:
            raise InvalidInputError(f"{value.strip()!r} is not a listed location.", "location")
        return name

    def _area(self, sqft: float, bhk: int) -> float:
        if sqft > self.bounds.max_sqft:
            raise InvalidInputError(
                f"total_sqft must be at most {self.bounds.max_sqft:,.0f}.", "total_sqft"
            )
        if sqft < MIN_SQFT_PER_BHK * bhk:
            raise InvalidInputError(
                f"total_sqft must be at least {MIN_SQFT_PER_BHK} per bedroom "
                f"({MIN_SQFT_PER_BHK * bhk} for {bhk} BHK).",
                "total_sqft",
            )
        return sqft


def _key(name: str) -> str:
    return " ".join(name.split()).lower()


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise InvalidInputError(f"{name} must be a number.", name)
    try:
        number = float(value)
    except ValueError:
        raise InvalidInputError(f"{name} must be a number.", name) from None
    if not math.isfinite(number):
        raise InvalidInputError(f"{name} must be a finite number.", name)
    return number


def _whole(name: str, value: object, low: int, high: int) -> int:
    number = _number(name, value)
    if not number.is_integer() or not low <= number <= high:
        raise InvalidInputError(f"{name} must be a whole number from {low} to {high}.", name)
    return int(number)
