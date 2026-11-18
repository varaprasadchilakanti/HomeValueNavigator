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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The pooled bucket for rare locations: the baseline of the fit, not a place to ask about.
OTHER = "other"
#: A number sent as text: ASCII digits and one optional decimal part, once spaces at the
#: ends and thousands commas are removed. The page applies the same rule (app.js, parseArea).
_DIGITS = re.compile(r"[0-9]+(\.[0-9]+)?")


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
    """The ranges the training data covered, as `train.py` wrote them."""

    min_sqft_per_bhk: float
    #: The largest area listed for each bedroom count the data has; a count it lacks is refused.
    max_sqft_by_bhk: dict[int, float]
    max_extra_bath: int


@dataclass(frozen=True, slots=True)
class Estimate:
    """A price with its typical range, in the model's unit, and the location it was priced for."""

    location: str
    price: float
    low: float
    high: float


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
    #: were within this fraction of the asking price.
    median_error: float
    #: The price unit, such as "lakh INR", from the scope the model was trained for.
    unit: str
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
            bounds=Bounds(
                min_sqft_per_bhk=float(bounds["min_sqft_per_bhk"]),
                max_sqft_by_bhk={int(k): float(v) for k, v in bounds["max_sqft_by_bhk"].items()},
                max_extra_bath=int(bounds["max_extra_bath"]),
            ),
            median_error=float(data["scores"]["held_out"]["median_abs_pct_error"]),
            unit=str(data["unit"]),
        )

    def location_names(self) -> list[str]:
        """The locations a price can be asked for, sorted."""
        return sorted(self._index.values())

    def estimate(self, location: object, sqft: object, bhk: object, bath: object) -> Estimate:
        """The price and its typical range: about half of held-out prices fell within it."""
        place = self.location_name(location)
        price = self.predict(place, sqft, bhk, bath)
        spread = price * self.median_error
        return Estimate(place, price, round(price - spread, 2), round(price + spread, 2))

    def predict(self, location: object, sqft: object, bhk: object, bath: object) -> float:
        """Price in the model's unit, rounded to two places, or InvalidInputError saying why not."""
        place = self.location_name(location)
        rooms = self._bedrooms(bhk)
        baths = _whole("bath", bath, 1, rooms + self.bounds.max_extra_bath)
        area = self._area(_number("total_sqft", sqft), rooms)
        log_price = (
            self.intercept
            + self.log_sqft * math.log(area)
            + self.bath * baths
            + self.bhk * rooms
            + self.locations[place]
        )
        return round(math.exp(log_price), 2)

    def location_name(self, value: object) -> str:
        """The listed name `value` matches, ignoring case and spacing, or InvalidInputError."""
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError("location must be one of the listed names.", "location")
        name = self._index.get(_key(value))
        if name is None:
            raise InvalidInputError(f"{value.strip()!r} is not a listed location.", "location")
        return name

    def _bedrooms(self, value: object) -> int:
        rooms = _whole("bhk", value, 1, max(self.bounds.max_sqft_by_bhk))
        if rooms not in self.bounds.max_sqft_by_bhk:
            raise InvalidInputError(f"No listing in the data has {rooms} bedrooms.", "bhk")
        return rooms

    def _area(self, sqft: float, bhk: int) -> float:
        most = self.bounds.max_sqft_by_bhk[bhk]
        if sqft > most:
            raise InvalidInputError(
                f"total_sqft must be at most {most:,.0f} for {bhk} BHK, "
                "the largest such listing in the data.",
                "total_sqft",
            )
        least = self.bounds.min_sqft_per_bhk
        if sqft < least * bhk:
            raise InvalidInputError(
                f"total_sqft must be at least {least:,.0f} per bedroom "
                f"({least * bhk:,.0f} for {bhk} BHK).",
                "total_sqft",
            )
        return sqft


def _key(name: str) -> str:
    return " ".join(name.split()).lower()


def _number(name: str, value: object) -> float:
    """A JSON number, or text in the form `_DIGITS` allows; anything else is refused."""
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not _DIGITS.fullmatch(text):
            raise InvalidInputError(f"{name} must be a number, such as 1200.", name)
        value = text
    elif isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInputError(f"{name} must be a number, such as 1200.", name)
    try:
        number = float(value)
    except OverflowError:
        raise InvalidInputError(f"{name} is too large.", name) from None
    if not math.isfinite(number):
        raise InvalidInputError(f"{name} must be a finite number.", name)
    return number


def _whole(name: str, value: object, low: int, high: int) -> int:
    number = _number(name, value)
    if not number.is_integer() or not low <= number <= high:
        raise InvalidInputError(f"{name} must be a whole number from {low} to {high}.", name)
    return int(number)
