# SPDX-License-Identifier: MIT
"""Train the model from the CSV and write its weights.

    uv run python -m homevalue.train     # writes model/model.json and model/model.js

The data and its cleaning come from the codebasics course. The model differs in four ways,
each measured (README, "Model card"):
- It leaves out plot-area listings, whose area is land rather than floor space.
- It predicts log price from log area, bathrooms, bedrooms and location, so its errors
  scale with price and it never gives a negative estimate.
- It fits with a Huber loss, so a few mistyped listings pull the weights less than they
  would under least squares. The model stays linear: the page needs only the weights.
- The course removes price outliers from all the data before it splits off a test set, so
  its score is measured on listings already filtered by their own price. Here the test set
  gets only the rules a visitor's input also passes. The course method and a naive
  baseline are scored the same way, beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import sklearn
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from homevalue.model import OTHER

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "bengaluru_house_prices.csv"
OUT = ROOT / "model" / "model.json"
BROWSER_GLOBAL = "window.HOMEVALUE_MODEL"
#: The 2024 notebook's split and seed.
TEST_SIZE = 0.2
SPLIT_SEED = 10
FOLDS = 5
FOLD_SEED = 0
LAKH = 100_000  # rupees
#: Below this area per bedroom, the course treated a listing as a data error.
MIN_SQFT_PER_BHK = 300
#: Listings with more bathrooms than bedrooms plus this are removed.
MAX_EXTRA_BATH = 1
PLOT_AREA = "Plot  Area"  # spelled with two spaces in the data
RARE_LOCATION = 10  # a location with this many listings or fewer becomes OTHER
MIN_SMALLER_LISTINGS = 5  # the BHK rule applies only where the smaller size has more than this


def sqft_to_number(value: str) -> float | None:
    """'1200' -> 1200.0, '2100 - 2850' -> the midpoint, anything with a unit -> None."""
    parts = value.split("-")
    try:
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return float(value)
    except ValueError:
        return None


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Homes measured by floor area, then the course's cleaning in its order.

    Plot-area listings measure land, not floor space, so the same number means a
    different thing; the course kept them, this model does not.
    """
    raw = raw[raw["area_type"] != PLOT_AREA]
    df = raw.drop(columns=["area_type", "availability", "society", "balcony"]).dropna()
    df = df.assign(bhk=df["size"].map(lambda s: int(str(s).split(" ")[0])))
    df = df.assign(total_sqft=df["total_sqft"].map(lambda s: sqft_to_number(str(s)))).dropna()
    df = df.assign(location=df["location"].str.strip())
    df = df.assign(price_per_sqft=df["price"] * LAKH / df["total_sqft"])
    counts = df["location"].value_counts()
    rare = set(counts[counts <= RARE_LOCATION].index)
    return df.assign(location=df["location"].where(~df["location"].isin(rare), OTHER))


def plausible(df: pd.DataFrame) -> pd.DataFrame:
    """The rules that use no price, so a visitor's input passes them too."""
    keep = (df["total_sqft"] / df["bhk"] >= MIN_SQFT_PER_BHK) & (
        df["bath"] <= df["bhk"] + MAX_EXTRA_BATH
    )
    return df[keep].reset_index(drop=True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """The course's outlier rules in its order: area per bedroom, price, BHK, bathrooms."""
    df = df[df["total_sqft"] / df["bhk"] >= MIN_SQFT_PER_BHK]
    df = _within_one_std(df)
    df = _drop_cheaper_than_smaller(df)
    return df[df["bath"] <= df["bhk"] + MAX_EXTRA_BATH]


def _within_one_std(df: pd.DataFrame) -> pd.DataFrame:
    """Per location, keep price per sqft within one population std of its mean."""
    grouped = df.groupby("location")["price_per_sqft"]
    stats = pd.DataFrame({"mean": grouped.mean(), "std": grouped.std(ddof=0)})
    joined = df.join(stats, on="location")
    keep = (joined["price_per_sqft"] > joined["mean"] - joined["std"]) & (
        joined["price_per_sqft"] <= joined["mean"] + joined["std"]
    )
    return df[keep].reset_index(drop=True)


def _drop_cheaper_than_smaller(df: pd.DataFrame) -> pd.DataFrame:
    """Drop N-BHK listings priced per sqft below the mean (N-1)-BHK listing nearby.

    Applies only where the location has more than five (N-1)-BHK listings.
    """
    stats = df.groupby(["location", "bhk"])["price_per_sqft"].agg(["mean", "count"])
    smaller = stats.rename(index=lambda b: b + 1, level="bhk")
    joined = df.join(smaller, on=["location", "bhk"])
    drop = (joined["count"] > MIN_SMALLER_LISTINGS) & (joined["price_per_sqft"] < joined["mean"])
    return df[~drop]


def features(df: pd.DataFrame, locations: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Area, bathrooms, bedrooms and one-hot location over `locations`; OTHER is the baseline."""
    place = pd.Categorical(df["location"], categories=locations)
    dummies = pd.get_dummies(place, dtype=float).drop(columns=OTHER)
    dummies.index = df.index
    x = pd.concat([df[["total_sqft", "bath", "bhk"]], dummies], axis="columns")
    return x, df["price"]


def log_area(x: pd.DataFrame) -> pd.DataFrame:
    """The same features with area on a log scale."""
    return x.assign(total_sqft=np.log(x["total_sqft"]))


def evaluate(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Scores on held-out listings, stated the way the page states accuracy.

    A non-positive estimate is refused, never shown, so it is counted and left out.
    R^2 is on log price: on price itself a handful of very expensive homes decide it.
    """
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    shown = predicted > 0
    error = np.abs(predicted[shown] - y[shown]) / y[shown]
    return {
        "median_abs_pct_error": round(float(np.median(error)), 4),
        "within_20_pct": round(float(np.mean(error <= 0.2)), 4),
        "r2_log": round(float(r2_score(np.log(y[shown]), np.log(predicted[shown]))), 4),
        "refused": int((~shown).sum()),
        "rows": len(y),
    }


def fit(x: pd.DataFrame, y: pd.Series) -> HuberRegressor:
    """Log price on log area, bathrooms, bedrooms and location, with a Huber loss.

    The location columns are almost all zero; as a sparse matrix the fit takes a tenth
    of the time and memory, for the same weights.
    """
    return HuberRegressor(max_iter=1000).fit(_sparse(x), np.log(y))


def predict(model: HuberRegressor, x: pd.DataFrame) -> np.ndarray:
    """Prices in lakh INR for the rows of `x`."""
    return np.asarray(np.exp(model.predict(_sparse(x))), dtype=float)


def _sparse(x: pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(log_area(x).to_numpy(dtype=float))


@dataclass(frozen=True)
class Trained:
    """A fitted model, the columns it was fitted on, its training rows and its scores."""

    model: HuberRegressor
    columns: list[str]
    rows: pd.DataFrame
    scores: dict[str, Any]


def train(df: pd.DataFrame) -> Trained:
    """Split first, fit on the training rows, score on the untouched test rows."""
    df = plausible(df)
    locations = sorted(df["location"].unique())
    train_rows, test_rows = train_test_split(df, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    x_train, y_train = features(train_rows, locations)
    x_test, y_test = features(test_rows, locations)
    model = fit(x_train, y_train)
    scores = {
        "held_out": evaluate(y_test.to_numpy(), predict(model, x_test)),
        "rows_train": len(train_rows),
        "baseline": evaluate(y_test.to_numpy(), baseline(train_rows, test_rows)),
        "course_method": course_method(train_rows, test_rows, locations),
        "cross_validated": cross_validated(df, locations),
    }
    return Trained(model, list(x_train.columns), train_rows, scores)


def baseline(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
    """The naive estimate a model has to beat: the location's median price per sq ft, times area."""
    rate = train_rows.groupby("location")["price_per_sqft"].median()
    per_sqft = test_rows["location"].map(rate).fillna(train_rows["price_per_sqft"].median())
    return np.asarray(per_sqft * test_rows["total_sqft"] / LAKH)


def cross_validated(df: pd.DataFrame, locations: list[str]) -> dict[str, float]:
    """The same fit scored on every listing once, over five folds, so one split is not luck."""
    predicted = np.empty(len(df))
    for train_index, test_index in KFold(FOLDS, shuffle=True, random_state=FOLD_SEED).split(df):
        x_train, y_train = features(df.iloc[train_index], locations)
        x_test, _ = features(df.iloc[test_index], locations)
        predicted[test_index] = predict(fit(x_train, y_train), x_test)
    return {**evaluate(df["price"].to_numpy(), predicted), "folds": FOLDS}


def course_method(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, locations: list[str]
) -> dict[str, float]:
    """The course's linear model, fitted on price-filtered rows, scored on the same test rows."""
    x_train, y_train = features(remove_outliers(train_rows), locations)
    x_test, y_test = features(test_rows, locations)
    model = LinearRegression().fit(x_train, y_train)
    return evaluate(y_test.to_numpy(), model.predict(x_test))


def export(trained: Trained, data_sha256: str) -> dict[str, Any]:
    """Everything a prediction needs, its bounds, and where it came from."""
    model, rows = trained.model, trained.rows
    weights = dict(zip(trained.columns, (float(c) for c in model.coef_), strict=True))
    return {
        "intercept": float(model.intercept_),
        "target": "log(price in lakh INR)",
        "coefficients": {
            "log_total_sqft": weights.pop("total_sqft"),
            "bath": weights.pop("bath"),
            "bhk": weights.pop("bhk"),
        },
        "locations": {OTHER: 0.0, **weights},
        "default_location": str(rows.loc[rows["location"] != OTHER, "location"].mode()[0]),
        "unit": "lakh INR",
        "bounds": {
            "min_sqft_per_bhk": MIN_SQFT_PER_BHK,
            "max_sqft": float(rows["total_sqft"].max()),
            "max_bhk": int(rows["bhk"].max()),
            "max_extra_bath": MAX_EXTRA_BATH,
        },
        "scores": trained.scores,
        "data_sha256": data_sha256,
        "trained_with": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Train, then write the weights as JSON and as a script the page loads."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    trained = train(clean(pd.read_csv(args.data)))
    payload = export(trained, hashlib.sha256(args.data.read_bytes()).hexdigest())
    text = json.dumps(payload, indent=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    # The same weights as a script: browsers refuse to fetch a JSON file from disk.
    args.out.with_suffix(".js").write_text(f"{BROWSER_GLOBAL} = {text};\n", encoding="utf-8")
    ours, course = trained.scores["held_out"], trained.scores["course_method"]
    print(
        f"{args.out}: half of held-out estimates within {ours['median_abs_pct_error']:.1%} "
        f"(course method {course['median_abs_pct_error']:.1%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
