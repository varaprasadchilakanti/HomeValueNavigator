"""Train the model from the CSV and write its coefficients as JSON.

    uv run python -m homevalue.train            # writes model/model.json

The cleaning follows the 2024 notebook step by step (notebooks/), so the model it
writes is the one the course method produces. What changed is where it lives: a seeded
script with a held-out score, instead of cells run by hand, and coefficients in JSON
instead of a pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "bengaluru_house_prices.csv"
OUT = ROOT / "model" / "model.json"
#: The split and seeds the 2024 notebook used, kept so the score is comparable.
TEST_SIZE = 0.2
SPLIT_SEED = 10
CV_SEED = 0
RARE_LOCATION = 10  # a location with this many listings or fewer becomes "other"
MIN_SQFT_PER_BHK = 300  # smaller per bedroom was treated as a data error
MIN_SMALLER_LISTINGS = 5  # the BHK rule applies only where the smaller size has more than this


def sqft_to_number(value: str) -> float | None:
    """'1200' -> 1200.0, '2100 - 2850' -> the midpoint, anything with a unit -> None."""
    parts = value.split("-")
    try:
        if len(parts) == 2:  # a range: its two ends
            return (float(parts[0]) + float(parts[1])) / 2
        return float(value)
    except ValueError:
        return None


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """The notebook's cleaning, in its order: columns, nulls, BHK, area, rare locations."""
    df = raw.drop(columns=["area_type", "availability", "society", "balcony"]).dropna()
    df = df.assign(bhk=df["size"].map(lambda s: int(str(s).split(" ")[0])))
    df = df.assign(total_sqft=df["total_sqft"].map(lambda s: sqft_to_number(str(s)))).dropna()
    df = df.assign(location=df["location"].str.strip())
    df = df.assign(price_per_sqft=df["price"] * 100_000 / df["total_sqft"])
    counts = df["location"].value_counts()
    rare = set(counts[counts <= RARE_LOCATION].index)
    return df.assign(location=df["location"].where(~df["location"].isin(rare), "other"))


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's three outlier rules, then its bathroom rule."""
    df = df[df["total_sqft"] / df["bhk"] >= MIN_SQFT_PER_BHK]
    df = _within_one_std(df)
    df = _drop_cheaper_than_smaller(df)
    return df[df["bath"] < df["bhk"] + 2]


def _within_one_std(df: pd.DataFrame) -> pd.DataFrame:
    """Per location, keep price per sqft within one standard deviation of its mean."""
    stats = df.groupby("location")["price_per_sqft"].agg(["mean", "std"])
    # pandas' std is the sample std; the notebook used numpy's population std (ddof=0).
    stats["std"] = df.groupby("location")["price_per_sqft"].agg(lambda s: np.std(s))
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


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Area, bathrooms, bedrooms and one-hot location; "other" is the baseline."""
    dummies = pd.get_dummies(df["location"], dtype=float).drop(columns="other")
    x = pd.concat([df[["total_sqft", "bath", "bhk"]], dummies], axis="columns")
    return x, df["price"]


def fit(x: pd.DataFrame, y: pd.Series) -> tuple[LinearRegression, dict[str, float]]:
    """Fit on the training split; score on the held-out split and by 5-fold shuffle split."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    model = LinearRegression().fit(x_train, y_train)
    folds = ShuffleSplit(n_splits=5, test_size=TEST_SIZE, random_state=CV_SEED)
    cv = cross_val_score(LinearRegression(), x, y, cv=folds)
    return model, {
        "r2_held_out": round(float(model.score(x_test, y_test)), 4),
        "mae_held_out_lakh": round(float(mean_absolute_error(y_test, model.predict(x_test))), 2),
        "r2_cv_mean": round(float(cv.mean()), 4),
        "rows_train": len(x_train),
        "rows_held_out": len(x_test),
    }


def export(
    model: LinearRegression, x: pd.DataFrame, scores: dict[str, float], data_sha256: str
) -> dict[str, Any]:
    """Everything a prediction needs, and enough to say where it came from."""
    weights = dict(zip(x.columns, (float(c) for c in model.coef_), strict=True))
    return {
        "intercept": float(model.intercept_),
        "coefficients": {k: weights.pop(k) for k in ("total_sqft", "bath", "bhk")},
        "locations": {"other": 0.0, **weights},
        "unit": "lakh INR",
        "scores": scores,
        "data_sha256": data_sha256,
        "trained_with": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Train and write the JSON model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    raw_bytes = args.data.read_bytes()
    df = remove_outliers(clean(pd.read_csv(args.data)))
    x, y = features(df)
    model, scores = fit(x, y)
    payload = export(model, x, scores, hashlib.sha256(raw_bytes).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"{args.out}: {len(df)} listings, held-out R^2 {scores['r2_held_out']}, "
        f"typical error {scores['mae_held_out_lakh']} lakh"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
