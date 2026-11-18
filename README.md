# Home Value Navigator

**Bengaluru home asking-price estimates, in the browser or over HTTP.**
Give an area, bedrooms, bathrooms and a location; it returns an estimate in lakh rupees with
the range half of its estimates fall within. A codebasics course project from 2024, rebuilt
in 2026 with a measured model, tests and CI.

[![CI](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml)
[![Live page](https://img.shields.io/badge/live-GitHub%20Pages-253080)](https://varaprasadchilakanti.github.io/HomeValueNavigator/)
[![Python 3.11 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.13-253080)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-253080)](LICENSE)

**[Open the estimator](https://varaprasadchilakanti.github.io/HomeValueNavigator/).**
It runs in your browser from the model's published weights; nothing you enter leaves the page.

```bash
uv sync                                    # Python 3.11+, uv 0.13
uv run python -m homevalue.train           # rebuild the model from the CSV
uv run --group serve waitress-serve --port 8080 --call homevalue.server:create_app
```

## What the rebuild found

The course reports R^2 of about 0.84. It removes listings with unusual prices before it splits
off its test set, so that score is measured on listings already filtered by their own price;
a visitor's home is not. Scored on untouched test listings, the course method puts half of
its estimates within 19.7% of the asking price, and gives a negative price for 3 of them.

This model is fitted the same way it is tested: the split comes first, and the test rows get
only the rules a visitor's input also passes. On 2,194 untouched listings:

| | Half within | Within 20% | R^2, log price |
|---|---|---|---|
| **This model** | **15.8%** (95% interval 15.1 to 16.4) | 60% | 0.78 |
| Course method, scored the same way | 19.7% | 50% | 0.64 |
| Naive rule: the location's median price per sq ft, times area | 15.3% | 60% | 0.76 |

On the same resampled rows, the model's median error is 2.7 to 5.0 points below the course
method's. Against the naive rule the difference is -1.0 to +0.2 points: **no measured
difference.** What the model adds is a price that responds to bedrooms and bathrooms, which
the rule cannot give. Five-fold cross-validation on the training rows gives 15.4%.

## Model card

| | |
|---|---|
| **Model** | Linear in log price: log area, bathrooms, bedrooms and location, fitted with a Huber loss so a few mistyped listings pull it less. 209 locations; those with 10 or fewer listings are pooled as the baseline |
| **Data** | `data/bengaluru_house_prices.csv`: 13,320 listings with asking prices, from the codebasics course; 10,970 pass the rules below |
| **Evaluation** | 2,194 held-out listings (20%, seed 10), not filtered by price; figures above. All scores are in `scores` in `model/model.json` |
| **Intended use** | Learning and demonstration: an end-to-end project from raw data to a working page and API |
| **Not for** | Valuation, lending, pricing or any decision about a real property |

**Cleaning.**
- Plot-area listings are left out: their area is land, not floor space. The course kept them.
- From the course: rows with gaps are dropped; an area given as a range becomes its midpoint,
  and one given in other units is dropped; listings under 300 sq ft per bedroom, or with more
  than one bathroom over the bedroom count, are dropped.

The course also drops listings whose price per sq ft is unusual for their location. Those
rules need the price, which a visitor does not have, so this model is trained and tested
without them.

**Limitations.**
- **The listings are dated.** The year they were collected is not recorded with the data.
  They are asking prices, not sale prices.
- **A location adds the same percentage everywhere.** An extra bedroom cannot be worth more in
  one area than another.
- **Half of the estimates miss by about 16% or more.** An estimate is a starting point, not a
  figure to quote.

## The API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/locations` | | `{"locations": [...]}`, the 209 locations |
| `POST` | `/api/predict` | `{"location", "total_sqft", "bhk", "bath"}` | `{"location", "estimated_price", "likely_low", "likely_high", "unit": "lakh INR"}` |

Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem documents
(`application/problem+json`):
- `400` for a body that is not a JSON object;
- `413` for a body over 1 KiB;
- `415` for another content type;
- `422` for a request the model cannot answer, with `field` naming the input;
- `500`, logged, for a fault of the server's own.

The model answers only within the data it learned from, with the bounds in `model/model.json`:
an area of at least 300 sq ft per bedroom, up to 36,000 sq ft; 1 to 16 bedrooms; at most one
bathroom more than bedrooms; a listed location.

| Setting | Default | Purpose |
|---|---|---|
| `HOMEVALUE_MODEL` | `model/model.json` | the weights to serve |
| `HOMEVALUE_ALLOWED_ORIGIN` | none | the one origin allowed to call cross-origin; a wildcard is refused |

## What it costs to run

Measured on x86-64, 2 cores, Python 3.12 (`uv run python scripts/measure.py`):

| | |
|---|---|
| Page a visitor downloads | 33.6 KiB in 6 files (12.7 KiB gzipped); no third-party scripts, fonts or images |
| One prediction | 23 us median, in process |
| One API request | 1.8 ms median, Flask test client, no network |
| Training, with all the scores | 34 s, peak 220 MiB |

## Layout

| Path | What |
|---|---|
| `src/homevalue/train.py` | cleaning, the fit and every score; writes `model/model.json` and `model/model.js` |
| `src/homevalue/model.py` | prediction from the JSON weights, with no scikit-learn and no pickle |
| `src/homevalue/server.py` | the HTTP API |
| `client/` | the page: HTML, CSS and JavaScript reading `model/model.js` |
| `notebooks/` | the 2024 course notebook, kept for the record |
| `scripts/measure.py` | the cost figures above |

## Dependencies

| Group | Packages | Needed for |
|---|---|---|
| runtime | Flask | the API |
| `serve` | waitress | serving the API in production |
| `train` | NumPy, pandas, scikit-learn, SciPy | rebuilding the model |
| `dev` | the `train` group, pytest, pytest-cov, ruff, mypy, pandas-stubs | tests and checks |

Versions are locked in `uv.lock`. The page needs no package. The model is JSON, not a
pickle, so loading it runs no code.

## Checks

CI runs ruff, strict mypy and pytest with a 90% coverage floor on Python 3.11 and 3.13; the
commands are in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). The tests:
- refuse hostile inputs, each for its own stated reason, and accept every range at its limits;
- rebuild the committed model from the data and compare every figure;
- check that the JSON model predicts what scikit-learn predicts;
- check that no test listing was filtered by price.

## History

Built in 2024 while following the codebasics course
[Real Estate Price Prediction](https://codebasics.io/courses/data-science-project-real-estate-price-prediction):
- a Jupyter notebook (kept in `notebooks/`) cleaned the data and chose the model;
- a pickled model ran behind a Flask server;
- a jQuery page called that server.

Changes since then are in [CHANGELOG.md](CHANGELOG.md). Contributions:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- The data, the cleaning rules and the project's shape come from the codebasics course by
  Dhaval Patel.
- The course states that the data comes from Kaggle. Its licence is not stated, and the
  dataset is not covered by this repository's licence; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The code is [MIT](LICENSE). Security reports: [SECURITY.md](SECURITY.md).
