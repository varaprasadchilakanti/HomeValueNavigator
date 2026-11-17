# HomeValueNavigator

**Bengaluru home price estimates, in the browser or over HTTP.**
Give an area, bedrooms, bathrooms and a location; it returns an estimate in lakh rupees with
the range half of its estimates fall within. A codebasics course project from 2024, rebuilt
in 2026 with a measured model, tests and CI.

[![CI](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

**[Open the estimator](https://varaprasadchilakanti.github.io/HomeValueNavigator/client/app.html).**
It runs in your browser from the model's published weights; nothing you enter leaves the page.

```bash
uv sync                                    # Python 3.11+, uv 0.13+
uv run python -m homevalue.train           # rebuild the model from the CSV
uv run --group serve waitress-serve --port 8080 --call homevalue.server:create_app
```

## The API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/locations` | | `{"locations": [...]}`, the 240 locations |
| `POST` | `/api/predict` | `{"location", "total_sqft", "bhk", "bath"}` | `{"estimated_price", "likely_low", "likely_high", "unit": "lakh INR"}` |

Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem documents
(`application/problem+json`):
- `400` for a body that is not a JSON object;
- `415` for another content type;
- `413` for a body over 1 KiB;
- `422` for a request the model cannot answer, with `field` naming the input.

The model answers only within the data it learned from:
- an area of at least 300 sq ft per bedroom, up to 42,000 sq ft;
- 1 to 16 bedrooms;
- at most one bathroom more than bedrooms;
- a listed location.

| Setting | Default | Purpose |
|---|---|---|
| `HOMEVALUE_MODEL` | `model/model.json` | the weights to serve |
| `HOMEVALUE_ALLOWED_ORIGIN` | none | the one origin allowed to call cross-origin; a wildcard is refused |

## Model card

| | |
|---|---|
| **Model** | Linear regression of log price on log area, bathrooms, bedrooms and location (240 locations; those with 10 or fewer listings pooled as the baseline) |
| **Data** | `data/bengaluru_house_prices.csv`: 13,320 listings with asking prices, from the codebasics course; 12,303 pass the plausibility rules below |
| **Evaluation** | 2,461 held-out listings (20%, seed 10), not filtered by price: half of the estimates within **19.3%** of the listed price; 51% within 20% |
| **Intended use** | Learning and demonstration: an end-to-end project from raw data to a working page and API |
| **Not for** | Valuation, lending, pricing or any decision about a real property |

**Cleaning.** The course's rules:
- rows with gaps are dropped;
- an area given as a range becomes its midpoint, and one given in other units is dropped;
- listings under 300 sq ft per bedroom are dropped, and so are those with more than one
  bathroom over the bedroom count.

The course also drops listings whose price per sq ft is unusual for their location. Those
rules need the price, which a visitor does not have, so this model is trained and tested
without them.

**Compared with the course method.** The course fits a straight-line model and reports
R^2 of about 0.84, measured after removing price outliers from the test set as well.
Scored on the same untouched test listings, the course method puts half of its estimates
within 22.7% and gives a negative price for 12 of them. This model's figure is 19.3%, and
its estimates are always positive. `scores` in `model/model.json` holds both.

**Limitations.**
- **The listings are dated.** The year they were collected is not recorded with the data.
  They are asking prices, not sale prices.
- **A location adds the same percentage everywhere.** An extra bedroom cannot be worth more in
  one area than another.
- **Half of the estimates miss by more than 19%.** An estimate is a starting point, not a
  figure to quote.

## What it costs to run

Measured on x86-64, 2 cores, Python 3.12 (`uv run python scripts/measure.py`):

| | |
|---|---|
| Page a visitor downloads | 32.4 KiB in 4 files (12.3 KiB gzipped); no third-party scripts, fonts or images |
| One prediction | 24 us median, in process |
| One API request | 1.8 ms median, Flask test client, no network |
| Training | 10.3 s, peak 228 MiB |

## Dependencies

| Group | Packages | Needed for |
|---|---|---|
| runtime | Flask | the API |
| `serve` | waitress | serving the API in production |
| `train` | NumPy, pandas, scikit-learn | rebuilding the model |
| `dev` | the `train` group, pytest, pytest-cov, ruff, mypy, pandas-stubs | tests and checks |

Versions are locked in `uv.lock`; Dependabot proposes updates weekly. The page needs no
package: it is HTML, CSS and JavaScript reading `model/model.js`, the same weights as
`model/model.json`. The model is JSON, not a pickle, so loading it runs no code.

## Checks

CI runs these on Python 3.11 and 3.13:

```bash
uv run --locked ruff check . && uv run --locked ruff format --check .
uv run --locked mypy src tests scripts
uv run --locked pytest --cov          # fails under 90% coverage
```

The tests:
- refuse hostile inputs, each for its own stated reason;
- accept every range at its limits;
- rebuild the committed model from the data;
- check that the JSON model predicts what scikit-learn predicts;
- check that no test listing was filtered by price.

## History

Built in 2024 while following the codebasics course
[Real Estate Price Prediction](https://codebasics.io/courses/data-science-project-real-estate-price-prediction):
- a Jupyter notebook (kept in `notebooks/`) cleaned the data and chose the model;
- a pickled model ran behind a Flask server;
- a jQuery page called that server.

Changes since then are in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements

- The data, the cleaning rules and the project's shape come from the codebasics course by
  Dhaval Patel.
- The course states that the data comes from Kaggle. Its licence is not stated, and the
  dataset is not covered by this repository's licence; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The code is [MIT](LICENSE). Security reports: [SECURITY.md](SECURITY.md).
