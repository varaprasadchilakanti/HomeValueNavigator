# Home Value Navigator

**Bengaluru home asking-price estimates, in the browser or over HTTP.**
The course this began from scores R^2 of about 0.85 in its five-fold cross-validation, on
listings already filtered by their own price. Rebuilt so the test listings are untouched, the same method puts half
of its estimates within 21.8% of the asking price; this model, within 16.3%. Give an area,
bedrooms, bathrooms and a location; it returns an estimate in lakh rupees with the range
about half of asking prices fall within.

[![CI](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml)
[![Live page](https://img.shields.io/badge/live-GitHub%20Pages-253080)](https://varaprasadchilakanti.github.io/HomeValueNavigator/)
[![Python 3.11 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.13-253080)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-253080)](LICENSE)

**[Open the estimator](https://varaprasadchilakanti.github.io/HomeValueNavigator/).**
It runs in your browser from the model's published weights; nothing you enter leaves the page.

```bash
uv sync                                    # Python 3.11+, uv 0.13
uv run python -m homevalue.train           # rebuild the model from data/bengaluru/
uv run --group serve waitress-serve --port 8080 --max-request-body-size=1024 \
    --call homevalue.server:create_app
```

## What the rebuild found

The course removes listings with unusual prices before it splits off its test set, so its
score is measured on listings already filtered by their own price; a visitor's home is not.
Here the split comes first: exact repeats are dropped so no home sits on both sides, rare
locations are pooled by their count in the training rows, and the test rows get only the
rules a visitor's input also passes. On 2,032 untouched listings:

| | Half within | Within 20% | R^2, log price |
|---|---|---|---|
| **This model** | **16.3%** (95% interval 15.5 to 17.2) | 58% | 0.77 |
| Course method, scored the same way | 21.8% | 47% | 0.56 |
| Naive rule: the location's median price per sq ft, times area | 16.9% | 57% | 0.74 |

On the same resampled rows, the model's median error is 4.4 to 6.7 points below the course
method's, which also gives a negative price for 3 listings. Against the naive rule the
difference is -0.3 to +1.3 points: **no measured difference.** What the model adds is a
price that responds to bedrooms and bathrooms, which the rule cannot give. Five-fold
cross-validation on the training rows gives 16.4%, and the typical range holds 49.9% of
held-out asking prices.

The held-out figures were seen while the plot-area rule and the Huber loss were chosen, so
they are not a first look; the cross-validated figure, on training rows only, agrees.

## Model card

| | |
|---|---|
| **Model** | Linear in log price: log area, bathrooms, bedrooms and location, fitted with a Huber loss so a few mistyped listings pull it less. 164 locations; those with 10 or fewer training listings are pooled as the baseline |
| **Data** | `data/bengaluru/listings.csv`: 13,320 listings with asking prices, the codebasics course's copy of a dataset it says is from Kaggle ([notices](THIRD_PARTY_NOTICES.md)); 10,160 distinct listings pass the rules below |
| **Evaluation** | 2,032 held-out listings (20%, seed 10), not filtered by price; figures above. All scores are in `scores` in `model/bengaluru/model.json` |
| **Intended use** | Learning and demonstration: an end-to-end project from raw data to a working page and API |
| **Not for** | Valuation, lending, pricing or any decision about a real property |

**Cleaning.**
- Plot-area listings are left out: their area is land, not floor space. The course kept them.
- Exact repeats of a listing are dropped. The course kept them.
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
| `GET` | `/api/locations` | | `{"locations": [...]}`, the 164 locations |
| `POST` | `/api/predict` | `{"location", "total_sqft", "bhk", "bath"}` | `{"location", "estimated_price", "typical_low", "typical_high", "unit": "lakh INR"}` |

`typical_low` and `typical_high` bound the range about half of asking prices fall within.
Numbers are JSON numbers, or text of ASCII digits with optional thousands commas. A
location matches whatever its case and spacing, and the response names the one it matched.
Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem documents
(`application/problem+json`):
- `400` for a body that is not a JSON object;
- `405` for another method, with `Allow` listing the methods the path takes;
- `411` for a body sent without a length;
- `413` for a body over 1 KiB;
- `415` for another content type;
- `422` for a request the model cannot answer, with `field` naming the input;
- `500`, logged, for a fault of the server's own.

The model answers only within the data it learned from, with the bounds in `model/bengaluru/model.json`:
a bedroom count the training listings have; an area from 300 sq ft per bedroom up to the
largest training listing with that many bedrooms (6,000 sq ft for 1 BHK, 8,840 for 2); at
most one bathroom more than bedrooms; a listed location.

| Setting | Default | Purpose |
|---|---|---|
| `HOMEVALUE_MODEL` | `model/bengaluru/model.json` in a checkout | the weights to serve; set it when the package is installed elsewhere |

## What it costs to run

Measured with `uv run python scripts/measure.py`, on two cores:

| | |
|---|---|
| Page a visitor downloads | 49.5 KiB in 7 files (17.6 KiB gzipped); no third-party scripts, fonts or images. A test holds it under 20 KiB gzipped |
| One prediction | 27 us median, in process |
| One API request | 1.9 ms median, Flask test client, no network |
| Training, with all the scores | 27 s, peak 211 MiB |

## Layout

| Path | What |
|---|---|
| `data/<scope>/` | one city: `scope.json` (its name, units and data source) and `listings.csv` |
| `src/homevalue/train.py` | cleaning, the fit and every score; writes `model/<scope>/model.json`, the same as `model.js` for the page, and `model/scopes.js`, the page's list of cities |
| `src/homevalue/model.py` | prediction from the JSON weights, with no scikit-learn and no pickle |
| `src/homevalue/server.py` | the HTTP API |
| `client/` | the page: HTML, CSS and JavaScript reading `model/scopes.js`, then the chosen city's `model.js` |
| `tests/` | the checks below |
| `scripts/measure.py` | the cost figures above |

## Adding a city

A city is a folder, not a code change. Put its listings in `data/<scope>/listings.csv`, in
the columns the Bengaluru file has, beside a `scope.json` like `data/bengaluru/scope.json`;
then run `uv run python -m homevalue.train --scope <scope>` and commit `model/<scope>/` and
`model/scopes.js`. The page lists the new city's locations under its name
(`tests/test_page.py`), and `tests/test_scope.py` trains and serves a made-up town. The page formats prices in lakh
rupees and areas in sq ft, and refuses a model in other units. The API serves one city,
chosen by `HOMEVALUE_MODEL`.

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
- check that the JSON model predicts what scikit-learn predicts, and that the page's own
  JavaScript, run under Node, prices and reads input as the Python model does;
- check that no test listing was filtered by price, and that serving imports none of the
  training stack;
- check that the paths and version the documents state, and the figures in the pages'
  descriptions, are true; that
  the page stays within its size budget, and that its colors meet WCAG 2.2 AA in both themes.

## History

Built in 2024 while following the codebasics course
[Real Estate Price Prediction](https://codebasics.io/courses/data-science-project-real-estate-price-prediction):
- a Jupyter notebook, following the course's code, cleaned the data and chose the model
  (not kept here; the course's own is in its
  [repository](https://github.com/codebasics/py/tree/master/DataScience/BangloreHomePrices));
- a pickled model ran behind a Flask server;
- a jQuery page called that server.

Changes since then are in [CHANGELOG.md](CHANGELOG.md). Contributions:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- The codebasics course by Dhaval Patel is the baseline: the data, its cleaning rules and
  the project's first shape come from it. The evaluation, the model, the API and the page
  go further, and are measured against it above.
- The dataset is the course's copy; the course says it is from Kaggle. It is not covered
  by this repository's license: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- codebasics, Dhaval Patel and Kaggle are named for attribution only; none of them has
  reviewed or endorses this project.

## License

The code is [MIT](LICENSE). Security reports: [SECURITY.md](SECURITY.md).
