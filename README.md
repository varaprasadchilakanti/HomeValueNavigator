# HomeValueNavigator

**Bengaluru home price estimates from a linear regression, in the browser or over HTTP.**
Give an area, bedrooms, bathrooms and a location; it returns an estimate in lakh rupees and the
model's typical error. A course project from 2024, maintained to current practice since 2026.

[![CI](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Try it:** [the live page](https://varaprasadchilakanti.github.io/HomeValueNavigator/client/app.html)
predicts in your browser from the model's published weights. Nothing you enter leaves the page.

```bash
uv sync                                        # Python 3.11+, uv 0.13+
uv run python -m homevalue.train               # rebuilds model/model.json from the CSV
uv run flask --app homevalue.server run        # the API on http://127.0.0.1:5000
```

## The API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/locations` | | `{"locations": [...]}`, the 241 trained locations |
| `POST` | `/api/predict` | `{"location", "total_sqft", "bhk", "bath"}` | `{"estimated_price", "typical_error", "unit": "lakh INR"}` |

A request the model cannot answer gets `422` and `{"error": "<reason>"}`: a location it was not
trained on, an area under 300 sq ft per bedroom or over 30,000, bedrooms outside 1 to 10,
bathrooms more than one over bedrooms, a value that is not a finite number, or inputs whose
estimate is not positive. Cross-origin requests are allowed only from the origin set in
`HOMEVALUE_ALLOWED_ORIGIN`; unset, none is.

## Model card

| | |
|---|---|
| **Model** | Ordinary least squares on area, bathrooms, bedrooms and one-hot location (240 locations; rarer ones pooled as `other`, the baseline) |
| **Data** | `data/bengaluru_house_prices.csv`, 13,320 listings with asking prices in lakh INR; 7,239 kept after cleaning. SHA-256 `23ddf71be7d7...` |
| **Evaluation** | Held-out 20% (1,448 listings, seed 10): R^2 **0.843**, mean absolute error **18.1 lakh**. 5-fold shuffle split (seed 0): mean R^2 0.830 |
| **Trained with** | Python 3.12.3, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.5 (recorded in `model/model.json`) |
| **Intended use** | Learning and demonstration: an end-to-end ML project from raw data to a working page and API |
| **Not for** | Valuation, lending, pricing or any decision about a real property |

**Cleaning**, as the course teaches it and in its order (`src/homevalue/train.py`):
- columns that do not describe the property are dropped, and so are rows with gaps;
- an area given as a range becomes its midpoint, and one given in other units is dropped;
- locations with 10 or fewer listings become `other`;
- listings under 300 sq ft per bedroom are dropped;
- per location, price per sq ft outside one standard deviation of the mean is dropped;
- an N-bedroom listing priced per sq ft below the location's mean (N-1)-bedroom listing is dropped;
- listings with more than one bathroom over the bedroom count are dropped.

**Limitations.**
- The listings are dated; the year they were collected is not recorded with the data. Prices are asking prices, not sale prices, and are not current.
- A linear model prices every location as a fixed offset: it cannot learn that an extra bedroom is worth more in one area than another.
- The typical error is 18.1 lakh, so an estimate is a rough guide, not a number to quote.

## What it costs to run

Measured on an x86-64 machine with 2 cores, Python 3.12.3 (`uv run python scripts/measure.py`):

| | |
|---|---|
| Page a visitor downloads | 16.8 KiB in 4 files (7.5 KiB gzipped); no third-party scripts, fonts or images |
| Model file | 9.6 KiB of JSON; no pickle, so loading it runs no code |
| One prediction | 22 us median, in process |
| One API request | 1.9 ms median, Flask test client, no network |
| Training | 14.5 s, peak 221 MiB |

## Dependencies

| Group | Packages | Needed for |
|---|---|---|
| runtime | Flask | the API only |
| `train` | NumPy, pandas, scikit-learn | rebuilding the model |
| `dev` | the `train` group, pytest, pytest-cov, ruff, mypy, pandas-stubs | tests and checks |

Versions are locked in `uv.lock`. The page needs no package at all: it is plain HTML, CSS and
JavaScript reading `model/model.json`.

## Checks

`uv run ruff check . && uv run mypy src tests scripts && uv run pytest --cov` runs what CI
runs on Python 3.11 and 3.13. The tests:
- feed the model and the API hostile inputs;
- pair each refusal with an accepted case;
- check that the committed model is what the data and code produce;
- check that the JSON model predicts what scikit-learn predicts.

Coverage must stay at or above 90%.

## History

**As built in 2024**, following the codebasics course: the data cleaned and the model chosen
in a Jupyter notebook (kept in `notebooks/`) by grid search over linear regression, lasso and a
decision tree; a pickled model behind a Flask server; a jQuery page that called the server. The
course deployed the server on AWS EC2 behind nginx.

**Maintained since 2026:**
- The notebook's steps became a seeded script with a held-out score.
- The pickle became JSON weights, so the server and the page predict without scikit-learn.
- The page now predicts on its own, which is what lets the GitHub Pages link work.
- The API refuses invalid input instead of guessing; the 2024 version priced an unknown location as if it had none.
- Every estimate shows its typical error.
- The project gained tests, type checks and CI.

## Acknowledgements

- The method and the dataset come from the codebasics course
  [Real Estate Price Prediction](https://codebasics.io/courses/data-science-project-real-estate-price-prediction)
  by Dhaval Patel.
- The course states that the data comes from Kaggle. The dataset is not covered by this
  repository's license, and its own license is not stated by the course.

## License

The code is [MIT](LICENSE).
