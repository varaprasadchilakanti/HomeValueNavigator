# [Home Value Navigator](https://varaprasadchilakanti.github.io/HomeValueNavigator/)

Asking-price estimates for Bengaluru homes, computed in your browser. Enter the area, bedrooms,
bathrooms and location; get an estimate in lakh rupees with a typical range. Nothing you enter
leaves the page.

[![CI](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/varaprasadchilakanti/HomeValueNavigator/actions/workflows/ci.yml)
[![Python 3.11 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.13-253080)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-253080)](LICENSE)

## Accuracy

On 2,032 listings the model never saw, half of its estimates fall within 16.3% of the asking
price.

| | Half of estimates within | Within 20% |
|---|---|---|
| **This model** | **16.3%** (95% interval 15.5 to 17.2) | 58% |
| The course method, scored the same way | 21.8% | 47% |
| The location's median price per sq ft, times area | 16.9% | 57% |

- **Against the course method:** the model's median error is 4.4 to 6.7 points lower (95%
  interval, same resampled rows). The course method also prices 3 listings at zero or less;
  they are left out of its figures.
- **Against the per-sq-ft rule:** -0.3 to +1.3 points, so **no measured difference**. What the
  model adds is a price that moves with bedrooms and bathrooms.
- **Range:** the typical range held half (50%) of the held-out asking prices.
- **Why the course scores worse here:** it drops listings whose price per sq ft is unusual for
  their location before setting aside its test set
  ([its notebook](https://github.com/codebasics/py/tree/master/DataScience/BangloreHomePrices)),
  so its test listings were chosen by their own prices. A visitor's home has no price to filter
  on. Here the test set is set aside first, and rows identical in location, size, area,
  bathrooms and price are kept once, so a repeat cannot sit on both sides.
- **Caveat:** I looked at the held-out figures while choosing the plot-area rule and the Huber
  loss, so they are not a clean first look; five-fold cross-validation on the training rows
  alone gives 16.4%.

Every figure is in `scores` in `model/bengaluru/model.json`.

## Run it locally

```bash
uv sync                                  # Python 3.11+
uv run pytest -m "not rebuild"           # all but the 30 s retrain; the page test needs Node
uv run python -m homevalue.train         # retrain; rewrites model/
uv run --group serve waitress-serve --listen 127.0.0.1:8080 --max-request-body-size=1024 \
    --call homevalue.server:create_app
```

The API:

- `GET /api/locations` lists the locations.
- `POST /api/predict` takes `{"location", "total_sqft", "bhk", "bath"}` and returns
  `estimated_price`, `typical_low` and `typical_high` in lakh INR.
- Input the training listings don't cover, such as a 1 BHK over 6,000 sq ft, gets a 422 that
  names the field. Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem
  documents.

## How it works

- **The model:** linear, of log price on log area, bathrooms, bedrooms and location (164
  locations; those with 10 or fewer training listings are pooled), fitted with a Huber loss so a
  few mistyped listings pull it less.
- **The weights:** plain JSON, not a pickle, so loading the model runs no code. The page and the
  API both predict from it, and a test runs the page's JavaScript under Node to check it prices
  as the Python does.
- **Another city:** put its listings in `data/<scope>/listings.csv` beside a `scope.json`, where
  `<scope>` is the city's id. Run `uv run python -m homevalue.train --scope <scope>`, then commit
  `model/<scope>/` and `model/scopes.js`; no code changes. The API serves one city, the
  `model.json` that `HOMEVALUE_MODEL` names (Bengaluru's by default).

## Limits

- A learning project: don't use it to value a real property.
- Asking prices, not sale prices, from listings whose collection year is not recorded.
- A location adds the same percentage to every home in it.
- Half of the estimates miss by about 16% or more.

## Data, license and credit

- **Data:** 13,320 Bengaluru listings, the codebasics course's copy of a dataset it says is from
  Kaggle. Not covered by this repository's license; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **License:** the code and the page are [MIT](LICENSE). The fitted models are covered only as
  far as the data's terms allow.
- **Credit:** begun in 2024 as an exercise in the codebasics course
  [Real Estate Price Prediction](https://codebasics.io/courses/data-science-project-real-estate-price-prediction)
  by Dhaval Patel, which gave the data, its cleaning rules and the first shape; rebuilt in 2026.

The project is finished and not maintained. Forks are welcome under those terms; pull requests
are not merged. Security reports: [SECURITY.md](SECURITY.md).
