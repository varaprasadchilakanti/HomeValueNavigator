# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- A seeded training script, replacing the notebook as the model's source.
- Scores beside the model: a naive baseline, the course method scored the same way,
  five-fold cross-validation on the training rows, paired 95% intervals, and how often the
  typical range holds the asking price.
- A test that runs the page's JavaScript under Node and checks it against the Python model.
- The typical accuracy beside every estimate, as a typical range.
- The page estimates in the browser, so the GitHub Pages link works without a server.
- Light, dark and system themes, a searchable location list, accessible errors, the price
  per sq ft and a favicon.
- waitress, to serve the API in production.
- Tests, type checks, a 90% coverage floor, CI and `scripts/measure.py`.

### Changed
- The model predicts log price from log area with a Huber loss, and is tested on listings
  not filtered by their price. Half of its held-out estimates fall within 16.3% of the
  asking price; the course method, scored the same way, 21.8%.
- The model ships as JSON weights, with its input bounds, instead of a pickle. The area
  bound is the largest training listing for each bedroom count.
- The API is `GET /api/locations` and `POST /api/predict` with a JSON body. Errors are RFC
  9457 problem documents. It replaces `/get_location_names` and `/predict_home_price`.

### Fixed
- An unknown location is refused, instead of being priced as if it had none.
- Invalid input gets a reason, instead of a null price.
- Plot-area listings are left out: their area is land, not floor space.
- Exact repeats are dropped, so no listing is in both the training and the test rows.
- A number sent as text is read by one rule, the same in Python and on the page.

### Removed
- The pickled model, `requirements.txt`, jQuery and the background image.

### Security
- The API no longer allows cross-origin requests from every origin; it sends no
  cross-origin headers at all.
- Request bodies over 1 KiB, or without a length, are refused; responses carry
  `X-Content-Type-Options: nosniff`; a server fault returns no internal detail.
- The model is JSON, so loading it runs no code.

## [0.1.0] - 2024-12-17

The course project: notebook, pickled model, Flask server, jQuery page.

[Unreleased]: https://github.com/varaprasadchilakanti/HomeValueNavigator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/varaprasadchilakanti/HomeValueNavigator/tree/v0.1.0
