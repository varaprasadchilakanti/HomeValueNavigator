# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- A seeded training script, replacing the notebook as the model's source.
- Scores beside the model: a naive baseline, the course method scored the same way,
  five-fold cross-validation on the training rows, and paired 95% intervals.
- The typical accuracy beside every estimate, as a typical range.
- The page estimates in the browser, so the GitHub Pages link works without a server.
- Light, dark and system themes, a searchable location list, accessible errors, the price
  per sq ft and a favicon.
- waitress, to serve the API in production.
- Tests, type checks, a 90% coverage floor, CI and `scripts/measure.py`.

### Changed
- The model predicts log price from log area with a Huber loss, and is tested on listings
  not filtered by their price. Half of its held-out estimates fall within 15.8% of the
  asking price; the course method, scored the same way, 19.7%.
- The model ships as JSON weights, with its input bounds, instead of a pickle.
- The API is `GET /api/locations` and `POST /api/predict` with a JSON body. Errors are RFC
  9457 problem documents. It replaces `/get_location_names` and `/predict_home_price`.

### Fixed
- An unknown location is refused, instead of being priced as if it had none.
- Invalid input gets a reason, instead of a null price.
- Plot-area listings are left out: their area is land, not floor space.

### Removed
- The pickled model, `requirements.txt`, jQuery and the background image.

### Security
- Cross-origin requests are allowed only from `HOMEVALUE_ALLOWED_ORIGIN`, never from every
  origin; a wildcard is refused at start-up.
- Request bodies over 1 KiB are refused, and a server fault returns no internal detail.
- The model is JSON, so loading it runs no code.

## [0.1.0] - 2024-12-17

The course project: notebook, pickled model, Flask server, jQuery page.

[Unreleased]: https://github.com/varaprasadchilakanti/HomeValueNavigator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/varaprasadchilakanti/HomeValueNavigator/tree/v0.1.0
