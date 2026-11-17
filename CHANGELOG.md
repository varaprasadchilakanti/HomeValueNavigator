# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- A seeded training script, replacing the notebook as the model's source.
- The typical accuracy beside every estimate, as a likely range.
- The page estimates in the browser, so the GitHub Pages link works without a server.
- Light, dark and system themes, a searchable location list, accessible errors, the price
  per sq ft, a favicon and a Content Security Policy.
- Tests, type checks, a 90% coverage floor, CI, Dependabot and `scripts/measure.py`.

### Changed
- The model predicts log price from log area, and is tested on listings not filtered by
  their price. Half of its held-out estimates fall within 19.3% of the listed price;
  the course method, scored the same way, 22.7%.
- The model ships as JSON weights instead of a pickle.
- The API is `GET /api/locations` and `POST /api/predict` with a JSON body. Errors are RFC
  9457 problem documents. It replaces `/get_location_names` and `/predict_home_price`.
- Cross-origin requests are allowed only from `HOMEVALUE_ALLOWED_ORIGIN`, not from every
  origin.

### Fixed
- An unknown location is refused, instead of being priced as if it had none.
- Invalid input gets a reason, instead of a null price.

### Removed
- The pickled model, `requirements.txt`, jQuery and the background image.

## [0.1.0] - 2024-12

The course project: notebook, pickled model, Flask server, jQuery page.

[Unreleased]: https://github.com/varaprasadchilakanti/HomeValueNavigator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/varaprasadchilakanti/HomeValueNavigator/tree/v0.1.0
