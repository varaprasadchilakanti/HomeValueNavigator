# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-11-17

### Added
- A seeded training script with a held-out score, replacing the notebook as the model's source.
- The model as JSON weights, read without scikit-learn.
- In-browser prediction, so the GitHub Pages page works without a server.
- The typical error beside every estimate.
- Tests, type checks, a coverage floor, CI, and `scripts/measure.py`.

### Changed
- The API is `GET /api/locations` and `POST /api/predict` with a JSON body. It replaces `/get_location_names` and the form-encoded `/predict_home_price`.
- Cross-origin requests are allowed only from `HOMEVALUE_ALLOWED_ORIGIN`, instead of from every origin.

### Fixed
- An unknown location is refused, instead of being priced as if it had none.
- Invalid input returns a 422 with the reason, instead of a null price.

### Removed
- The pickled model, `requirements.txt`, and the jQuery dependency.

## 2024

The original course project, built in December 2024: notebook, pickled model, Flask server, jQuery page.
