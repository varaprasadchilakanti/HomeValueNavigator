# Contributing

Issues and pull requests are welcome. A pull request passes the checks in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml); run the same commands locally first.
After a change to `src/homevalue/train.py` or to a city's data, run
`uv run python -m homevalue.train --scope <scope>` and commit `model/<scope>/` and
`model/scopes.js` with it; a test fails if they disagree.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
A release moves the `[Unreleased]` notes in [CHANGELOG.md](CHANGELOG.md) under the new
version and date, sets the same version in `pyproject.toml`, and is tagged `v<version>` on
`main` once CI passes there.
