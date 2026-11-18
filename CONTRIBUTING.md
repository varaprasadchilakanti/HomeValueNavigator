# Contributing

Issues and pull requests are welcome. A pull request passes the checks in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml); run the same commands locally first.
After a change to `src/homevalue/train.py` or the data, run `uv run python -m homevalue.train`
and commit `model/model.json` and `model/model.js` with it; a test fails if they disagree.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
