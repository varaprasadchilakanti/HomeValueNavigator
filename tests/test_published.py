# SPDX-License-Identifier: MIT
"""What the repository tells the world stays true: paths, figures, version, weight, contrast."""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from homevalue import train

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"]
#: What a visitor downloads, with every model/*.js; the root index.html only redirects.
PAGE = [ROOT / "client" / n for n in ("app.html", "app.css", "theme.js", "app.js", "favicon.svg")]
PAGE_BUDGET_GZIP = 20 * 1024  # bytes, each file compressed alone; about 17.9 KiB on 2026-11-19
NAMED = re.compile(r"`([\w./<>*-]+)`")
FILE = re.compile(r"\.(py|js|json|csv|md|html|css|svg|toml|lock|yml|txt|ipynb)$")
#: Names of files not in this repository: removed ones, and the course's own.
ELSEWHERE = {
    "CHANGELOG.md": {"requirements.txt"},
    "THIRD_PARTY_NOTICES.md": {"bengaluru_house_prices.csv"},
}
LINK = re.compile(r"\]\((?!https?:|#)([^)#]+)")
RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.MULTILINE)
SINGLE = re.compile(r"--([\w-]+): (#[0-9a-f]{6});")
PAIR = re.compile(r"--([\w-]+): light-dark\((#[0-9a-f]{6}), (#[0-9a-f]{6})\)")
TEXT = ("ink", "muted", "royal", "error", "gold")
#: (foreground token, background token, the WCAG 2.2 AA minimum for that use)
CONTRAST = [
    *((fg, bg, 4.5) for fg in TEXT for bg in ("bg", "surface")),
    ("ink", "raised", 4.5),
    ("muted", "raised", 4.5),
    ("royal", "raised", 4.5),  # a match marked in the location list
    ("royal", "royal-soft", 4.5),
    ("ink", "royal-soft", 4.5),  # the highlighted option
    ("muted", "royal-soft", 4.5),  # the clearing row, highlighted
    ("on-royal", "royal-fill", 4.5),
    ("field", "surface", 3.0),
]


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def named(doc: str) -> list[str]:
    """Backticked names that look like paths: with a folder, or a file's extension; not URLs."""
    names = NAMED.findall(read(doc))
    return [n for n in names if not n.startswith("/") and ("/" in n or FILE.search(n))]


@pytest.mark.parametrize("doc", [d for d in DOCS if d != "SECURITY.md"])  # it names no file
def test_every_path_a_document_names_exists(doc: str) -> None:
    files = tracked()
    names = [n for n in named(doc) if n not in ELSEWHERE.get(doc, set())]
    assert names, f"{doc} names no path, so this test would check nothing"
    for name in names:
        pattern = re.escape(name.rstrip("/")).replace("<scope>", "[^/]+").replace(r"\*", "[^/]*")
        assert any(re.fullmatch(f"(.*/)?{pattern}(/.*)?", f) for f in files), f"{doc} names {name}"


@pytest.mark.parametrize("doc", DOCS)
def test_every_relative_link_resolves(doc: str) -> None:
    for target in LINK.findall(read(doc)):
        assert (ROOT / target).exists(), f"{doc} links to {target}"


def test_the_version_is_the_changelogs_latest_release() -> None:
    version = tomllib.loads(read("pyproject.toml"))["project"]["version"]
    changelog = read("CHANGELOG.md")
    latest = RELEASE.findall(changelog)[0]
    released = tuple(int(n) for n in latest.split("."))
    if version.endswith(".dev0"):  # the next release in progress, so above the latest
        assert tuple(int(n) for n in version.removesuffix(".dev0").split(".")) > released
    else:
        assert version == latest, f"pyproject says {version}, the changelog {latest}"
    assert f"[{latest}]: " in changelog, "the release heading needs its link"


def test_the_figures_the_pages_quote_are_the_models() -> None:
    model = json.loads(train.model_path(train.read_scope(train.DEFAULT_SCOPE)).read_text())
    held_out = model["scores"]["held_out"]
    rows, within = held_out["rows"], held_out["median_abs_pct_error"]
    quoted = f"Tested on {rows:,} unseen listings: half of estimates fall within {within:.1%}"
    for page in ("index.html", "client/app.html"):
        assert read(page).count(quoted) == 2, page  # the description, then og:description


def test_the_page_stays_within_its_budget() -> None:
    files = [*PAGE, *(ROOT / "model").rglob("*.js")]
    size = sum(len(gzip.compress(f.read_bytes())) for f in files)
    assert size <= PAGE_BUDGET_GZIP, f"{size} bytes gzipped, over {PAGE_BUDGET_GZIP}"


def luminance(color: str) -> float:
    """Relative luminance of #rrggbb, as WCAG 2.2 defines it."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def tokens() -> dict[str, tuple[str, str]]:
    """Each color token as its (light, dark) pair; a single color serves both themes."""
    css = read("client/app.css")
    found = {name: (color, color) for name, color in SINGLE.findall(css)}
    for name, light, dark in PAIR.findall(css):
        found[name] = (light, dark)
    return found


@pytest.mark.parametrize(("fg", "bg", "least"), CONTRAST, ids=[f"{f}-{b}" for f, b, _ in CONTRAST])
def test_colors_meet_wcag_aa_in_both_themes(fg: str, bg: str, least: float) -> None:
    colors = tokens()
    for theme, name in enumerate(("light", "dark")):
        got = ratio(colors[fg][theme], colors[bg][theme])
        assert got >= least, f"{fg} on {bg}, {name}: {got:.2f}:1"
