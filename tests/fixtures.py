"""Loading committed fixtures, and refusing to pass when they are missing.

cache/ is gitignored. When cache/2131 disappeared, `pytest.skip` quietly removed
six regression tests and the suite still reported green — the fixture vanished
and so did the coverage. Under GUTENBERG_FIXTURES=required (set in CI) a missing
or stale book is a failure instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

GOLDEN_BOOKS = sorted(p.stem for p in (FIXTURES / "golden").glob("*.json"))


def _manifest() -> dict[str, str]:
    return json.loads((FIXTURES / "manifest.json").read_text())


def _unavailable(book_id: str, why: str):
    msg = f"cache/{book_id}: {why} — run `python scripts/fetch_fixtures.py {book_id}`"
    if os.environ.get("GUTENBERG_FIXTURES") == "required":
        pytest.fail(msg)
    pytest.skip(msg)


def book_text(book_id: str) -> str:
    """The raw book, verified against the committed hash."""
    path = CACHE / book_id / "01-raw" / "book.txt"
    if not path.exists():
        _unavailable(book_id, "not present")
    raw = path.read_bytes()
    expected = _manifest().get(book_id)
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        # A silently different edition would move every boundary in the golden.
        _unavailable(book_id, "differs from the manifest hash")
    return raw.decode("utf-8")


def body_lines(book_id: str) -> list[str]:
    from gutenberg_reader import text_utils
    lines = book_text(book_id).splitlines()
    start, end = text_utils.find_body_bounds(lines)
    return lines[start:end]


def golden(book_id: str) -> dict:
    path = FIXTURES / "golden" / f"{book_id}.json"
    if not path.exists():
        pytest.fail(f"no golden for {book_id} — run scripts/build_fixtures.py")
    return json.loads(path.read_text())


def candidate_fixture(book_id: str) -> str:
    return (FIXTURES / "candidates" / f"{book_id}.txt").read_text()
