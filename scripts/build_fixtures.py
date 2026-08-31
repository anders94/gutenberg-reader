"""Regenerate the committed test fixtures from the books under cache/.

Fixtures are *derived* from the books, never the books themselves: a golden file
records where the chapters are, and a candidate file records the condensed view a
structure pass is shown. Both are small and public-domain-derived, so they can be
committed — which matters, because cache/ is gitignored and vanishes. When
cache/2131 disappeared, all six of its regression tests silently began skipping
and nothing noticed.

    python scripts/build_fixtures.py [book_id ...]

Goldens for books whose detection is verified are generated from the detector.
6400's is hand-written from the book's own table of contents, because the
detector is wrong about it — that is the point of the fixture.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutenberg_reader import candidates, text_utils  # noqa: E402
from gutenberg_reader.stages.s02_discovery import (  # noqa: E402
    _build_chapter_infos, _drop_leading_front_matter,
    _maybe_prepend_chapter_one, _split_headless_body,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
FIXTURES = ROOT / "tests" / "fixtures"

# Books whose structure the regex detector gets right, verified by the
# regression tests in test_structure.py.
VERIFIED = ["1184", "1260", "1342", "1661", "1727", "2641", "2701", "3296", "37106"]

# Books whose golden is hand-written because the detector cannot produce it:
#   2131 — needs the LLM path to find where the body starts, so a regex-only
#          run splits from a mid-prose line (7 parts over 20,663 words instead
#          of 13 over 36,916).
#   6400 — the detector is simply wrong about it; that is what the fixture is for.
HAND_WRITTEN = ["2131", "6400"]


def body_of(book_id: str) -> tuple[list[str], str]:
    raw = (CACHE / book_id / "01-raw" / "book.txt").read_text(encoding="utf-8")
    lines = raw.splitlines()
    start, end = text_utils.find_body_bounds(lines)
    return lines[start:end], hashlib.sha256(raw.encode()).hexdigest()


def detect(body: list[str]) -> list:
    raw = text_utils.detect_chapters_regex(body)
    raw = _maybe_prepend_chapter_one(raw, body)
    raw = _drop_leading_front_matter(raw, body)
    raw = _split_headless_body(raw, body)
    return _build_chapter_infos(raw, body, 0)


def write_golden(book_id: str) -> None:
    body, sha = body_of(book_id)
    chapters = detect(body)
    golden = {
        "book_id": book_id,
        "sha256_book": sha,
        "body_lines": len(body),
        "source": "detector (verified by tests/test_structure.py)",
        "has_chapter_structure": True,
        "exact_count": len(chapters),
        "chapters": [
            {
                "line": ci.start_line - 1,        # 0-based, body-relative
                "title": ci.title,
                "kind": ci.kind,
                "words": ci.word_count,
                # A synthetic chapter has no heading line in the source; the
                # detector invented it because the book prints none.
                "synthetic": not ci.start_marker,
            }
            for ci in chapters
        ],
    }
    path = FIXTURES / "golden" / f"{book_id}.json"
    path.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
    print(f"  golden/{book_id}.json  {len(chapters)} chapters")


def write_candidates(book_id: str) -> None:
    body, _ = body_of(book_id)
    cands = candidates.extract(body)
    path = FIXTURES / "candidates" / f"{book_id}.txt"
    path.write_text(candidates.render(cands) + "\n")
    print(f"  candidates/{book_id}.txt  {len(cands)} candidates")


def write_manifest(book_ids: list[str]) -> None:
    manifest = {}
    for b in book_ids:
        p = CACHE / b / "01-raw" / "book.txt"
        if p.exists():
            manifest[b] = hashlib.sha256(p.read_bytes()).hexdigest()
    path = FIXTURES / "manifest.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(manifest)
    path.write_text(json.dumps(dict(sorted(existing.items())), indent=2) + "\n")
    print(f"  manifest.json  {len(existing)} books")


def main(argv: list[str]) -> int:
    books = argv or VERIFIED + HAND_WRITTEN
    for b in books:
        if not (CACHE / b / "01-raw" / "book.txt").exists():
            print(f"  skip {b}: not in cache/ — run scripts/fetch_fixtures.py {b}")
            continue
        print(f"{b}:")
        write_candidates(b)
        if b in VERIFIED:
            write_golden(b)
        else:
            print(f"  golden/{b}.json  hand-written, not regenerated")
            assert b in HAND_WRITTEN, f"{b} has no golden and is not hand-written"
    write_manifest(books)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
