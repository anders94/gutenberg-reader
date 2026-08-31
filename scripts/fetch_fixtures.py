"""Restore the books the regression tests are built on.

cache/ is gitignored, so the books vanish — and when cache/2131 vanished, its six
regression tests silently began skipping and the suite stayed green. The fixtures
themselves are committed; this fetches the books they describe, and verifies each
against the committed hash so a re-download that differs is caught rather than
quietly used.

    python scripts/fetch_fixtures.py            # all books in the manifest
    python scripts/fetch_fixtures.py 2131 6400  # just these
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutenberg_reader.config import Config  # noqa: E402
from gutenberg_reader.stages import s01_download  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "fixtures" / "manifest.json"


def main(argv: list[str]) -> int:
    expected = json.loads(MANIFEST.read_text())
    books = argv or sorted(expected)
    failures = 0

    for book_id in books:
        path = ROOT / "cache" / book_id / "01-raw" / "book.txt"
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest == expected.get(book_id):
                print(f"{book_id}: ok")
                continue
            print(f"{book_id}: present but differs from the manifest — re-fetching")

        config = Config(book_id=book_id, cache_dir=ROOT / "cache")
        for stage in range(1, 8):
            config.stage_dir(stage).mkdir(parents=True, exist_ok=True)
        try:
            s01_download.run(config)
        except Exception as e:
            print(f"{book_id}: download failed: {e}")
            failures += 1
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if book_id not in expected:
            print(f"{book_id}: fetched (not in manifest — add it with build_fixtures.py)")
        elif digest != expected[book_id]:
            print(
                f"{book_id}: FETCHED COPY DIFFERS from the manifest.\n"
                f"  expected {expected[book_id][:16]}...\n"
                f"  got      {digest[:16]}...\n"
                "  Project Gutenberg re-issued this text. Re-verify the golden "
                "before regenerating it — the boundaries may have moved."
            )
            failures += 1
        else:
            print(f"{book_id}: fetched and verified")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
