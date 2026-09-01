"""File already-finished books into library/.

Stage 07 installs on every complete run, so this is only needed to backfill books
processed before the library existed, or to re-file after changing the slug rules.

    python scripts/install_library.py [book_id ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutenberg_reader import library  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    ids = argv or sorted(p.name for p in (ROOT / "cache").iterdir() if p.is_dir())
    installed = 0
    for book_id in ids:
        # Only the whole-book output; "1342-ch1-2.json" is a fragment.
        final = ROOT / "cache" / book_id / "07-final" / f"{book_id}.json"
        if not final.exists():
            continue
        out = library.install(final, ROOT / "library")
        size = out.stat().st_size / 1_048_576
        title = json.loads(final.read_text())["metadata"]["title"]
        print(f"{out.name:<52} {size:>5.1f} MB  {title}")
        installed += 1
    print(f"\n{installed} book(s) in {ROOT / 'library'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
