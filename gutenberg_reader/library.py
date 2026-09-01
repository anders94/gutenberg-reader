"""Installing a finished book into library/.

cache/ is scratch: gitignored, rewritten by every run, safe to delete. library/
is the opposite — a finished book, committed, so that re-processing a title shows
up as a reviewable diff against the last time it was good.

That only works if an unchanged analysis produces an unchanged file, so the run's
own timing is dropped on the way in. Everything else is the work and stays.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Long enough to stay recognisable, short enough to read in a directory listing.
MAX_SLUG_CHARS = 60

# Fields that describe the run rather than the book. Keeping them would make
# every re-processing a diff even when nothing about the analysis changed, which
# is exactly the question the library exists to answer.
VOLATILE_STATISTICS = ("processing_time_seconds",)


def slug_for(title: str) -> str:
    """A filename-safe slug from a book's title.

    Subtitles are cut at the first colon or semicolon: "Moby Dick; Or, The Whale"
    is filed as moby-dick, which is both how the book is referred to and a
    directory listing that can be read at a glance.
    """
    head = re.split(r"[;:]", title, maxsplit=1)[0]
    normal = unicodedata.normalize("NFKD", head)
    ascii_only = normal.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    if len(slug) > MAX_SLUG_CHARS:
        slug = slug[:MAX_SLUG_CHARS].rsplit("-", 1)[0]
    return slug or "untitled"


def library_name(book_id: str, title: str) -> str:
    return f"{book_id}-{slug_for(title)}.json"


def install(final_path: Path, library_dir: Path) -> Path:
    """Copy a finished book into the library, minus the run's own timing."""
    data = json.loads(final_path.read_text(encoding="utf-8"))
    stats = data.get("statistics")
    if isinstance(stats, dict):
        for key in VOLATILE_STATISTICS:
            stats.pop(key, None)

    book_id = data.get("metadata", {}).get("gutenberg_id", final_path.stem)
    title = data.get("metadata", {}).get("title", "")
    library_dir.mkdir(parents=True, exist_ok=True)
    out = library_dir / library_name(book_id, title)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out
