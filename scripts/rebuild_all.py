"""Rebuild every cached book, smallest first, a few at a time.

Smallest first so a systemic failure shows up in minutes rather than after the
longest book. Books are independent — the rolling roster is sequential within a
book, not across them — so a small worker pool is safe and roughly halves the
wall clock. Each book gets its own log; the summary says what came out.

    python scripts/rebuild_all.py [--workers N] [book_id ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "cache" / "_rebuild-logs"

# Smallest first. Word counts from the committed goldens.
ORDER = ["2131", "2641", "1661", "3296", "1727", "1342",
         "1260", "37106", "2701", "6400", "1184"]

STRUCTURE_URL = "http://goldberry:8080"
VALIDATOR_URL = "http://goldberry:8080"


def rebuild(book_id: str) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{book_id}.log"
    started = time.time()
    cmd = [
        str(ROOT / ".venv" / "bin" / "gutenberg-reader"), book_id,
        "--structure-base-url", STRUCTURE_URL,
        "--validator-base-url", VALIDATOR_URL,
        "--verbose",
    ]
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    out = ROOT / "cache" / book_id / "07-final" / f"{book_id}.json"
    result = {"book": book_id, "rc": proc.returncode,
              "minutes": round(elapsed / 60, 1)}
    if out.exists():
        d = json.loads(out.read_text())
        st = d["statistics"]
        result.update(
            chapters=st["total_chapters"],
            words=st["total_words"],
            segments=st["total_segments"],
            characters=st["total_characters"],
            quality=round(st["discovery_confidence"]["avg_confidence"], 3),
            needs_review=st.get("chapters_needing_review", []),
            unreviewed=st.get("unreviewed_critic_windows", 0),
        )
    print(f"[{time.strftime('%H:%M')}] {book_id}: rc={result['rc']} "
          f"{result['minutes']}min "
          + (f"{result.get('chapters')} ch, {result.get('words', 0):,} w, "
             f"quality {result.get('quality')}" if out.exists() else "NO OUTPUT"),
          flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("books", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    books = args.books or ORDER

    print(f"rebuilding {len(books)} books, {args.workers} at a time", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(rebuild, books))

    print(f"\n=== done in {(time.time() - started) / 60:.0f} min ===", flush=True)
    print(f"{'PG':>6} {'rc':>3} {'min':>6} {'chapters':>9} {'words':>9} "
          f"{'segs':>7} {'chars':>6} {'qual':>6}  flags")
    for r in results:
        flags = []
        if r["rc"]:
            flags.append("FAILED")
        if r.get("needs_review"):
            flags.append(f"review={r['needs_review']}")
        if r.get("unreviewed"):
            flags.append(f"unreviewed={r['unreviewed']}")
        print(f"{r['book']:>6} {r['rc']:>3} {r['minutes']:>6} "
              f"{r.get('chapters', '-'):>9} {r.get('words', 0):>9,} "
              f"{r.get('segments', 0):>7,} {r.get('characters', 0):>6} "
              f"{r.get('quality', '-'):>6}  {' '.join(flags)}")

    (LOGS / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 1 if any(r["rc"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
