# Chapter Boundary Detection — Front Matter and Back Matter

> **Status: addressed.** This document describes an earlier layout
> (`src/gutenberg_reader/download/parser.py`); the defects carried over into the
> staged pipeline and were fixed there — back-matter terminal boundary and
> front-matter guard in `gutenberg_reader/stages/s02_discovery.py`, heading
> classification in `gutenberg_reader/text_utils.py`, size-outlier warning,
> LLM entry validation, and `--include-front-matter` / `--include-back-matter`
> CLI flags. Regression tests: `tests/test_structure.py` (PG 1727 fixture).

## What Went Wrong

The Odyssey (PG 1727) render produced two defects that had to be repaired by hand
downstream, after the TTS pass had already run:

1. **`25_book-xxiv.mp3` was 113 minutes** — five times any other chapter. It contained
   Book XXIV followed by the edition's *entire* footnote appendix (~87 minutes of
   "reading Phorcys for Sipho, Od. XIII.1" and Greek transliteration).
2. **Chapter 1 was the dedication and Butler's two prefaces, titled `FOOTNOTES:`** —
   the title came from one TOC entry, the content from an entirely different part of
   the book.

Neither was caught by anything in the pipeline. Both cost real TTS time: the footnote
appendix alone was 87 minutes of synthesis and 62 MB of MP3 that were then discarded.

## Reproduction

The back-matter swallow is deterministic and does **not** require the LLM. Both
non-LLM paths reproduce it on the current code:

```python
import sys; sys.path.insert(0, 'src')
from gutenberg_reader.download.parser import _parse_text, _parse_html

_parse_text(open('pg1727.txt').read())    # -> 24 chapters, last = 79,271 chars
_parse_html(open('pg1727.html').read())   # -> 24 chapters, last = 79,483 chars
```

Median of chapters 1–23 is **24,052 chars**. The last chapter is **79,271 — 3.3×
the median**. Book XXIV on its own is 27,486 chars (1.14× median), so roughly 52,000
characters of appendix are in there that should not be.

For the LLM path, a stub client that returns the TOC in document order parses the book
*correctly* (27 chapters: 2 prefaces, 24 books, footnotes as its own chapter). A stub
that returns the same entries with `FOOTNOTES:` **first** produces:

```
25 TOC entries -> 24 chapters      # one entry silently vanished
  BOOK XXIV: 79,483 chars          # and the last chapter absorbed it
```

That is the shipped failure mode.

## Root Causes

### 1. The last chapter has no terminal boundary

All three parse paths end the final chapter at end-of-document:

- `parser.py:371` — `end = matches[i + 1].start() if i + 1 < len(matches) else len(text)`
- `parser.py:208` — `chapter_end_pos = len(all_elements)`
- `parser.py:114-122` — the final flush in `_parse_html()`

So *everything* after the last chapter heading becomes part of that chapter: footnotes,
appendices, indexes, glossaries, errata, transcriber's notes. This is not specific to
the Odyssey — it is the default behavior for every book with back matter.
`_remove_gutenberg_wrapper()` (`parser.py:469`) strips only the PG boilerplate, which is
why the license text stays out but the edition's own apparatus does not.

**Suggested fix.** Recognize back-matter headings and use the first one after the final
chapter start as the terminal boundary:

```python
BACK_MATTER = re.compile(
    r'^\s*(FOOTNOTES?|ENDNOTES?|NOTES?|APPENDIX|APPENDICES|INDEX|GLOSSARY|'
    r'BIBLIOGRAPHY|ERRATA|COLOPHON|TRANSCRIBER.?S?\s+NOTES?)\b[:.]?\s*$',
    re.IGNORECASE | re.MULTILINE,
)
```

In `_split_by_matches()`, apply it to the final chapter's text; in the HTML paths, scan
forward from the last chapter's element index for the first heading that matches and use
that index as the end. Log what was trimmed and how much.

Verified against the real book — trimming the last chapter at the first `BACK_MATTER`
match gives:

```
before: 79,271 chars (3.30x median)
after : 27,485 chars (1.14x median)     # true Book XXIV is 27,486 — off by one char
tail  : "...and presently made a covenant of peace between the two contending parties."
```

and matches nothing in chapters 1–23, so there are no false positives to trade for it.

### 2. Front and back matter are eligible to become chapters at all

Prefaces and the footnote appendix are legitimate TOC entries, so the LLM TOC path
treats them as chapters — that is the "correct" 27-chapter parse above. Correct as
*structure*, wrong as *audiobook*: nobody wants 87 minutes of scholarly apparatus read
aloud, and it is the direct cause of both Odyssey defects.

**Suggested fix.** Classify each TOC entry as `front` / `body` / `back` and carry it on
`ChapterText` as a `kind` field. Default to emitting only `body`; expose
`--include-front-matter` / `--include-back-matter` for completists. Front-matter cues:
`PREFACE`, `INTRODUCTION`, `DEDICATION`, `FOREWORD`, `TRANSLATOR'S NOTE`, `CONTENTS`,
`ILLUSTRATIONS`. Back matter as above.

This one change would have prevented both Odyssey problems.

### 3. The LLM TOC path never validates match ordering

`parser.py:177-198` builds `chapter_headings` in **TOC order**, and the positions are
never sorted or checked for monotonicity. `parser.py:184-189` searches from the start of
the document for every entry and `break`s on the first fuzzy match, so an entry can bind
to a heading that sits *before* the previous chapter's heading.

Then `parser.py:203-227` derives each chapter's end from the **next TOC entry's**
position. When that position is smaller, `range(start + 1, end)` is empty, and the
`if chapter_content:` guard at `parser.py:220` silently drops the chapter — while the
neighbouring chapter's span balloons. That is exactly the 25-in / 24-out result above:
one chapter disappeared without a single warning.

**Suggested fixes:**

- Search with a **monotonic cursor** — only consider headings after the previously
  matched position.
- After matching, sort by position and assert strictly increasing; on violation, warn
  loudly with both titles rather than proceeding.
- At `parser.py:220`, warn when a located TOC entry yields no content. A silent `if` is
  how a missing chapter ships.

### 4. Fuzzy matching is substring-based

`parser.py:285` — `if pattern_no_period in text_no_period` — means `BOOK I` matches
`BOOK II`, `BOOK III`, `BOOK IX`, and `BOOK XIII`. On this book it happens to work only
because the correct heading comes first in document order. Prefer, in order: exact match
on the normalized string, then a word-boundary regex (`\bBOOK\s+I\b`), and only then
substring. A monotonic cursor (above) makes this considerably safer either way.

### 5. Nothing sanity-checks chapter sizes

A chapter 3.3× the median shipped without comment, and the problem was only discovered
after listening. A post-parse check is cheap and would have caught this before the TTS
pass:

```python
sizes = [len(c.text) for c in chapters]
med = statistics.median(sizes)
for c in chapters:
    ratio = len(c.text) / med
    if ratio > 2.5 or ratio < 0.2:
        console.print(f"[yellow]⚠ chapter {c.number} ({c.title!r}) is "
                      f"{ratio:.1f}× the median — {len(c.text):,} vs {med:,.0f} chars. "
                      f"Check for swallowed front/back matter.[/yellow]")
```

Worth running the same check on *titles*: a chapter whose title matches `BACK_MATTER`
almost certainly should not be in the render.

## Suggested Test Fixture

PG 1727 is a good regression fixture — it exercises front matter, back matter, and
`BOOK <roman>` numbering at once. Suggested assertions for `tests/test_structure.py`:

- 24 chapters from both `_parse_text` and `_parse_html`
- first chapter title is `BOOK I`, last is `BOOK XXIV`
- last chapter is within ~1.5× the median (currently 3.3×)
- no chapter text contains `FOOTNOTES`
- no chapter title matches the front/back-matter patterns

## Downstream Audit (tts-audiobook)

Independent of the parser, a post-render check would have flagged this in seconds. The
narration rate is remarkably stable — measured across the 23 good Odyssey chapters it is
**17.47 chars/sec with 1.6% stdev**. So `len(chapter_text) / duration_seconds` should
land near a book-wide constant, and any chapter that deviates badly indicates text that
was rendered but does not belong (footnotes and citation-dense material run ~9.9
chars/sec — a very clear signal).

This is how the bad chapter was actually diagnosed after the fact: silence detection was
useless (the render has no pause longer than 0.9s anywhere), but the rate calibration
predicted the Book XXIV / footnotes seam to within 4 seconds of its true position at
1569.8s.

## Files to Modify

1. **`src/gutenberg_reader/download/parser.py`**
   - Add `BACK_MATTER` / `FRONT_MATTER` patterns and a `_trim_back_matter()` helper
   - Terminal boundary for the last chapter in `_split_by_matches()`, `_parse_html()`,
     and `_parse_with_llm_toc()`
   - Monotonic cursor + ordering validation in `_parse_with_llm_toc()`
   - Warn instead of silently dropping at `parser.py:220`
   - Tighten `_fuzzy_match_chapter()` match precedence

2. **`src/gutenberg_reader/models.py`**
   - `kind: Literal["front", "body", "back"]` on `ChapterText` (default `"body"`)

3. **`src/gutenberg_reader/utils/validation.py`**
   - Chapter-size outlier check, run after parsing and before segmentation

4. **`src/gutenberg_reader/cli.py`**
   - `--include-front-matter` / `--include-back-matter` flags, both default off

5. **`tests/test_structure.py`** (new)
   - PG 1727 fixture per above

## Impact on Existing Renders

Worth re-checking any book whose last chapter is an outlier. Of the current
gutenberg.andrs.dev catalog, Pride and Prejudice (1342) and The Count of Monte Cristo
(1184) both end on a real final chapter and look clean; the Odyssey was repaired by hand
(footnote appendix cut losslessly at 1569.45s, prefaces track dropped, files renumbered
01–24). Those repairs live only in the published site's `books/` directory — re-running
gutenberg-reader on 1727 with the current code will reintroduce both defects.
