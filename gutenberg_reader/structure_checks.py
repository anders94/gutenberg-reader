"""Post-discovery structure checks that can fail a run.

Every structure defect this pipeline has shipped was *detectable* at discovery
time. PG 1727's footnote appendix made the last chapter 3.3x the median; PG 6400
put 173,461 words — 82% of the book — into one chapter. Both printed a warning,
or would have, and both shipped anyway, because nothing acted on a warning. The
Odyssey defect then cost 87 minutes of TTS synthesis that was thrown away.

So these return findings with a severity, and the caller exits non-zero on a
failure. A SystemExit costs a re-run; a bad structure costs an afternoon of
synthesis and a hand repair downstream.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from gutenberg_reader.candidates import Candidate
from gutenberg_reader.models import ChapterInfo

# A chapter holding only its own heading. Twenty words is generous; a table of
# contents mistaken for the body produces entries of two or three.
MIN_CHAPTER_WORDS = 20
DEGENERATE_SHARE = 0.2

# Two chapters have no useful median — each sits at it — so they are compared
# against each other. PG 2131 split 647 against 36,916 and no median could see it.
PAIR_HIGH = 10.0

SIZE_HIGH = 2.5
SIZE_LOW = 0.2

# Residue: candidates the detector left unexplained inside a single chapter.
# A real chapter contains no evenly spaced series of same-shaped headings; a
# missed division contains exactly that. Thresholds measured over the corpus —
# see test_structure_checks.py, which pins the zero-false-positive result.
RESIDUE_MIN_RUN = 5
RESIDUE_MIN_GAP = 200

# Shapes that are never a chapter heading however evenly they recur. A bare
# number is a page marker: PG 6400 prints "(479)", "(506)", "(524)" through the
# body, and a long chapter can hold five of them 200+ lines apart without
# anything being wrong. Excluding them costs nothing — 6400's real missed
# headings are all-caps names, not numerals.
RESIDUE_IGNORED_SHAPES = {"shape:9", "shape:R", "shape:p"}


@dataclass
class Finding:
    severity: str          # "warn" | "fail"
    code: str
    message: str
    evidence: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


def _tiling(chapters: list[ChapterInfo]) -> list[Finding]:
    """Chapters must be ordered and contiguous — a structural invariant.

    Ends are derived from the next chapter's start, so an out-of-order entry
    silently empties one chapter while its neighbour balloons (PG 1727's LLM
    path did exactly this: 25 entries in, 24 chapters out, no warning).
    """
    out: list[Finding] = []
    for a, b in zip(chapters, chapters[1:]):
        if b.start_line != a.end_line + 1:
            out.append(Finding(
                "fail", "tiling",
                f"chapter {b.number} starts at line {b.start_line:,} but chapter "
                f"{a.number} ends at {a.end_line:,} — the text between them belongs "
                "to no chapter",
                {"after": a.number, "before": b.number},
            ))
    for ci in chapters:
        if ci.end_line < ci.start_line:
            out.append(Finding(
                "fail", "tiling",
                f"chapter {ci.number} ({ci.title!r}) spans no lines",
                {"chapter": ci.number},
            ))
    return out


def _degenerate(chapters: list[ChapterInfo]) -> list[Finding]:
    """Most chapters holding nothing is a table of contents read as the body."""
    if not chapters:
        return [Finding("fail", "degenerate", "no chapters were detected")]
    empty = [ci for ci in chapters if ci.word_count < MIN_CHAPTER_WORDS]
    if len(empty) < DEGENERATE_SHARE * len(chapters):
        return []
    return [Finding(
        "fail", "degenerate",
        f"{len(empty)} of {len(chapters)} chapters hold fewer than "
        f"{MIN_CHAPTER_WORDS} words — detection probably matched a table of "
        "contents or an index rather than the body",
        {"chapters": [ci.number for ci in empty[:20]]},
    )]


def _pair_lopsided(chapters: list[ChapterInfo]) -> list[Finding]:
    if len(chapters) != 2:
        return []
    big, small = sorted(chapters, key=lambda ci: ci.word_count, reverse=True)
    if small.word_count and big.word_count / small.word_count < PAIR_HIGH:
        return []
    return [Finding(
        "fail", "pair_lopsided",
        f"the book split into 2 chapters of {big.word_count:,} and "
        f"{small.word_count:,} words — a split this lopsided usually means the "
        "source carries no chapter headings and detection matched its title page",
        {"chapters": [big.number, small.number]},
    )]


def _size_outliers(chapters: list[ChapterInfo]) -> list[Finding]:
    """Warn only: Moby-Dick's Cetology is genuinely 3.6x the median."""
    if len(chapters) < 3:
        return []
    median = statistics.median(ci.word_count for ci in chapters)
    if not median:
        return []
    out = []
    for ci in chapters:
        ratio = ci.word_count / median
        if ratio > SIZE_HIGH or ratio < SIZE_LOW:
            out.append(Finding(
                "warn", "size_outlier",
                f"chapter {ci.number} ({ci.title!r}) is {ratio:.1f}x the median "
                f"({ci.word_count:,} vs {median:,.0f} words)",
                {"chapter": ci.number, "ratio": round(ratio, 2)},
            ))
    return out


def _unexplained_structure(
    chapters: list[ChapterInfo],
    cands: list[Candidate],
    body_start: int,
) -> list[Finding]:
    """A series of same-shaped, widely spaced candidates *inside* one chapter.

    This is the check that catches PG 6400. Its twelve Caesars share a shape and
    sit ~1,200 lines apart; the detector chose two of them and swallowed the rest
    into a 173,461-word chapter. A genuine chapter — even a long one — does not
    contain an evenly spaced series of identically shaped headings.

    Only *unchosen* candidates count, so a correctly detected book has nothing
    left over and reports nothing.
    """
    if not chapters:
        return []
    chosen = {ci.start_line for ci in chapters}
    out: list[Finding] = []

    for ci in chapters:
        by_shape: dict[str, list[Candidate]] = defaultdict(list)
        for c in cands:
            abs_line = body_start + c.line + 1
            if not (ci.start_line < abs_line <= ci.end_line) or abs_line in chosen:
                continue
            shape = next((f for f in c.flags if f.startswith("shape:")), "shape:?")
            by_shape[shape].append(c)

        for shape, group in by_shape.items():
            if shape in RESIDUE_IGNORED_SHAPES or len(group) < RESIDUE_MIN_RUN:
                continue
            gaps = [b.line - a.line for a, b in zip(group, group[1:])]
            if gaps and min(gaps) >= RESIDUE_MIN_GAP:
                out.append(Finding(
                    "fail", "unexplained_structure",
                    f"chapter {ci.number} ({ci.title!r}, {ci.word_count:,} words) "
                    f"contains {len(group)} unclassified blocks sharing "
                    f"{shape}, spaced {min(gaps):,}-{max(gaps):,} lines apart — "
                    "these look like chapter headings that were missed",
                    {
                        "chapter": ci.number,
                        "ordinals": [c.ordinal for c in group],
                        "examples": [c.text for c in group[:4]],
                    },
                ))
    return out


def check(
    chapters: list[ChapterInfo],
    cands: list[Candidate] | None = None,
    body_start: int = 0,
) -> list[Finding]:
    """Every finding about a discovered structure, failures first."""
    findings = (
        _tiling(chapters)
        + _degenerate(chapters)
        + _pair_lopsided(chapters)
        + _unexplained_structure(chapters, cands or [], body_start)
        + _size_outliers(chapters)
    )
    return sorted(findings, key=lambda f: f.severity != "fail")
