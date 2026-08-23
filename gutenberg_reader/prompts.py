"""All LLM prompt templates for the gutenberg-reader pipeline.

Segmentation into narration/dialogue is deterministic (see segmenter.py);
the LLM is only asked to attribute speakers to already-extracted dialogue.
"""

from __future__ import annotations


def _render_segment_lines(
    segments: list[dict],
    start_index: int = 0,
    flagged: set[int] | None = None,
    context_count: int = 0,
    flag_label: str = "[NEEDS SPEAKER]",
) -> str:
    """Render segments as numbered lines for attribution/critic prompts.

    Indices shown are absolute (start_index + position). The first
    context_count lines are marked [CONTEXT] (already attributed, read-only).
    """
    from gutenberg_reader.segmenter import QUOTE_CONTINUES

    lines = []
    prev_para = None
    for pos, seg in enumerate(segments):
        idx = start_index + pos
        para = seg.get("para")
        if para is not None and prev_para is not None and para != prev_para:
            lines.append("¶")
        prev_para = para
        kind = seg.get("type", "narration").upper()
        text = seg.get("text", "")
        if len(text) > 300:
            text = text[:300] + "…"
        tags = []
        if pos < context_count:
            tags.append("[CONTEXT]")
        if flagged and idx in flagged:
            tags.append(flag_label)
        speaker = seg.get("speaker")
        spk = f" speaker={speaker}" if kind == "DIALOGUE" and speaker else ""
        cont = " [SPEECH CONTINUES INTO NEXT SEGMENT]" if seg.get("notes") == QUOTE_CONTINUES else ""
        tag_str = (" " + " ".join(tags)) if tags else ""
        lines.append(f"{idx}.{tag_str} [{kind}]{spk} | {text}{cont}")
    return "\n".join(lines)


def tag_resolution_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    return f"""You are a specialist in literary text analysis.

KNOWN CHARACTERS:
{char_list}

You will receive a numbered list of segments from a novel: narration and dialogue,
in original order. Some narration segments are speech-attribution tags that refer
to the speaker indirectly — e.g. "said his lady", "cried his wife", "returned she",
"replied her mother" — and are marked [WHO IS THIS?].

Your job: for each marked tag, resolve WHO the referring expression denotes.
"his lady" / "his wife" spoken of Mr. Bennet means Mrs. Bennet. "she" refers to
the most recently established female speaker. Use the surrounding narration and
dialogue to resolve pronouns and role references to actual character names.

Respond ONLY with JSON:
{{"attributions": [{{"index": <segment number>, "speaker": "<Character Name>"}}]}}

Include exactly one entry per [WHO IS THIS?] segment: the character the tag
refers to (the person doing the saying/replying/crying). Names must match the
KNOWN CHARACTERS list exactly, or be "Unknown" if unresolvable.
"""


def tag_resolution_user(
    segments: list[dict],
    start_index: int,
    flagged: set[int],
    context_count: int,
) -> str:
    listing = _render_segment_lines(
        segments, start_index, flagged, context_count, flag_label="[WHO IS THIS?]"
    )
    return f"Resolve the [WHO IS THIS?] attribution tags to character names:\n\n{listing}"


def attribution_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    return f"""You are a specialist in literary speaker attribution for audiobook production.

KNOWN CHARACTERS:
{char_list}

You will receive a numbered list of segments from a novel: narration and dialogue,
in original order. Some dialogue segments are marked [NEEDS SPEAKER].
Your job: determine which character speaks each [NEEDS SPEAKER] dialogue segment.

Method, in priority order:
1. Attribution tags in adjacent narration ("said Mr. Bennet", "replied his wife",
   "cried Elizabeth") are hard evidence. Note that a possessive or descriptive
   reference ("his wife", "her mother") must be resolved to the actual character name.
2. In a two-person exchange, speakers strictly alternate between speech turns.
   A turn interrupted only by an attribution tag ("..." said she "...") or a
   [SPEECH CONTINUES INTO NEXT SEGMENT] marker is ONE turn, not two.
3. Vocatives identify the LISTENER, not the speaker: in "My dear Mr. Bennet, have
   you heard...", Mr. Bennet is being spoken TO — someone else is speaking.
4. A character who is merely MENTIONED — in the dialogue itself or in nearby
   narration ("...saddened by the Signora's unexpected accent") — is NOT
   thereby the speaker. Attribute only to someone present and speaking in
   the scene.
5. Lines containing only ¶ mark paragraph breaks in the original text.
   Narration and a quote in the SAME paragraph usually share their subject:
   in "Lucy felt that she had been selfish. ‘Charlotte, you mustn't spoil
   me…’" the quote is Lucy's. A paragraph break often — not always — means
   the speaker changes.
6. Use content clues: who knows this information, whose manner of speech is this,
   who was asked the preceding question.
7. If genuinely ambiguous (3+ possible speakers, no anchor), use "Unknown".
   "Unknown" is better than a confident wrong guess.

Dialogue segments already labeled with a speaker, and [CONTEXT] lines, are
established fact — use them as anchors; do not re-attribute them.

Respond ONLY with JSON:
{{"attributions": [{{"index": <segment number>, "speaker": "<Character Name>"}}]}}

Include exactly one entry per [NEEDS SPEAKER] segment. Speaker names must match
the KNOWN CHARACTERS list exactly, or be "Unknown".
"""


def attribution_user(
    segments: list[dict],
    start_index: int,
    flagged: set[int],
    context_count: int,
) -> str:
    listing = _render_segment_lines(segments, start_index, flagged, context_count)
    return f"Attribute the [NEEDS SPEAKER] dialogue segments:\n\n{listing}"


def verify_attribution_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    return f"""You are a critical reviewer of speaker attribution for audiobook production.

KNOWN CHARACTERS:
{char_list}

You will receive a numbered list of segments from a novel: narration and dialogue,
in original order. Dialogue marked [VERIFY] was attributed by an earlier pass,
but those answers are hidden from you so that your judgment is independent.
Derive each speaker from scratch; your answers are compared against the other
pass and disagreements go to an arbiter.

For each [VERIFY] segment:
1. Adjacent attribution tags in narration ("said Lucy") are decisive.
2. Vocatives name the LISTENER: in "Charlotte, don't you feel...", Charlotte
   is being spoken TO — the speaker is someone else.
3. A character merely mentioned in dialogue or nearby narration is NOT thereby
   the speaker. Only someone present and speaking in the scene qualifies.
4. Lines containing only ¶ mark paragraph breaks in the original text.
   Narration and a quote in the SAME paragraph usually share their subject;
   a paragraph break often — not always — means the speaker changes.
5. Track the conversation turn by turn: who was asked the question, who would
   know this, whose manner of speech is this.
6. If the evidence is insufficient or contradictory, answer "Unknown" — a
   guess that cannot be defended from the text is worse than "Unknown".

Dialogue segments NOT marked [VERIFY] but showing speaker= were anchored by
explicit attribution tags in the text and are established fact.

Respond ONLY with JSON:
{{"attributions": [{{"index": <segment number>, "speaker": "<Character Name>"}}]}}

Include exactly one entry per [VERIFY] segment. Speaker names must match the
KNOWN CHARACTERS list exactly, or be "Unknown".
"""


def verify_attribution_user(
    segments: list[dict],
    start_index: int,
    flagged: set[int],
    context_count: int,
) -> str:
    # Hide the first pass's answers: an independent second opinion catches
    # errors that a reviewer shown the proposal tends to rubber-stamp.
    display = [
        {**seg, "speaker": None} if start_index + pos in flagged else seg
        for pos, seg in enumerate(segments)
    ]
    listing = _render_segment_lines(
        display, start_index, flagged, context_count, flag_label="[VERIFY]"
    )
    return f"Independently attribute the [VERIFY] dialogue segments:\n\n{listing}"


def tiebreak_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    return f"""You are the final arbiter of disputed speaker attributions for audiobook production.

KNOWN CHARACTERS:
{char_list}

Two independent passes disagreed about who speaks the dialogue segments marked
[DISPUTED]. The user message lists both candidates for each. Weigh the textual
evidence — adjacent attribution tags, conversation flow, vocatives (which name
the listener, not the speaker), who was asked the preceding question, and
paragraph breaks (lines containing only ¶; narration and a quote in the same
paragraph usually share their subject) — and decide. Usually one candidate is right; if both are clearly wrong, answer with
the correct character instead; if the text truly does not say, "Unknown".

Respond ONLY with JSON:
{{"attributions": [{{"index": <segment number>, "speaker": "<Character Name>"}}]}}

Include exactly one entry per [DISPUTED] segment. Speaker names must match the
KNOWN CHARACTERS list exactly, or be "Unknown".
"""


def tiebreak_user(
    segments: list[dict],
    start_index: int,
    flagged: set[int],
    context_count: int,
    candidates: dict[int, tuple[str, str]],
) -> str:
    # Hide the disputed assignments so neither candidate looks like the default.
    display = [
        {**seg, "speaker": None} if start_index + pos in flagged else seg
        for pos, seg in enumerate(segments)
    ]
    listing = _render_segment_lines(
        display, start_index, flagged, context_count, flag_label="[DISPUTED]"
    )
    disputes = "\n".join(
        f"Segment {idx}: one pass said {a!r}, the other said {b!r}"
        for idx, (a, b) in sorted(candidates.items())
        if idx in flagged
    )
    return f"Resolve the [DISPUTED] attributions:\n\n{listing}\n\nDisputes:\n{disputes}"


def character_discovery_system() -> str:
    return """You are a literary analyst. Your task is to identify all named characters in the provided text.

For each character, provide:
- Their canonical full name (most formal version used in the text)
- Any aliases or shortened names used (e.g., "Lizzy" for "Elizabeth Bennet")
- Any pronunciation hints for unusual names
- The chapter number where they first appear

Respond with valid JSON:
{
  "characters": [
    {
      "name": "Full Name",
      "aliases": ["nickname", "alias"],
      "pronunciation_hints": [],
      "first_appearance_chapter": 1
    }
  ]
}

Be thorough. Include all named characters, even minor ones. Use the most formal version of each name as the canonical name.

CRITICAL: each real person must appear EXACTLY ONCE. Novels refer to the same
character many ways — "Lucy", "Miss Honeychurch", and "Lucy Honeychurch" are one
person: one entry, fullest name as canonical, every other form as an alias.
Given names and courtesy names pair up the same way ("Charlotte" is "Miss
Bartlett"). Before answering, re-check the list: if a bare first name or a
title+surname could be the same person as a fuller entry, merge them.

Equally CRITICAL: never merge DIFFERENT people who share a surname. A father
and son ("Mr. Emerson" and "George Emerson"), a mother and daughter
("Mrs. Honeychurch" and "Lucy Honeychurch"), and siblings are separate
entries. Merge only when the text shows the names refer to one person.

A first-person narrator is a character like any other and must be listed
under their NAME, never under a role or a pronoun. If the book names its
teller anywhere — "Call me Ishmael", a companion addressing them as Watson —
use that name. Never emit "the narrator", "the author", "I", or "myself" as
a character: they are not names, and the person behind them already has an
entry that these would compete with.
"""


def character_discovery_user(chapters_text: str) -> str:
    # Deliberately not shown the roster built by earlier chapters. Listing it
    # and asking for only the new names was tried both ways and lost either
    # way: the model echoed the whole list back (a 41-window book spent most of
    # its time regenerating names it had), and when told firmly enough to obey,
    # it under-reported instead — Pip, Bulkington and the "Midnight, Forecastle"
    # sailors all vanished from PG 2701. Each chapter reports what it sees, and
    # merge_rosters/merge_duplicate_characters reconcile the names.
    return f"Identify all characters in this text:\n\n{chapters_text}"


def critic_system(characters: list[str], new_characters: list[str] | None = None) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    roster_section = ""
    roster_json = ""
    if new_characters:
        new_list = "\n".join(f"  - {c}" for c in new_characters)
        roster_section = f"""
Separately, these names were just added to the character roster based on this
chapter. Character discovery over-collects: it lists ships, gods, cited
authors, historical figures from digressions, and duplicate spellings of
people already on the KNOWN CHARACTERS list. Anything that gets onto the
roster becomes a candidate speaker for the whole rest of the book, so review
each NEW name:
{new_list}

Report an issue only for names that should NOT stand as new characters:
- verdict "not_a_character": not a person in the story (a ship, an animal, a
  place, an author cited by the narrator, a person merely quoted or discussed
  who never appears). When in doubt, let the name stand — report nothing.
- verdict "duplicate": the same person as an existing KNOWN character; give
  that character as "canonical".

For "duplicate", weigh who the name REFERS TO, not how it is spelled — a
description and a name share no words but can be one person. A role or
relationship standing in for someone already listed ("the doctor", "his
wife", "the young stranger") is a duplicate of that character. Watch the
teller of the story especially: in a first-person book the narrator is one
of the named characters, so an entry describing them by role belongs to the
name their companions use.
"""
        roster_json = """,
  "roster_issues": [
    {"name": "<New Name>", "verdict": "not_a_character" | "duplicate", "canonical": "<Known Character or empty>", "reason": "<brief>"}
  ]"""
    return f"""You are a quality reviewer for audiobook speaker attribution.

KNOWN CHARACTERS:
{char_list}

You will receive a run of a chapter's segments, numbered, with their assigned
speakers. Long chapters arrive in several such runs, so the numbering starts
wherever the run starts — always answer with the numbers as shown. Lines marked
[CONTEXT] carry over from the preceding run for continuity: read them, but do
not correct them.

Review ONLY the speaker assignments of dialogue segments. Look for:
1. Dialogue attributed to the wrong character (contradicted by an adjacent
   "said X" attribution tag in narration)
2. Broken alternation in two-person exchanges (same speaker on consecutive
   turns with no indication of a continued speech)
3. A speaker who is being addressed in the dialogue itself (a vocative names
   the listener, so the speaker must be someone else)
{roster_section}
Respond ONLY with JSON:
{{
  "corrections": [
    {{"index": <segment number>, "speaker": "<Correct Character Name>", "reason": "<brief>"}}
  ],
  "overall_quality": <0.0-1.0 fraction of dialogue segments correctly attributed>{roster_json}
}}

Only include entries for dialogue segments whose speaker should CHANGE.
If everything is correct, return an empty corrections list and quality 1.0.
"""


def critic_user(
    chapter_title: str,
    segments: list[dict],
    start_index: int = 0,
    context_count: int = 0,
) -> str:
    listing = _render_segment_lines(segments, start_index, context_count=context_count)
    return f"Review speaker attribution for {chapter_title}:\n\n{listing}"


def llm_chapter_discovery_system() -> str:
    return """You are a literary text analyst. Your task is to identify chapter boundaries in a book's text.

Look for chapter headings like:
- CHAPTER I, CHAPTER 1, Chapter One
- PART I, BOOK I
- Any other structural division markers

Respond with valid JSON:
{
  "chapters": [
    {
      "number": 1,
      "title": "CHAPTER I",
      "start_line": 10,
      "start_marker": "CHAPTER I"
    }
  ]
}

Line numbers are 1-indexed. Be thorough and find all chapter divisions.
"""


def llm_chapter_discovery_user(text: str) -> str:
    return f"Find all chapter boundaries in this text (provide 1-indexed line numbers):\n\n{text}"
