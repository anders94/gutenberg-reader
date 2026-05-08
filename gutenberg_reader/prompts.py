"""All LLM prompt templates for the gutenberg-reader pipeline."""

from __future__ import annotations


def segmentation_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none identified yet)"
    return f"""You are an expert literary annotator preparing audiobook segments.
Your task is to split a passage of text into narration and dialogue segments with speaker attribution.

KNOWN CHARACTERS:
{char_list}

RULES (follow in strict order):
1. Every word from the input must appear in exactly one segment — no additions, removals, or alterations to the text.
2. Use "narration" for all non-dialogue text (description, action, attribution tags like "said she", "he replied").
3. Use "dialogue" only for text inside quotation marks that is spoken aloud by a character.
4. Attribution tags ("said Mr. Bennet", "replied his wife") are "narration", NOT "dialogue". They must be their OWN separate narration segment, never bundled with the dialogue before or after them.
5. speaker must be null for all narration segments.
6. Speaker names must exactly match a name from the KNOWN CHARACTERS list above.
7. If the speaker is unknown or ambiguous, use "Unknown" with a note explaining why.
8. Preserve ALL punctuation exactly — including the opening and closing quotation marks of dialogue. A dialogue segment's text must start with \" and end with \".
9. Do NOT split a sentence across multiple segments. Each narration segment should be a complete sentence or clause — never end mid-sentence. If a narration sentence is long, include it entirely in one segment.
10. For back-and-forth dialogue with no explicit attribution tag, infer the speaker from the nearest "said X" or "replied X" narration and apply strict alternation. Use "Unknown" only if there are 3 or more potential speakers with no nearby attribution to resolve them.
11. Respond ONLY with valid JSON in this exact format:

{{
  "segments": [
    {{
      "type": "narration"|"dialogue",
      "text": "exact text here",
      "speaker": null|"Character Name",
      "pronunciation_hints": [],
      "notes": null|"explanation"
    }}
  ],
  "discovered_characters": [
    {{
      "name": "Character Name",
      "aliases": ["alias1"],
      "pronunciation_hints": [],
      "first_appearance_chapter": 1
    }}
  ]
}}

EXAMPLE A — given input: `said she, "I am quite well."`
CORRECT output segments:
  {{"type": "narration", "text": "said she,", "speaker": null, ...}}
  {{"type": "dialogue", "text": "\"I am quite well.\"", "speaker": "Character", ...}}
Note: the dialogue text starts with \" (escaped quote) and ends with \" — the quotation marks are PART of the text field.

EXAMPLE B — given input: `"My dear sir," said his wife, "have you heard the news?"`
CORRECT output segments (3 segments):
  {{"type": "dialogue", "text": "\"My dear sir,\"", "speaker": "Wife", ...}}
  {{"type": "narration", "text": "said his wife,", "speaker": null, ...}}
  {{"type": "dialogue", "text": "\"have you heard the news?\"", "speaker": "Wife", ...}}
WRONG — do NOT merge attribution into dialogue:
  {{"type": "dialogue", "text": "\"My dear sir,\" said his wife, \"have you heard the news?\"", ...}}
The attribution phrase "said his wife," is NARRATION even when it appears between two dialogue fragments.
Any phrase like "said X", "replied X", "answered X", "returned X", "cried X", "asked X" is ALWAYS narration.

EXAMPLE C — given input: `"But it is," returned she; "for I heard it from Mrs. Long."`
CORRECT output segments (3 segments):
  {{"type": "dialogue", "text": "\"But it is,\"", "speaker": "Speaker", ...}}
  {{"type": "narration", "text": "returned she;", "speaker": null, ...}}
  {{"type": "dialogue", "text": "\"for I heard it from Mrs. Long.\"", "speaker": "Speaker", ...}}
WRONG — do NOT bundle the attribution and the following dialogue together:
  {{"type": "narration", "text": "returned she; \"for I heard it from Mrs. Long.\"", ...}}
Any text inside quotation marks is ALWAYS a dialogue segment, even if it follows an attribution phrase on the same line.

CRITICAL: The concatenation of all segment "text" fields must equal the input text exactly (same characters, same order, same punctuation). Do not drop, add, or alter any character — including quotation marks.
"""


def segmentation_user(chunk_text: str, context_segments: list[dict] | None = None) -> str:
    parts = []
    if context_segments:
        context_str = "\n".join(
            f"[{s['type'].upper()}] {s.get('speaker', '') or ''}: {s['text'][:100]}..."
            if len(s['text']) > 100 else f"[{s['type'].upper()}] {s.get('speaker', '') or ''}: {s['text']}"
            for s in context_segments[-3:]
        )
        parts.append(f"CONTEXT (last segments from previous chunk, do NOT re-segment):\n{context_str}\n")
    parts.append(f"TEXT TO SEGMENT:\n{chunk_text}")
    return "\n".join(parts)


def segmentation_retry(chunk_text: str, issues: list[str], context_segments: list[dict] | None = None) -> str:
    issues_str = "\n".join(f"  - {issue}" for issue in issues)
    base = segmentation_user(chunk_text, context_segments)
    return (
        f"{base}\n\n"
        f"PREVIOUS ATTEMPT HAD INTEGRITY ISSUES — the following text was missing or altered:\n"
        f"{issues_str}\n\n"
        f"You MUST include every word of the input exactly. Fix the issues above."
    )


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
"""


def character_discovery_user(chapters_text: str) -> str:
    return f"Identify all characters in these chapters:\n\n{chapters_text}"


def critic_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    return f"""You are a quality reviewer for audiobook segment attribution.

KNOWN CHARACTERS:
{char_list}

Review the provided segments for:
1. Speaker attribution errors (wrong character assigned to dialogue)
2. Mismatched attribution tags ("said X" paired with wrong dialogue segment)
3. Back-and-forth dialogue pattern errors (dialogue wrongly attributed in rapid exchanges)
4. Character name inconsistencies (misspellings, wrong canonical form)

Respond with valid JSON:
{{
  "missing_text": [],
  "attribution_issues": ["description of issue"],
  "name_inconsistencies": ["description"],
  "overall_quality": 0.0-1.0,
  "needs_reprocessing": true|false,
  "fixed_segments": [
    {{
      "type": "narration"|"dialogue",
      "text": "exact text",
      "speaker": null|"Character Name",
      "pronunciation_hints": [],
      "notes": null|"explanation"
    }}
  ]
}}

If overall_quality >= 0.85 and no coverage gaps: set needs_reprocessing to false and return the segments as-is (with any minor fixes) in fixed_segments.
If below 0.85 or serious issues: set needs_reprocessing to true.
"""


def critic_user(chapter_title: str, segments: list[dict]) -> str:
    segments_str = "\n".join(
        f"{i+1}. [{s['type'].upper()}] speaker={s.get('speaker')} | {s['text'][:120]}"
        for i, s in enumerate(segments)
    )
    return f"Review segments for {chapter_title}:\n\n{segments_str}"


def attribution_review_system(characters: list[str]) -> str:
    char_list = "\n".join(f"  - {c}" for c in characters) if characters else "  (none)"
    return f"""You are a specialist in literary speaker attribution for audiobook production.

KNOWN CHARACTERS:
{char_list}

You will receive a window of segments from a chapter, with some dialogue turns marked [NEEDS REVIEW].
Your job: determine the correct speaker for each [NEEDS REVIEW] dialogue segment.

Method:
1. Find any "said X", "replied X", "cried X" narration segments — these are hard anchors.
2. Use strict alternation from the nearest anchor to fill in unanchored turns.
3. Use the dialogue content for further clues (addressing someone by name, topic continuity).
4. If genuinely ambiguous (3+ speakers possible, no anchor), use "Unknown".

Respond ONLY with valid JSON:
{{
  "corrections": [
    {{
      "segment_index": 0,
      "speaker": "Character Name",
      "reason": "brief explanation"
    }}
  ]
}}

Only include segments that need correction (the [NEEDS REVIEW] ones). Do not modify confirmed segments.
"""


def attribution_review_user(window_segments: list[dict], flagged_idxs: set[int]) -> str:
    lines = []
    for i, seg in enumerate(window_segments):
        tag = "[NEEDS REVIEW] " if i in flagged_idxs else ""
        sp = seg.get("speaker") or "null"
        txt = seg.get("text", "")[:120]
        lines.append(f"{i}. {tag}[{seg['type'].upper()}] speaker={sp} | {txt}")
    return "Review speaker attribution for flagged segments:\n\n" + "\n".join(lines)


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
