"""JSON Schemas for structured (guided-decoding) LLM outputs.

Passed to vLLM as response_format json_schema — the server constrains
generation so responses always parse and validate. Where possible, speaker
names are constrained to an enum of known characters, making invented or
misspelled names impossible.
"""

from __future__ import annotations


def speaker_enum(char_names: list[str]) -> list[str]:
    names = list(dict.fromkeys(char_names))  # dedupe, keep order
    if "Unknown" not in names:
        names.append("Unknown")
    return names


def attribution_schema(char_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "attributions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "speaker": {"type": "string", "enum": speaker_enum(char_names)},
                    },
                    "required": ["index", "speaker"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["attributions"],
        "additionalProperties": False,
    }


def critic_schema(char_names: list[str], new_names: list[str] | None = None) -> dict:
    """Attribution review for one window. Speaker labels only.

    Roster objections used to ride along here, which meant every window was
    asked about the same new names and the first verdict won by accident of
    ordering. They have their own call now — see roster_review_schema.
    """
    return {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "speaker": {"type": "string", "enum": speaker_enum(char_names)},
                        "reason": {"type": "string"},
                    },
                    "required": ["index", "speaker", "reason"],
                    "additionalProperties": False,
                },
            },
            "overall_quality": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["corrections", "overall_quality"],
        "additionalProperties": False,
    }


def roster_review_schema(char_names: list[str], new_names: list[str]) -> dict:
    """One verdict per name discovery just added, asked once for the chapter.

    "name" is constrained to this chapter's additions — the rest of the roster is
    settled. "canonical" is the roster plus "", which is what a not_a_character
    verdict carries there.
    """
    return {
        "type": "object",
        "properties": {
            "roster_issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(dict.fromkeys(new_names))},
                        "verdict": {
                            "type": "string",
                            "enum": ["keep", "not_a_character", "duplicate"],
                        },
                        "canonical": {
                            "type": "string",
                            "enum": [*dict.fromkeys(char_names), ""],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "verdict", "canonical", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["roster_issues"],
        "additionalProperties": False,
    }


CHARACTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "pronunciation_hints": {"type": "array", "items": {"type": "string"}},
                    "first_appearance_chapter": {"type": "integer"},
                },
                "required": ["name", "aliases", "pronunciation_hints", "first_appearance_chapter"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["characters"],
    "additionalProperties": False,
}


CHAPTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "start_marker": {"type": "string"},
                },
                "required": ["number", "title", "start_line", "start_marker"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chapters"],
    "additionalProperties": False,
}


def structure_schema(n_candidates: int) -> dict:
    """Constrain a structure verdict to choices that exist.

    The model returns *ordinals into the candidate list*, never line numbers and
    never titles. A bounded integer makes a hallucinated position structurally
    impossible — the same trick the speaker enum plays on invented characters —
    and titles are then sliced from the candidate the ordinal names, so the model
    never has to reproduce text it might alter.
    """
    return {
        "type": "object",
        "properties": {
            "work_type": {
                "type": "string",
                "enum": ["novel", "story_collection", "history", "philosophy",
                         "essays", "letters", "memoir", "drama", "poetry", "other"],
            },
            "has_chapter_structure": {"type": "boolean"},
            "body_starts_before_first_heading": {"type": "boolean"},
            "headings": {
                "type": "array",
                # An empty array is a free escape hatch, and models take it.
                # Measured on PG 3296: unconstrained, DeepSeek reasoned its way to
                # "Blocks 5-17 are BOOK I through BOOK XIII ... a series of 13
                # books" and then returned []. With minItems it returns all 13.
                # A book with no chapter divisions still has a title page to
                # classify, so requiring one costs nothing.
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "ordinal": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(0, n_candidates - 1),
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["body", "front", "back", "title_page",
                                     "toc", "section_marker"],
                        },
                    },
                    "required": ["ordinal", "kind"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["work_type", "has_chapter_structure",
                     "body_starts_before_first_heading", "headings"],
        "additionalProperties": False,
    }


def span_type_schema(n_spans: int) -> dict:
    """Constrain a verdict on what each quoted span is.

    Ordinals again, never text: the model says which spans are speech, and code
    decides where the boundaries fall. A `term` verdict removes a boundary, so
    the surrounding narration re-forms with its punctuation intact — there is
    nothing to repair because nothing was taken apart.
    """
    return {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "ordinal": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(0, n_spans - 1),
                        },
                        "label": {
                            "type": "string",
                            "enum": ["speech", "term", "title", "citation",
                                     "rhetorical", "unsure"],
                        },
                    },
                    "required": ["ordinal", "label"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["spans"],
        "additionalProperties": False,
    }


def narration_schema() -> dict:
    """Who is telling the book, and under what name.

    The speaker enum is the roster, so a first-person narrator who is never named
    in their own text cannot be attributed at all: guided decoding has no token
    for them. PG 3296 is the case — the Confessions is addressed to God and
    Augustine never writes "Augustine said", so 18 of his own 26 first-person
    lines came back Unknown while Jane Eyre's 186 and Ishmael's 56 were all
    correct, because those two are named by other characters.
    """
    return {
        "type": "object",
        "properties": {
            "person": {
                "type": "string",
                "enum": ["first_person", "third_limited", "third_omniscient",
                         "epistolary", "mixed"],
            },
            # The name to file the narrator under, empty when nobody is telling
            # the story in their own voice. Never a role: "the narrator" and "I"
            # are rejected downstream.
            "narrator_name": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["person", "narrator_name", "confidence"],
        "additionalProperties": False,
    }


def voice_spec_schema() -> dict:
    """A castable voice, every closed attribute an enum.

    Downstream (tts-audiobook) scores these against a tagged library of
    reference clips: sex is a hard filter, age_band an ordered scale, the
    accent block drives locale/region matching, and the free-text fields are
    keyword tiebreakers and seed-speech direction. Free strings only where
    the vocabulary is genuinely open.
    """
    return {
        "type": "object",
        "properties": {
            "sex": {"type": "string", "enum": ["male", "female", "neutral"]},
            "age_band": {
                "type": "string",
                "enum": ["child", "teen", "young_adult", "adult",
                         "middle_aged", "elderly"],
            },
            "accent": {
                "type": "object",
                "properties": {
                    # BCP-47-style tag (en-GB, en-US, en-IE); the language the
                    # audiobook is read in, located, not the character's
                    # in-story mother tongue.
                    "locale": {"type": "string"},
                    # Regional origin within the locale (Hertfordshire,
                    # Yorkshire); "" when nothing more specific is known.
                    "origin": {"type": "string"},
                    "strength": {
                        "type": "string",
                        "enum": ["none", "light", "moderate", "strong"],
                    },
                },
                "required": ["locale", "origin", "strength"],
                "additionalProperties": False,
            },
            "social_rank": {"type": "string"},
            "register": {"type": "string"},
            "pitch": {"type": "string", "enum": ["low", "medium", "high"]},
            "pace": {
                "type": "string",
                "enum": ["slow", "measured", "medium", "brisk", "fast"],
            },
            "timbre": {"type": "string"},
            # Playable direction for a voice actor ("arch, teasing; lands the
            # last word"), not a physical description.
            "distinctive": {"type": "string"},
        },
        "required": ["sex", "age_band", "accent", "social_rank", "register",
                     "pitch", "pace", "timbre", "distinctive"],
        "additionalProperties": False,
    }


def production_schema(char_names: list[str]) -> dict:
    """Book-level production notes: synopsis, author, narration voice.

    narrator_character is an enum of the actual roster plus "" so a narrator
    who is also a character (Ishmael, Jane Eyre) can be linked to their
    roster entry, and an invented link is structurally impossible.
    """
    return {
        "type": "object",
        "properties": {
            "synopsis": {"type": "string"},
            "author": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "years": {"type": "string"},
                    "nationality": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["name", "years", "nationality", "note"],
                "additionalProperties": False,
            },
            "narration": {
                "type": "object",
                "properties": {
                    "person": {
                        "type": "string",
                        "enum": ["first_person", "third_limited",
                                 "third_omniscient", "epistolary", "mixed"],
                    },
                    "narrator_character": {
                        "type": "string",
                        "enum": [*dict.fromkeys(char_names), ""],
                    },
                    "voice": voice_spec_schema(),
                    "basis": {
                        "type": "string",
                        "enum": ["author_nationality", "narrator_character",
                                 "work_setting", "house_style"],
                    },
                },
                "required": ["person", "narrator_character", "voice", "basis"],
                "additionalProperties": False,
            },
            "casting_notes": {"type": "string"},
        },
        "required": ["synopsis", "author", "narration", "casting_notes"],
        "additionalProperties": False,
    }


def voices_schema(char_names: list[str]) -> dict:
    """Voice specs for one batch of characters.

    Exactly one casting per shown character: name is an enum of the batch and
    the array is pinned to its length, so a skipped or invented character
    cannot be encoded (duplicates are caught in Python).
    """
    names = [*dict.fromkeys(char_names)]
    return {
        "type": "object",
        "properties": {
            "castings": {
                "type": "array",
                "minItems": len(names),
                "maxItems": len(names),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "description": {"type": "string"},
                        "voice": voice_spec_schema(),
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "basis": {
                            "type": "string",
                            "enum": ["known_work", "inferred_from_text"],
                        },
                    },
                    "required": ["name", "description", "voice",
                                 "confidence", "basis"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["castings"],
        "additionalProperties": False,
    }
