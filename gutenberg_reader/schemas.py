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


def critic_schema(char_names: list[str]) -> dict:
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
