"""Stage 08 casting: production notes and voice specs on the final JSON."""

from __future__ import annotations

import json

from gutenberg_reader.config import Config
from gutenberg_reader.stages import s08_casting


def _cfg(tmp_path, **kw) -> Config:
    return Config(book_id="0", cache_dir=tmp_path, max_retries=1, **kw)


def _seg(text: str, speaker: str | None, seg_type: str = "dialogue") -> dict:
    return {"type": seg_type, "text": text, "speaker": speaker,
            "pronunciation_hints": [], "notes": None, "start": 0, "end": 1}


def _book(tmp_path, segments, characters):
    data = {
        "metadata": {"title": "Test Book", "author": "A. Writer",
                     "language": "English", "gutenberg_id": "0"},
        "chapters": [{
            "chapter": {"number": 1, "title": "Chapter I"},
            "processed": {"chapter_number": 1, "chapter_title": "Chapter I",
                          "segments": segments},
            "validation": None, "needs_review": False,
        }],
        "characters": [{"name": n, "aliases": [], "pronunciation_hints": [],
                        "first_appearance_chapter": 1} for n in characters],
        "statistics": {"total_chapters": 1},
        "processing_config": {},
    }
    path = tmp_path / "0.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


VOICE = {"sex": "female", "age_band": "adult",
         "accent": {"locale": "en-GB", "origin": "", "strength": "light"},
         "social_rank": "", "register": "educated", "pitch": "medium",
         "pace": "measured", "timbre": "warm", "distinctive": "steady"}

PRODUCTION = {
    "synopsis": "A test.", "casting_notes": "One accent.",
    "author": {"name": "A. Writer", "years": "", "nationality": "English",
               "note": ""},
    "narration": {"person": "third_omniscient", "narrator_character": "",
                  "voice": VOICE, "basis": "author_nationality"},
}


class _Client:
    """Answers each call by kind, or fails it."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def chat_json(self, model, messages, schema=None, **kw):
        system = messages[0]["content"]
        kind = "production" if "production notes" in system else "voices"
        self.calls.append(kind)
        answer = self.answers.get(kind)
        if answer is None:
            from gutenberg_reader.llm import LLMError
            raise LLMError("no answer configured")
        return answer


def _castings_for(names):
    return {"castings": [
        {"name": n, "description": f"{n} speaks.", "voice": dict(VOICE),
         "confidence": "high", "basis": "known_work"} for n in names]}


def test_speaking_characters_get_voices_and_silent_ones_do_not(tmp_path):
    path = _book(tmp_path,
                 [_seg("“Hello.”", "Ada"), _seg("She waved.", None, "narration")],
                 ["Ada", "Silent Sam"])
    client = _Client({"production": dict(PRODUCTION),
                      "voices": _castings_for(["Ada"])})
    s08_casting.run(_cfg(tmp_path), client, path)
    data = json.loads(path.read_text())

    assert data["production"]["narration"]["voice"]["accent"]["locale"] == "en-GB"
    ada = next(c for c in data["characters"] if c["name"] == "Ada")
    assert ada["voice"]["sex"] == "female"
    assert ada["dialogue_segments"] == 1
    sam = next(c for c in data["characters"] if c["name"] == "Silent Sam")
    assert "voice" not in sam
    assert sam["dialogue_segments"] == 0
    assert data["statistics"]["cast_characters"] == 1


def test_reserved_speakers_are_never_cast(tmp_path):
    path = _book(tmp_path,
                 [_seg("“Quoted verse.”", "Citation"),
                  _seg("“Who said this?”", "Unknown"),
                  _seg("“Mine.”", "Ada")],
                 ["Ada"])
    client = _Client({"production": dict(PRODUCTION),
                      "voices": _castings_for(["Ada"])})
    s08_casting.run(_cfg(tmp_path), client, path)
    data = json.loads(path.read_text())
    assert [c["name"] for c in data["characters"] if c.get("voice")] == ["Ada"]


def test_a_lost_casting_call_does_not_lose_the_book(tmp_path):
    """The book must install with or without its production notes: casting
    runs at the end of a job that can take hours."""
    path = _book(tmp_path, [_seg("“Hi.”", "Ada")], ["Ada"])
    client = _Client({})  # every call fails
    s08_casting.run(_cfg(tmp_path), client, path)
    data = json.loads(path.read_text())
    assert "production" not in data
    assert "voice" not in data["characters"][0]
    assert data["statistics"]["cast_characters"] == 0
    # dialogue counts are deterministic and survive regardless
    assert data["characters"][0]["dialogue_segments"] == 1


def test_a_cached_casting_is_reused_and_a_changed_roster_is_not(tmp_path):
    path = _book(tmp_path, [_seg("“Hi.”", "Ada")], ["Ada"])
    client = _Client({"production": dict(PRODUCTION),
                      "voices": _castings_for(["Ada"])})
    cfg = _cfg(tmp_path)
    s08_casting.run(cfg, client, path)
    first_calls = len(client.calls)

    # Stage 07 always reassembles, so the enrichment must come back from
    # cache without paying for the LLM again.
    path = _book(tmp_path, [_seg("“Hi.”", "Ada")], ["Ada"])
    s08_casting.run(cfg, client, path)
    assert len(client.calls) == first_calls
    assert "production" in json.loads(path.read_text())

    # More dialogue for Ada -> fingerprint differs -> recomputed.
    path = _book(tmp_path, [_seg("“Hi.”", "Ada"), _seg("“Again.”", "Ada")],
                 ["Ada"])
    s08_casting.run(cfg, client, path)
    assert len(client.calls) > first_calls


def test_a_third_person_book_cannot_link_a_narrator_character(tmp_path):
    """The schema lets the model pick any roster name; the pairing check is
    here because person and narrator_character are answered together and a
    third-person book has nobody telling it in their own voice."""
    prod = json.loads(json.dumps(PRODUCTION))
    prod["narration"]["narrator_character"] = "Ada"  # contradicts third_omniscient
    path = _book(tmp_path, [_seg("“Hi.”", "Ada")], ["Ada"])
    client = _Client({"production": prod, "voices": _castings_for(["Ada"])})
    s08_casting.run(_cfg(tmp_path), client, path)
    data = json.loads(path.read_text())
    assert data["production"]["narration"]["narrator_character"] == ""


def test_casting_decodes_greedily():
    """The installed library file must be byte-stable across identical
    analyses; a sampled casting would diff on every rebuild."""
    assert s08_casting.CASTING_TEMPERATURE == 0.0


def test_casting_schemas_exist():
    import gutenberg_reader.schemas as sch

    for factory, args in (
        (sch.voice_spec_schema, ()),
        (sch.production_schema, (["Ada"],)),
        (sch.voices_schema, (["Ada", "Sam"],)),
    ):
        assert isinstance(factory(*args), dict), factory.__name__
    batch = sch.voices_schema(["Ada", "Sam"])
    assert batch["properties"]["castings"]["minItems"] == 2
    assert batch["properties"]["castings"]["maxItems"] == 2
