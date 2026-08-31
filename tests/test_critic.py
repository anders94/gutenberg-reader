"""The critic, now that it runs by default.

Three things were harmless while it was opt-in and systematic once it is not: a
window whose review failed was skipped silently while the chapter still reported
a quality score; needs_reprocessing was set and never read; and roster
objections rode along on every attribution window, so the same name was judged
five or ten times a chapter and the first answer won by accident of ordering.
"""

from __future__ import annotations

import pytest

from gutenberg_reader import prompts, schemas
from gutenberg_reader.config import Config
from gutenberg_reader.models import (
    CharacterInfo, CriticReport, ProcessedChapter, Segment,
)
from gutenberg_reader.stages import s06_critic


def _seg(text, speaker=None, kind="dialogue"):
    return Segment(type=kind, text=text, speaker=speaker, start=0, end=len(text))


def _chapter(segments, number=1, title="CHAPTER I."):
    return ProcessedChapter(chapter_number=number, chapter_title=title,
                            segments=segments, word_count=100)


class _Client:
    """Answers each call by name, or fails it."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list[str] = []

    def chat_json(self, model, messages, schema=None, **kw):
        system = messages[0]["content"]
        kind = "roster" if "Characters were just discovered" in system else "window"
        self.calls.append(kind)
        answer = self.answers.get(kind)
        if answer is None:
            from gutenberg_reader.llm import LLMError
            raise LLMError("no answer configured")
        return answer


def _cfg(**kw):
    return Config(book_id="0", max_retries=1, **kw)


# ── The critic runs by default now ───────────────────────────────────────────

def test_critic_is_on_unless_asked_otherwise():
    assert Config(book_id="0").critic is True
    assert Config(book_id="0", critic=False).critic is False


# ── (a) a score must not cover work nobody did ───────────────────────────────

def test_a_failed_window_is_recorded_not_silently_skipped():
    chapter = _chapter([_seg(f"“Line {i}.”", "Ahab") for i in range(6)])
    client = _Client({})           # every call fails
    corrections, quality, unreviewed = s06_critic._llm_critique(
        chapter, ["Ahab"], [], _cfg(), client)

    assert corrections == []
    assert unreviewed, "a window that failed every retry must be recorded"
    assert quality == 0.0, "a passing score here would describe work never done"


def test_a_partly_reviewed_chapter_asks_for_another_pass():
    report = CriticReport(chapter_number=1, overall_quality=1.0,
                          unreviewed_windows=[[0, 20]])
    report.needs_reprocessing = (
        report.overall_quality < s06_critic.QUALITY_THRESHOLD
        or bool(report.unreviewed_windows)
    )
    assert report.needs_reprocessing


def test_unreviewed_windows_survive_the_cache_round_trip():
    r = CriticReport(chapter_number=3, unreviewed_windows=[[0, 12], [40, 60]])
    assert CriticReport.from_dict(r.to_dict()).unreviewed_windows == [[0, 12], [40, 60]]


# ── (c) the roster is judged once, with evidence ─────────────────────────────

def test_roster_is_no_longer_asked_once_per_window():
    """It used to ride along on the attribution schema, so a ten-window chapter
    asked ten times and issues_by_name.setdefault kept whichever came first."""
    assert "roster_issues" not in schemas.critic_schema(["Ahab"], ["Pip"])["properties"]
    assert "roster_issues" in schemas.roster_review_schema(["Ahab"], ["Pip"])["properties"]


def test_roster_review_is_one_call_for_the_chapter():
    chapter = _chapter([_seg(f"“Line {i}.”", "Ahab") for i in range(20)])
    client = _Client({
        "window": {"corrections": [], "overall_quality": 1.0},
        "roster": {"roster_issues": [
            {"name": "Pequod", "verdict": "not_a_character",
             "canonical": "", "reason": "a ship"},
        ]},
    })
    s06_critic._critique_chapter(chapter, [CharacterInfo(name="Ahab")],
                                 ["Pequod"], _cfg(), client)
    assert client.calls.count("roster") == 1
    assert client.calls.count("window") >= 1


def test_keep_is_a_verdict_not_an_objection():
    """A name the critic is happy with must not travel on as an issue."""
    chapter = _chapter([_seg("“Line.”", "Ahab")])
    client = _Client({
        "window": {"corrections": [], "overall_quality": 1.0},
        "roster": {"roster_issues": [
            {"name": "Pip", "verdict": "keep", "canonical": "", "reason": "real"},
            {"name": "Pequod", "verdict": "not_a_character", "canonical": "",
             "reason": "a ship"},
        ]},
    })
    issues = s06_critic._review_roster(
        chapter, ["Ahab"], ["Pip", "Pequod"], _cfg(), client)
    assert [i["name"] for i in issues] == ["Pequod"]


def test_roster_review_shows_where_a_name_appears():
    """Evidence sliced by code, rather than asking the model to remember."""
    chapter = _chapter([
        _seg("The Pequod sailed at dawn with every sail set.", kind="narration"),
        _seg("“Hard down!” he cried.", "Ahab"),
    ])
    assert any("Pequod" in s for s in s06_critic._roster_evidence(chapter, "Pequod"))
    assert s06_critic._roster_evidence(chapter, "Nobody") == []


def test_roster_review_is_skipped_when_nothing_was_discovered():
    client = _Client({})
    assert s06_critic._review_roster(
        _chapter([_seg("x")]), ["Ahab"], [], _cfg(), client) == []
    assert client.calls == []


# ── Prompts say what they mean ───────────────────────────────────────────────

def test_roster_prompt_warns_against_striking_a_real_character():
    text = prompts.roster_review_system()
    assert "keep" in text.lower()
    assert "refers to" in text.lower() or "REFERS TO" in text


# ── Smoke: the schemas the stages actually reference ─────────────────────────

def test_every_schema_a_stage_references_exists():
    """A refactor deleted CHARACTERS_SCHEMA and the whole suite still passed —
    only a live run found it, at stage 05, after two minutes of work. Module
    constants have no call site to fail at import, so name them here."""
    import gutenberg_reader.schemas as sch

    for name in ("CHARACTERS_SCHEMA", "CHAPTERS_SCHEMA"):
        assert isinstance(getattr(sch, name), dict), name
    for factory, args in (
        (sch.attribution_schema, (["Ahab"],)),
        (sch.critic_schema, (["Ahab"],)),
        (sch.roster_review_schema, (["Ahab"], ["Pip"])),
        (sch.structure_schema, (10,)),
        (sch.span_type_schema, (4,)),
    ):
        assert isinstance(factory(*args), dict), factory.__name__


def test_a_score_with_nothing_applied_is_not_believed():
    """PG 1342 chapter 5 scored 0.0 with nothing applied: every objection was to
    a segment a deterministic "said Charlotte" tag had already named, and the tag
    outranks the critic. Run alone the same chapter scored 1.0. Flagging it would
    send a reader back to re-listen to something correct."""
    chapter = _chapter([_seg("“Line.”", "Ahab") for _ in range(4)])
    client = _Client({
        "window": {"corrections": [], "overall_quality": 0.0},
        "roster": {"roster_issues": []},
    })
    report, _, _ = s06_critic._critique_chapter(
        chapter, [CharacterInfo(name="Ahab")], [], _cfg(), client)
    assert report.overall_quality == 1.0
    assert not report.needs_reprocessing


def test_a_score_backed_by_an_applied_correction_stands():
    chapter = _chapter([_seg("“Line.”", "Ahab"), _seg("“Other.”", "Ahab")])
    client = _Client({
        "window": {
            "corrections": [{"index": 1, "speaker": "Pip", "reason": "tag"}],
            "overall_quality": 0.5,
        },
        "roster": {"roster_issues": []},
    })
    report, chap, _ = s06_critic._critique_chapter(
        chapter, [CharacterInfo(name="Ahab"), CharacterInfo(name="Pip")],
        [], _cfg(), client)
    assert report.attribution_issues        # it was applied
    assert report.overall_quality == 0.5    # so the score stands


def test_the_critic_decodes_greedily():
    """A review you cannot reproduce is a review you cannot act on: identical
    input scored 0.996 and 0.796 on consecutive runs at temperature 0.1."""
    assert s06_critic.CRITIC_TEMPERATURE == 0.0


def test_temperature_reaches_the_client():
    seen = {}

    class _T:
        def chat_json(self, model, messages, schema=None, temperature=None, **kw):
            seen["t"] = temperature
            return {"ok": True}

    from gutenberg_reader.llm import call_json_with_retries
    call_json_with_retries(_T(), "m", [], temperature=0.0)
    assert seen["t"] == 0.0
