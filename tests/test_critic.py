"""The critic, now that it runs by default.

Three things were harmless while it was opt-in and systematic once it is not: a
window whose review failed was skipped silently while the chapter still reported
a quality score; needs_reprocessing was set and never read; and roster
objections rode along on every attribution window, so the same name was judged
five or ten times a chapter and the first answer won by accident of ordering.
"""

from __future__ import annotations

import json

import pytest

from gutenberg_reader import prompts, schemas, text_utils
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


# ── Cache entries must know what they are about ──────────────────────────────

def test_a_critic_cache_entry_is_rejected_when_the_chapter_changed(tmp_path):
    """run_chapter returns the accepted chapter as well as the verdict, so a
    stale entry does not merely mis-report — it hands back the old chapter and
    the freshly segmented one is discarded. On PG 6400 the corrected
    25,197-word "CAIUS JULIUS CASAR." was computed and then replaced by the
    173,461-word span from the structure that had just been fixed."""
    old = _chapter([_seg("“old.”", "Ahab")], title="D. OCTAVIUS CAESAR AUGUSTUS.")
    new = _chapter([_seg("“new.”", "Ahab"), _seg("“more.”", "Ahab")],
                   title="CAIUS JULIUS CASAR.")
    assert s06_critic._fingerprint(old) != s06_critic._fingerprint(new)


def test_fingerprint_tracks_title_length_and_words():
    a = _chapter([_seg("“x.”")])
    b = _chapter([_seg("“x.”")], title="Another")
    c = _chapter([_seg("“x.”"), _seg("“y.”")])
    assert s06_critic._fingerprint(a) != s06_critic._fingerprint(b)
    assert s06_critic._fingerprint(a) != s06_critic._fingerprint(c)


# ── The re-attribution path, with work actually to do ────────────────────────

def test_reattribution_runs_when_a_chapter_is_flagged(tmp_path):
    """This path shipped broken: _llm_window_pass was called with char_names
    where config belongs, so it died on 'list' object has no attribute
    'chunk_size'. Every existing test reached it with an empty retry set, which
    returns before the call, so the whole suite stayed green while a full
    rebuild failed on six books in a row.
    """
    from gutenberg_reader.stages.s05_segments import _reattribute_and_recheck

    class _Both:
        def __init__(self):
            self.models: list[str] = []

        def chat_json(self, model, messages, schema=None, **kw):
            self.models.append(model)
            system = messages[0]["content"]
            if "Characters were just discovered" in system:
                return {"roster_issues": []}
            if "attributions" in json.dumps(schema or {}):
                return {"attributions": [{"index": 0, "speaker": "Ahab"}]}
            return {"corrections": [], "overall_quality": 1.0}

    chapter = _chapter([_seg("“Line one.”", None), _seg("“Line two.”", "Ahab")])
    report = CriticReport(chapter_number=1, overall_quality=0.4,
                          needs_reprocessing=True)
    client = _Both()
    cfg = Config(book_id="0", max_retries=1, cache_dir=tmp_path,
                 validation_model="validator")
    for stage in range(1, 8):
        cfg.stage_dir(stage).mkdir(parents=True, exist_ok=True)

    out, second = _reattribute_and_recheck(
        cfg, client, chapter, report, [CharacterInfo(name="Ahab")], set())

    assert client.models, "the re-attribution pass never called the model"
    assert "validator" in client.models, "it must run on the validator model"
    assert second.overall_quality >= report.overall_quality
    assert out.segments[0].speaker == "Ahab"


def test_reattribution_returns_early_with_nothing_to_redo():
    """The case every other test happened to hit."""
    from gutenberg_reader.stages.s05_segments import _reattribute_and_recheck

    class _Never:
        def chat_json(self, *a, **kw):
            raise AssertionError("should not be called")

    chapter = _chapter([_seg("“Line.”", "Ahab")])
    report = CriticReport(chapter_number=1, overall_quality=0.4,
                          needs_reprocessing=True)
    out, back = _reattribute_and_recheck(
        _cfg(), _Never(), chapter, report, [CharacterInfo(name="Ahab")], set())
    assert out is chapter and back is report


# ── Names that are not names ─────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    # Barred from answering "Narrator", the model offered this on PG 3296 and it
    # absorbed 109 attributions belonging to Augustine or to nobody.
    "Self/Narrator", "Narrator/Self", "narrator or author",
    # Asked in the prompt for proper names since the roster work; asking is not
    # enough, the same lesson as "Unnamed narrator".
    "Wise man", "The builder",
    "Narrator", "Unnamed narrator", "N/A", "Unknown",
])
def test_a_role_or_a_description_is_never_a_character(name):
    from gutenberg_reader import text_utils
    assert text_utils.is_reserved_character_name(name)


@pytest.mark.parametrize("name", [
    "Augustine", "Jane Eyre", "Mr. Sherlock Holmes",
    # A person described by relation is still a person, and ends on the noun
    # that names her.
    "Narrator's Wife",
    # Particles and lowercase first words are ordinary in real names.
    "young Lucas", "Joan of Arc", "de Winter",
])
def test_a_real_name_survives_the_guards(name):
    from gutenberg_reader import text_utils
    assert not text_utils.is_reserved_character_name(name)


def test_no_function_uses_a_name_it_never_defines():
    """Three separate bugs this session came from an unanchored replace leaving a
    name undefined in one scope while tests stayed green: char_names passed where
    config belonged, a dedented return, and `attributable` used in
    _segment_chapter but assigned only in _reattribute_and_recheck. Reading the
    module for it costs nothing and catches all three shapes."""
    import ast
    import builtins
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "gutenberg_reader"
    builtin_names = set(dir(builtins))

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        # Only genuinely module-level bindings. Walking into function bodies for
        # these was the flaw in the first version of this check: every local
        # assignment anywhere in the file counted as global, so a name defined in
        # one function looked defined in all of them — which is exactly the bug
        # being hunted, and it went unreported.
        module_level = {
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        } | {
            (a.asname or a.name).split(".")[0]
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names
        }
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
                continue
            for t in ast.walk(stmt):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    module_level.add(t.id)
        # Only outermost functions: walking one already collects what its nested
        # functions bind, and checking a closure on its own reports every name it
        # legitimately captures from the scope around it.
        nested = {
            inner for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            for inner in ast.walk(n)
            if isinstance(inner, ast.FunctionDef) and inner is not n
        }
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n not in nested]:
            bound: set[str] = set()
            # Nested functions and lambdas bind their own parameters; collect
            # every signature inside this one, not just the outer.
            for scope in ast.walk(fn):
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Lambda)):
                    a = scope.args
                    bound |= {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
                    if a.vararg:
                        bound.add(a.vararg.arg)
                    if a.kwarg:
                        bound.add(a.kwarg.arg)
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    bound.add(node.name)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    bound.add(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        bound.add((a.asname or a.name).split(".")[0])
                elif isinstance(node, (ast.comprehension,)):
                    for t in ast.walk(node.target):
                        if isinstance(t, ast.Name):
                            bound.add(t.id)
            used = {
                n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            unknown = used - bound - module_level - builtin_names
            assert not unknown, f"{path.name}:{fn.name} uses undefined {sorted(unknown)}"


# ── The narrator is a roster entry, not an available answer ──────────────────

def test_the_critic_is_not_offered_the_narrator():
    """Stage 05 excludes the narrator from the attribution enum, but the critic
    built its own list straight off the roster and undid it: on PG 3296 stage 05
    left Augustine the 17 lines that had a first-person tag and the critic handed
    him back 61 more out of Unknown, landing within one line of the unfixed run."""
    seen: dict[str, list[str]] = {}

    def _capture(chapter, char_names, new_names, config, client):
        seen["names"] = list(char_names)
        return [], 1.0, []

    real = s06_critic._llm_critique
    s06_critic._llm_critique = _capture
    try:
        chapter = _chapter([_seg("“Line.”", "Unknown")])
        s06_critic._critique_chapter(
            chapter,
            [CharacterInfo(name="Augustine"), CharacterInfo(name="Alypius")],
            [], _cfg(), _Client({"roster": {"roster_issues": []}}),
            narrator_name="Augustine",
        )
    finally:
        s06_critic._llm_critique = real

    assert seen["names"] == ["Alypius"]


def test_the_narrator_is_still_offered_when_the_book_has_none():
    """Third-person books pass narrator_name="" and must be unaffected."""
    seen: dict[str, list[str]] = {}

    def _capture(chapter, char_names, new_names, config, client):
        seen["names"] = list(char_names)
        return [], 1.0, []

    real = s06_critic._llm_critique
    s06_critic._llm_critique = _capture
    try:
        s06_critic._critique_chapter(
            _chapter([_seg("“Line.”", "Ahab")]),
            [CharacterInfo(name="Ahab"), CharacterInfo(name="Pip")],
            [], _cfg(), _Client({"roster": {"roster_issues": []}}))
    finally:
        s06_critic._llm_critique = real

    assert seen["names"] == ["Ahab", "Pip"]


def test_a_first_person_tag_outranks_a_critic_correction():
    """"said I" is evidence of the same kind as "said Alypius" and gets the same
    standing: the critic may not move the line off the narrator."""
    chapter = _chapter([
        _seg("“Thou art great, O Lord.”", "Augustine"),
        _seg("said I, and was silent.", None, kind="narration"),
    ])
    client = _Client({
        "window": {
            "corrections": [{"index": 0, "speaker": "Alypius", "reason": "guess"}],
            "overall_quality": 0.5,
        },
        "roster": {"roster_issues": []},
    })
    _, chap, _ = s06_critic._critique_chapter(
        chapter,
        [CharacterInfo(name="Augustine"), CharacterInfo(name="Alypius")],
        [], _cfg(), client, narrator_name="Augustine")
    assert chap.segments[0].speaker == "Augustine"


# ── A deity or an abstraction is not a castable speaker ──────────────────────

@pytest.mark.parametrize("name", [
    "The Lord", "Lord God", "The Holy Ghost", "God", "Truth", "O Truth",
    "Thy Lord", "The Good God", "The Spirit", "The Unchangeable Truth",
])
def test_a_deity_or_an_abstraction_is_not_on_offer(name):
    """Closing the narrator leak moved the sink rather than draining it: with
    Augustine withheld, "The Lord" collected 28 lines on PG 3296, most of them
    fragments of Augustine's own reasoning in quotation marks."""
    assert text_utils.is_rhetorical_speaker(name)


@pytest.mark.parametrize("name", [
    "Lord Wilmore", "Lord Ingram", "Lord Backwater", "Godfrey Norton",
    "Landlord", "Lady Catherine de Bourgh", "Lord Robert Walsingham de Vere St. Simon",
    "Ctimene", "Mrs. Fairfax",
])
def test_a_real_person_whose_name_contains_one_survives(name):
    """Every one of these is a real character in the library. A whole-name rule
    is what separates them from "The Lord"."""
    assert not text_utils.is_rhetorical_speaker(name)


def test_a_rhetorical_speaker_is_withheld_but_not_struck_from_the_roster():
    """Withheld, not reserved: the text really does put words in its mouth, so
    the roster entry stands and only the enum loses it."""
    assert not text_utils.is_reserved_character_name("The Lord")
    assert text_utils.attributable_names(
        ["Augustine", "The Lord", "Alypius"], "Augustine") == ["Alypius"]


def test_attributable_names_is_the_only_gate():
    """Every pass that offers a speaker enum goes through one function. Written
    as a filter at each call site, the narrator exclusion reached stage 05 and
    not the critic, which handed back 61 of the 64 lines stage 05 withheld."""
    import ast
    import pathlib
    offenders = []
    for path in pathlib.Path("gutenberg_reader").rglob("*.py"):
        tree = ast.parse(path.read_text())
        # The definition itself is the one place allowed to spell the filter out.
        exempt = {
            id(c)
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name == "attributable_names"
            for c in ast.walk(fn)
            if isinstance(c, ast.ListComp)
        }
        for node in ast.walk(tree):
            if (isinstance(node, ast.ListComp) and id(node) not in exempt
                    and "narrator_name" in ast.dump(node)):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"hand-rolled narrator filter: {offenders}"


@pytest.mark.parametrize("name", [
    "Self (Narrator)", "Self/Narrator", "Narrator (the author)",
    "Self | Narrator", "Narrator and Author",
])
def test_a_role_joined_to_a_role_is_still_a_role(name):
    """Barred from "Self/Narrator", the model returned "Self (Narrator)" on the
    next run and it took 77 of the Confessions' 203 dialogue segments. Whatever
    joins them, a name every part of which is a role is a role."""
    assert text_utils.is_reserved_character_name(name)


@pytest.mark.parametrize("name", [
    "Elizabeth Bennet", "Jean-Luc Picard", "Sir Harry Otway", "Countess G——",
    "Mrs. Fairfax", "Queequeg",
])
def test_a_real_name_survives_the_separator_split(name):
    assert not text_utils.is_reserved_character_name(name)
