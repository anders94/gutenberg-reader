"""Whole-book cast regularization (PG 37106: one person, several names)."""

from __future__ import annotations

from collections import Counter

from gutenberg_reader import cast, prompts, schemas
from gutenberg_reader.models import CharacterInfo


def _roster():
    return [
        CharacterInfo(name="Mr. Laurence", aliases=["Laurie", "Grandfather"], first_appearance_chapter=2),
        CharacterInfo(name="Theodore Laurence", aliases=["Laurie"], first_appearance_chapter=10),
        CharacterInfo(name="Meg March", aliases=["Meg", "Meggy"], first_appearance_chapter=1),
        CharacterInfo(name="Margaret March", aliases=["Meg", "Margaret"], first_appearance_chapter=2),
        CharacterInfo(name="Mother", aliases=["Marmee"], first_appearance_chapter=1),
        CharacterInfo(name="Marmee March", aliases=["Mrs. March"], first_appearance_chapter=1),
        CharacterInfo(name="Aunt March", aliases=["the old lady"], first_appearance_chapter=4),
        CharacterInfo(name="Hannah March", aliases=["Hannah"], first_appearance_chapter=1),
    ]


def test_conflicts_are_real_names_claimed_twice():
    conflicts = cast.alias_conflicts(_roster())
    assert conflicts["Laurie"] == ["Mr. Laurence", "Theodore Laurence"]
    assert conflicts["Meg"] == ["Meg March", "Margaret March"]
    assert "Grandfather" not in conflicts and "Mother" not in conflicts


def test_a_merge_needs_a_name_in_common():
    """The model folded Aunt March into Hannah March; they share only the
    family name."""
    roster = _roster()
    _, relabel, log, _ = cast.apply(roster, [{"name": "Aunt March", "canonical": "Hannah March"}], {})
    assert relabel == {} and any("no name in common" in l for l in log)


def test_people_who_talk_to_each_other_are_not_merged():
    """Grandfather into grandson, against an explicit instruction. Tagged
    lines by both, a few segments apart, twice, is two people."""
    conversing = Counter({frozenset(("Mr. Laurence", "Theodore Laurence")): 2})
    roster, relabel, log, _ = cast.apply(
        _roster(), [{"name": "Mr. Laurence", "canonical": "Theodore Laurence"}], {}, conversing)
    assert relabel == {} and any("talk to each other" in l for l in log)
    assert {c.name for c in roster} >= {"Mr. Laurence", "Theodore Laurence"}


def test_a_merge_carries_name_and_aliases_and_relabels():
    roster, relabel, _, _ = cast.apply(
        _roster(), [{"name": "Meg March", "canonical": "Margaret March"},
                    {"name": "Mother", "canonical": "Marmee March"}], {})
    by = {c.name: c for c in roster}
    assert "Meg March" not in by and "Mother" not in by
    assert {"Meg March", "Meggy", "Meg"} <= set(by["Margaret March"].aliases)
    assert "Marmee" in by["Marmee March"].aliases and "Mother" in by["Marmee March"].aliases
    assert relabel == {"Meg March": "Margaret March", "Mother": "Marmee March"}
    assert by["Margaret March"].first_appearance_chapter == 1


def test_a_merge_chain_resolves_and_a_cycle_is_ignored():
    roster, relabel, _, _ = cast.apply(
        _roster(), [{"name": "Meg March", "canonical": "Margaret March"},
                    {"name": "Margaret March", "canonical": "Meg March"}], {})
    # A cycle: nothing is folded.
    assert relabel == {} and {"Meg March", "Margaret March"} <= {c.name for c in roster}


def test_ownership_strips_the_alias_from_the_loser():
    roster, _, log, moved = cast.apply(_roster(), [], {"Laurie": "Theodore Laurence"})
    by = {c.name: c for c in roster}
    assert "Laurie" not in by["Mr. Laurence"].aliases and "Laurie" in by["Theodore Laurence"].aliases
    assert moved == [("Laurie", "Theodore Laurence", "Mr. Laurence")]


def test_conversing_pairs_count_only_tag_backed_lines():
    chapters = [{"processed": {"chapter_number": 21, "segments": [
        {"type": "dialogue", "speaker": "Mr. Laurence", "evidence": "tag"},
        {"type": "narration"},
        {"type": "dialogue", "speaker": "Theodore Laurence", "evidence": "tag"},
        {"type": "dialogue", "speaker": "Mr. Laurence", "evidence": "inferred"},
        {"type": "dialogue", "speaker": "Theodore Laurence", "evidence": "tag"},
    ]}}]
    pairs = cast.conversing_pairs(chapters)
    assert pairs[frozenset(("Mr. Laurence", "Theodore Laurence"))] == 1


def test_reanchor_moves_tag_backed_lines_to_the_alias_owner():
    roster, *_ = cast.apply(_roster(), [], {"Laurie": "Theodore Laurence"})
    chapters = [{"processed": {"chapter_number": 5, "segments": [
        {"type": "dialogue", "text": "“Better, thank you,”", "speaker": "Mr. Laurence", "evidence": "tag"},
        {"type": "narration", "text": "said Laurie."},
        {"type": "dialogue", "text": "“Nothing.”", "speaker": "Mr. Laurence", "evidence": "inferred"},
    ]}}]
    assert cast.reanchor(chapters, roster) == 1
    segs = chapters[0]["processed"]["segments"]
    assert segs[0]["speaker"] == "Theodore Laurence" and segs[2]["speaker"] == "Mr. Laurence"


def test_suspect_lines_are_the_losers_guesses_before_the_owner_existed():
    roster = _roster()
    moved = [("Laurie", "Theodore Laurence", "Mr. Laurence")]
    chapters = [
        {"processed": {"chapter_number": 5, "segments": [
            {"type": "dialogue", "speaker": "Mr. Laurence", "evidence": "inferred"},
            {"type": "dialogue", "speaker": "Mr. Laurence", "evidence": "tag"},
            {"type": "dialogue", "speaker": "Jo March", "evidence": "inferred"},
        ]}},
        {"processed": {"chapter_number": 21, "segments": [
            {"type": "dialogue", "speaker": "Mr. Laurence", "evidence": "inferred"},
        ]}},
    ]
    assert cast.suspect_lines(chapters, roster, moved) == {0: {0}}


def test_the_cast_review_prompt_and_schema_are_bounded_and_explicit():
    system = prompts.cast_review_system()
    for field in ("merges", "name", "reason", "canonical", "alias_owners"):
        assert f'"{field}"' in system
    schema = schemas.cast_review_schema(["A", "B", "C"], {"Laurie": ["A", "B"]})
    assert schema["properties"]["merges"]["maxItems"] == 3
    owners = schema["properties"]["alias_owners"]
    assert owners["properties"]["Laurie"]["enum"] == ["A", "B"] and owners["required"] == ["Laurie"]
    props = list(schema["properties"]["merges"]["items"]["properties"])
    assert props.index("reason") < props.index("canonical")
