"""Speaker attribution — tag resolution and character regularization.

Regression tests for the Room with a View (PG 2641) defects: "And a Cockney,
besides!" attributed to Signora because the tag "said Lucy, who had been
further saddened by the Signora's unexpected accent." was resolved by
longest-alias-anywhere instead of the name adjacent to the speech verb; and
the character roster carrying the same person under several names ('Lucy' /
'Lucy Honeychurch', 'Sir Harry' / 'Sir Harry Otway').
"""

from __future__ import annotations

from gutenberg_reader import text_utils
from gutenberg_reader.models import CharacterInfo


def _alias_map(*names_with_aliases: tuple[str, list[str]]) -> dict[str, str]:
    chars = [CharacterInfo(name=n, aliases=a) for n, a in names_with_aliases]
    return text_utils._build_alias_map(chars)


class TestTagSpeaker:
    def test_verb_adjacent_name_beats_longer_mention(self):
        # The PG 2641 bug: Signora is a longer alias but Lucy follows the verb.
        amap = _alias_map(("Lucy", []), ("Signora", []))
        tag = "said Lucy, who had been further saddened by the Signora’s unexpected accent."
        assert text_utils._tag_speaker(tag, amap) == "Lucy"

    def test_verb_adjacent_name_resolves_through_aliases(self):
        amap = _alias_map(("Lucy Honeychurch", ["Lucy"]), ("Signora", []))
        tag = "said Lucy, who had been further saddened by the Signora’s unexpected accent."
        assert text_utils._tag_speaker(tag, amap) == "Lucy Honeychurch"

    def test_subject_first_known_name(self):
        amap = _alias_map(("Miss Bartlett", []), ("Lucy", []))
        assert text_utils._tag_speaker("Miss Bartlett continued;", amap) == "Miss Bartlett"

    def test_subject_first_pronoun_not_taken_as_name(self):
        amap = _alias_map(("Lucy", []),)
        assert text_utils._tag_speaker("They said nothing more.", amap) is None

    def test_unknown_name_after_verb_still_anchors(self):
        # Discovery missed Lydia; the tag still names her deterministically.
        amap = _alias_map(("Mrs. Bennet", []),)
        assert text_utils._tag_speaker("cried Lydia,", amap) == "Lydia"

    def test_ambiguous_multi_mention_defers_to_llm(self):
        # No name adjacent to the verb, two characters mentioned: don't guess.
        amap = _alias_map(("Lucy", []), ("Charlotte", []))
        tag = "said the elder of them, glancing from Lucy to Charlotte."
        assert text_utils._tag_speaker(tag, amap) is None

    def test_single_mention_fallback_still_works(self):
        amap = _alias_map(("Signora", []),)
        assert text_utils._tag_speaker("said the Signora,", amap) == "Signora"


class TestAnchorExtraction:
    def test_backward_anchor_picks_verb_adjacent_name(self):
        segments = [
            {"type": "dialogue", "text": "“And a Cockney, besides!”",
             "speaker": None, "pronunciation_hints": [], "notes": None},
            {"type": "narration",
             "text": "said Lucy, who had been further saddened by the Signora’s unexpected accent.",
             "speaker": None, "pronunciation_hints": [], "notes": None},
        ]
        chars = [CharacterInfo(name="Lucy"), CharacterInfo(name="Signora")]
        anchors = text_utils.extract_attribution_anchors(segments, chars)
        assert anchors == {0: "Lucy"}


class TestMergeDuplicateCharacters:
    def test_name_matching_anothers_alias_merges(self):
        chars = [
            CharacterInfo(name="Lucy", aliases=["Lucia"], first_appearance_chapter=1),
            CharacterInfo(name="Lucy Honeychurch", aliases=["Lucy"], first_appearance_chapter=6),
        ]
        merged = text_utils.merge_duplicate_characters(chars)
        assert len(merged) == 1
        assert merged[0].name == "Lucy Honeychurch"
        assert "Lucia" in merged[0].aliases
        assert merged[0].first_appearance_chapter == 1

    def test_token_subset_merges(self):
        chars = [
            CharacterInfo(name="Sir Harry Otway", first_appearance_chapter=10),
            CharacterInfo(name="Sir Harry", first_appearance_chapter=8),
        ]
        merged = text_utils.merge_duplicate_characters(chars)
        assert len(merged) == 1
        assert merged[0].name == "Sir Harry Otway"
        assert merged[0].aliases == ["Sir Harry"]
        assert merged[0].first_appearance_chapter == 8

    def test_subset_of_alias_tokens_merges(self):
        # 'Charlotte' shares no tokens with 'Miss Bartlett', but the alias
        # 'Charlotte Bartlett' bridges them.
        chars = [
            CharacterInfo(name="Miss Bartlett", aliases=["Charlotte Bartlett"]),
            CharacterInfo(name="Charlotte"),
        ]
        merged = text_utils.merge_duplicate_characters(chars)
        assert len(merged) == 1
        assert merged[0].name == "Miss Bartlett"
        assert "Charlotte" in merged[0].aliases

    def test_differing_titles_never_merge(self):
        # Mrs. Honeychurch is Lucy's mother, not Lucy.
        chars = [
            CharacterInfo(name="Lucy Honeychurch"),
            CharacterInfo(name="Mrs. Honeychurch"),
        ]
        assert len(text_utils.merge_duplicate_characters(chars)) == 2

    def test_ambiguous_subset_merges_into_neither(self):
        chars = [
            CharacterInfo(name="John"),
            CharacterInfo(name="John Smith"),
            CharacterInfo(name="John Brown"),
        ]
        assert len(text_utils.merge_duplicate_characters(chars)) == 3

    def test_idempotent(self):
        chars = [
            CharacterInfo(name="Minnie"),
            CharacterInfo(name="Minnie Beebe"),
            CharacterInfo(name="Mr. Beebe"),
        ]
        once = text_utils.merge_duplicate_characters(chars)
        twice = text_utils.merge_duplicate_characters(once)
        assert [c.name for c in twice] == ["Minnie Beebe", "Mr. Beebe"]


class TestMergeRosters:
    def test_new_names_append(self):
        roster = [CharacterInfo(name="Ishmael", first_appearance_chapter=1)]
        found = [CharacterInfo(name="Queequeg", first_appearance_chapter=3)]
        merged = text_utils.merge_rosters(roster, found)
        assert [c.name for c in merged] == ["Ishmael", "Queequeg"]

    def test_same_name_contributes_aliases_keeps_first_appearance(self):
        roster = [CharacterInfo(name="Starbuck", first_appearance_chapter=20)]
        found = [
            CharacterInfo(name="Starbuck", aliases=["Mr. Starbuck"], first_appearance_chapter=36)
        ]
        merged = text_utils.merge_rosters(roster, found)
        assert len(merged) == 1
        assert merged[0].aliases == ["Mr. Starbuck"]
        assert merged[0].first_appearance_chapter == 20

    def test_name_matching_existing_alias_is_dropped(self):
        roster = [CharacterInfo(name="Captain Ahab", aliases=["Ahab"])]
        found = [CharacterInfo(name="Ahab")]
        merged = text_utils.merge_rosters(roster, found)
        assert [c.name for c in merged] == ["Captain Ahab"]

    def test_case_insensitive(self):
        roster = [CharacterInfo(name="Pip")]
        found = [CharacterInfo(name="PIP")]
        assert len(text_utils.merge_rosters(roster, found)) == 1


class TestApplyRosterIssues:
    def _roster(self):
        return [
            CharacterInfo(name="Captain Ahab", aliases=["Ahab"], first_appearance_chapter=16),
            CharacterInfo(name="Peleg", first_appearance_chapter=16),
            CharacterInfo(name="Pequod", first_appearance_chapter=16),
            CharacterInfo(name="Captain Peleg", first_appearance_chapter=16),
        ]

    def test_not_a_character_drops(self):
        from gutenberg_reader.stages.s06_critic import apply_roster_issues

        roster, applied = apply_roster_issues(
            self._roster(),
            [{"name": "Pequod", "verdict": "not_a_character", "canonical": "", "reason": "a ship"}],
            protected=set(),
        )
        assert "Pequod" not in [c.name for c in roster]
        assert len(applied) == 1

    def test_anchor_protection_beats_critic(self):
        from gutenberg_reader.stages.s06_critic import apply_roster_issues

        roster, applied = apply_roster_issues(
            self._roster(),
            [{"name": "Pequod", "verdict": "not_a_character", "canonical": "", "reason": ""}],
            protected={"pequod"},
        )
        assert "Pequod" in [c.name for c in roster]
        assert applied == []

    def test_protection_extends_to_aliases(self):
        from gutenberg_reader.stages.s06_critic import apply_roster_issues

        # "said Ahab" anchored the alias; the entry survives under any name.
        roster, applied = apply_roster_issues(
            self._roster(),
            [{"name": "Captain Ahab", "verdict": "not_a_character", "canonical": "", "reason": ""}],
            protected={"ahab"},
        )
        assert "Captain Ahab" in [c.name for c in roster]

    def test_duplicate_merges_as_alias(self):
        from gutenberg_reader.stages.s06_critic import apply_roster_issues

        roster, applied = apply_roster_issues(
            self._roster(),
            [{"name": "Peleg", "verdict": "duplicate", "canonical": "Captain Peleg", "reason": ""}],
            protected=set(),
        )
        names = [c.name for c in roster]
        assert "Peleg" not in names
        target = next(c for c in roster if c.name == "Captain Peleg")
        assert "Peleg" in target.aliases

    def test_idempotent_on_resumed_snapshot(self):
        from gutenberg_reader.stages.s06_critic import apply_roster_issues

        issues = [
            {"name": "Pequod", "verdict": "not_a_character", "canonical": "", "reason": ""},
            {"name": "Peleg", "verdict": "duplicate", "canonical": "Captain Peleg", "reason": ""},
        ]
        once, _ = apply_roster_issues(self._roster(), issues, protected=set())
        twice, applied = apply_roster_issues(once, issues, protected=set())
        assert [c.name for c in twice] == [c.name for c in once]
        assert applied == []

    def test_unknown_name_and_self_merge_ignored(self):
        from gutenberg_reader.stages.s06_critic import apply_roster_issues

        roster, applied = apply_roster_issues(
            self._roster(),
            [
                {"name": "Moby Dick", "verdict": "not_a_character", "canonical": "", "reason": ""},
                {"name": "Peleg", "verdict": "duplicate", "canonical": "Peleg", "reason": ""},
            ],
            protected=set(),
        )
        assert len(roster) == 4
        assert applied == []


class TestRosterSnapshotRoundTrip:
    def test_snapshot_restores_chapter_roster_and_anchors(self, tmp_path):
        from gutenberg_reader.cache import atomic_write_json, chapter_file
        from gutenberg_reader.config import Config
        from gutenberg_reader.models import ProcessedChapter, Segment
        from gutenberg_reader.stages.s05_segments import _load_cached

        config = Config(book_id="test", cache_dir=tmp_path)
        chapter = ProcessedChapter(
            chapter_number=3,
            chapter_title="Chapter 3",
            segments=[Segment(type="narration", text="Call me Ishmael.", speaker=None)],
            word_count=3,
        )
        roster = [CharacterInfo(name="Ishmael", first_appearance_chapter=1)]
        path = chapter_file(tmp_path, 3)
        data = chapter.to_dict()
        data["roster_after"] = [c.to_dict() for c in roster]
        data["anchor_names"] = ["ishmael"]
        atomic_write_json(path, data)

        loaded = _load_cached(path, config)
        assert loaded is not None
        got_chapter, got_roster, got_anchors = loaded
        assert got_chapter.chapter_number == 3
        assert [c.name for c in got_roster] == ["Ishmael"]
        assert got_anchors == {"ishmael"}

    def test_pre_snapshot_file_counts_as_incomplete(self, tmp_path):
        from gutenberg_reader.cache import atomic_write_json, chapter_file
        from gutenberg_reader.config import Config
        from gutenberg_reader.models import ProcessedChapter
        from gutenberg_reader.stages.s05_segments import _load_cached

        config = Config(book_id="test", cache_dir=tmp_path)
        chapter = ProcessedChapter(chapter_number=3, chapter_title="Chapter 3", segments=[])
        path = chapter_file(tmp_path, 3)
        atomic_write_json(path, chapter.to_dict())  # no roster_after

        assert _load_cached(path, config) is None

    def test_force_stage_invalidates(self, tmp_path):
        from gutenberg_reader.cache import atomic_write_json, chapter_file
        from gutenberg_reader.config import Config
        from gutenberg_reader.models import ProcessedChapter
        from gutenberg_reader.stages.s05_segments import _load_cached

        chapter = ProcessedChapter(chapter_number=3, chapter_title="Chapter 3", segments=[])
        path = chapter_file(tmp_path, 3)
        data = chapter.to_dict()
        data["roster_after"] = []
        atomic_write_json(path, data)

        # force-stage 4 and 5 both re-run the loop; 6 keeps its cache
        for stage, expect_cached in [(4, False), (5, False), (6, True)]:
            config = Config(book_id="test", cache_dir=tmp_path, force_stage=stage)
            assert (_load_cached(path, config) is not None) == expect_cached, stage
