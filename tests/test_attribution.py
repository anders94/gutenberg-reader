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


class TestSplitSentences:
    def test_abbreviations_do_not_split(self):
        from gutenberg_reader.segmenter import split_sentences

        text = "Mr. Bennet made no answer. Mrs. Long and Capt. Carter arrived."
        assert split_sentences(text) == [
            "Mr. Bennet made no answer.",
            "Mrs. Long and Capt. Carter arrived.",
        ]

    def test_initials_do_not_split(self):
        from gutenberg_reader.segmenter import split_sentences

        text = "So says J. Ross Browne in his book. He was wrong."
        assert split_sentences(text) == [
            "So says J. Ross Browne in his book.",
            "He was wrong.",
        ]

    def test_lowercase_continuation_does_not_split(self):
        from gutenberg_reader.segmenter import split_sentences

        text = "What then? he wondered aloud, and walked on."
        assert split_sentences(text) == [text]

    def test_terminal_quotes_stay_attached(self):
        from gutenberg_reader.segmenter import split_sentences

        text = "It was called “the whale.” Nobody argued."
        assert split_sentences(text) == ["It was called “the whale.”", "Nobody argued."]


class TestSplitLongNarration:
    def _seg(self, text, kind="narration", para=0):
        return {
            "type": kind, "text": text, "speaker": None,
            "pronunciation_hints": [], "notes": None, "para": para,
        }

    def test_short_narration_untouched(self):
        from gutenberg_reader.segmenter import split_long_narration

        segs = [self._seg("A short line.")]
        assert split_long_narration(segs) == segs

    def test_long_narration_packs_to_limit(self):
        from gutenberg_reader.segmenter import split_long_narration

        sentence = "This sentence is exactly fifty characters long!!! "
        segs = [self._seg((sentence * 12).strip())]  # ~600 chars
        out = split_long_narration(segs, max_chars=400)
        assert len(out) > 1
        assert all(len(s["text"]) <= 400 for s in out)
        # No text lost, order preserved
        rejoined = " ".join(s["text"] for s in out)
        assert rejoined == segs[0]["text"]
        # Chunks are packed, not one-sentence-each
        assert any(len(s["text"]) > 200 for s in out)

    def test_monster_sentence_stays_whole(self):
        from gutenberg_reader.segmenter import split_long_narration

        monster = "and so on, " * 60  # one 700-char "sentence", no terminals
        segs = [self._seg("Short opener. " + monster.strip() + ".")]
        out = split_long_narration(segs, max_chars=400)
        assert any(len(s["text"]) > 400 for s in out)  # lenient: not cut mid-sentence

    def test_dialogue_never_split(self):
        from gutenberg_reader.segmenter import split_long_narration

        long_speech = "“" + ("I will talk. " * 60).strip() + "”"
        segs = [self._seg(long_speech, kind="dialogue")]
        assert split_long_narration(segs, max_chars=400) == segs

    def test_pieces_keep_paragraph_and_type(self):
        from gutenberg_reader.segmenter import split_long_narration

        text = " ".join(f"Sentence number {i} is here." for i in range(30))
        out = split_long_narration([self._seg(text, para=7)], max_chars=400)
        assert all(s["para"] == 7 and s["type"] == "narration" for s in out)

    def test_segment_text_coverage_still_exact(self):
        from gutenberg_reader.segmenter import segment_text

        text = (
            "It is a truth universally acknowledged, that a single man in "
            "possession of a good fortune, must be in want of a wife. " * 6
            + "\n\n“Come here,” said Mr. Bennet. “Now go.”"
        )
        segs = segment_text(text)
        ok, issues = text_utils.verify_segment_coverage(text, segs)
        assert ok, issues
        assert any(s["type"] == "dialogue" for s in segs)
        assert all(
            len(s["text"]) <= 400 for s in segs if s["type"] == "narration"
        )


class TestCanonicalNamePreference:
    def test_descriptive_primary_demoted_to_alias(self):
        # Discovery emitted the description as primary and the real name as
        # alias; 125 P&P segments shipped labeled "Jane's mother".
        chars = [
            CharacterInfo(name="Jane's mother", aliases=["Mrs. Bennet", "his wife"]),
        ]
        merged = text_utils.merge_duplicate_characters(chars)
        assert merged[0].name == "Mrs. Bennet"
        assert "Jane's mother" in merged[0].aliases
        assert "his wife" in merged[0].aliases

    def test_merge_prefers_proper_name_canonical(self):
        chars = [
            CharacterInfo(name="Mrs. Bennet", first_appearance_chapter=1),
            CharacterInfo(name="Jane's mother", aliases=["Mrs. Bennet"], first_appearance_chapter=2),
        ]
        merged = text_utils.merge_duplicate_characters(chars)
        assert len(merged) == 1
        assert merged[0].name == "Mrs. Bennet"
        assert merged[0].first_appearance_chapter == 1

    def test_title_keeps_canonical_over_bare_name(self):
        chars = [CharacterInfo(name="Captain Ahab", aliases=["Ahab", "old man"])]
        merged = text_utils.merge_duplicate_characters(chars)
        assert merged[0].name == "Captain Ahab"

    def test_full_name_keeps_canonical_over_nickname(self):
        chars = [CharacterInfo(name="Elizabeth Bennet", aliases=["Lizzy", "Eliza"])]
        merged = text_utils.merge_duplicate_characters(chars)
        assert merged[0].name == "Elizabeth Bennet"


class TestDiscoveryWindowing:
    def test_short_chapter_is_one_window(self):
        from gutenberg_reader.stages.s04_characters import _split_for_discovery

        text = "A short chapter.\n\nWith two paragraphs."
        assert _split_for_discovery(text, 6000) == [text]

    def test_long_chapter_splits_at_paragraphs(self):
        from gutenberg_reader.stages.s04_characters import _split_for_discovery

        para = " ".join(["word"] * 100)
        text = "\n\n".join([para] * 50)  # 5,000 words
        windows = _split_for_discovery(text, 1000)
        assert len(windows) == 5
        assert all(text_utils.word_count(w) <= 1000 for w in windows)
        # No text lost, order preserved
        assert "\n\n".join(windows) == text

    def test_oversized_paragraph_kept_whole(self):
        from gutenberg_reader.stages.s04_characters import _split_for_discovery

        text = " ".join(["word"] * 3000)
        assert _split_for_discovery(text, 1000) == [text]


class TestNarratorRoleNames:
    def test_narrator_roles_rejected(self):
        for name in ("Narrator", "The Narrator", "the narrator", "I", "me",
                     "Myself", "The Author", "storyteller",
                     # Told not to emit "Narrator", the model qualifies it instead.
                     "Unnamed narrator", "An anonymous narrator", "Our Storyteller",
                     "first-person narrator"):
            assert text_utils.is_narrator_role(name), name
            assert text_utils.is_reserved_character_name(name), name

    def test_real_names_survive(self):
        for name in ("Ishmael", "Dr. Watson", "Mary", "Ivan", "Io",
                     # A real person described by relation — ends on the noun
                     # that names her, so the whole-name match must not fire.
                     "Narrator's Wife", "The Narrator's Brother",
                     "Mr. I. M. Wright", "Author Jones", "Mr. Speaker Clay"):
            assert not text_utils.is_narrator_role(name), name
            assert not text_utils.is_reserved_character_name(name), name

    def test_placeholders_still_reserved(self):
        for name in ("N/A", "None", "Unknown", "No named characters found"):
            assert text_utils.is_reserved_character_name(name), name

    def test_discovery_filters_narrator_entry(self):
        # The PG 1661 defect: "The Narrator" (aliases: I, Narrator) sat beside
        # "Dr. Watson" and took 106 of Watson's lines. It shares no tokens with
        # "Dr. Watson", so merge_duplicate_characters can never reconcile them.
        raw = {
            "characters": [
                {"name": "Dr. Watson", "aliases": ["Watson"],
                 "pronunciation_hints": [], "first_appearance_chapter": 1},
                {"name": "The Narrator", "aliases": ["I", "Narrator"],
                 "pronunciation_hints": [], "first_appearance_chapter": 1},
            ]
        }
        kept = [
            c["name"] for c in raw["characters"]
            if not text_utils.is_reserved_character_name(c["name"])
        ]
        assert kept == ["Dr. Watson"]
