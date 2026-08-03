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
