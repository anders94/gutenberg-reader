"""Filing a finished book for committing.

cache/ is scratch and gitignored; library/ is the finished work under version
control. The point is that re-processing a title produces a reviewable diff
against the last time it was good — which only holds if an unchanged analysis
produces an unchanged file.
"""

from __future__ import annotations

import json

import pytest

from gutenberg_reader import library


@pytest.mark.parametrize("title,expected", [
    ("Pride and Prejudice", "pride-and-prejudice"),
    ("The Adventures of Sherlock Holmes", "the-adventures-of-sherlock-holmes"),
    # Subtitles are cut: how the book is referred to, and a readable listing.
    ("Moby Dick; Or, The Whale", "moby-dick"),
    ("Jane Eyre: An Autobiography", "jane-eyre"),
    ("A Room with a View", "a-room-with-a-view"),
    ("The Confessions of St. Augustine", "the-confessions-of-st-augustine"),
    # Accents and punctuation flatten rather than escaping into a filename.
    ("Les Misérables", "les-miserables"),
    ("What Maisie Knew!!", "what-maisie-knew"),
    ("", "untitled"),
])
def test_slug(title, expected):
    assert library.slug_for(title) == expected


def test_a_very_long_title_is_cut_on_a_word():
    slug = library.slug_for("The Life and Strange Surprizing Adventures of "
                            "Robinson Crusoe of York Mariner")
    assert len(slug) <= library.MAX_SLUG_CHARS
    assert not slug.endswith("-")


def test_the_filename_carries_the_id_and_the_title():
    assert library.library_name("1342", "Pride and Prejudice") == \
        "1342-pride-and-prejudice.json"


def _book(tmp_path, **stats):
    src = tmp_path / "1342.json"
    src.write_text(json.dumps({
        "metadata": {"gutenberg_id": "1342", "title": "Pride and Prejudice"},
        "chapters": [], "characters": [],
        "statistics": {"total_words": 121700, "processing_time_seconds": 7554.2,
                       **stats},
    }))
    return src


def test_install_writes_the_slugged_name(tmp_path):
    out = library.install(_book(tmp_path), tmp_path / "library")
    assert out.name == "1342-pride-and-prejudice.json"
    assert json.loads(out.read_text())["metadata"]["title"] == "Pride and Prejudice"


def test_run_timing_is_dropped_so_a_rerun_diffs_clean(tmp_path):
    """Keeping it would make every re-processing a diff even when nothing about
    the analysis changed — the exact question the library exists to answer."""
    lib = tmp_path / "library"
    fast = tmp_path / "fast"
    slow = tmp_path / "slow"
    fast.mkdir()
    slow.mkdir()

    first = library.install(_book(fast), lib).read_text()
    again = library.install(
        _book(slow, processing_time_seconds=9999.0), lib).read_text()

    assert "processing_time_seconds" not in first
    assert first == again, "a slower run must not show up as a change"
    assert json.loads(first)["statistics"]["total_words"] == 121700


def test_a_partial_run_is_never_installed():
    """--chapters produces a fragment; installing it would overwrite a whole
    book with a piece of one."""
    from gutenberg_reader.config import Config
    from gutenberg_reader.stages.s07_assemble import _install_to_library

    cfg = Config(book_id="1342", chapters_only=[1, 2])
    # No file is touched because the guard returns before reading anything.
    _install_to_library(cfg, None)


def test_an_install_failure_cannot_lose_the_book(tmp_path, capsys):
    """This runs at the end of a job that can take hours. A filename or
    permission problem must not throw away a result already written to cache/."""
    from gutenberg_reader.config import Config
    from gutenberg_reader.stages.s07_assemble import _install_to_library

    cfg = Config(book_id="1342", library_dir=tmp_path / "library")
    _install_to_library(cfg, tmp_path / "does-not-exist.json")
    assert "could not install" in capsys.readouterr().out
