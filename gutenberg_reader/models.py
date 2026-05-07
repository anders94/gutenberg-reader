"""Data structures for the gutenberg-reader pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Segment:
    type: str  # "narration" | "dialogue"
    text: str
    speaker: str | None
    pronunciation_hints: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
            "speaker": self.speaker,
            "pronunciation_hints": self.pronunciation_hints,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(
            type=d["type"],
            text=d["text"],
            speaker=d.get("speaker"),
            pronunciation_hints=d.get("pronunciation_hints", []),
            notes=d.get("notes"),
        )


@dataclass
class ChapterInfo:
    number: int
    title: str
    start_line: int
    end_line: int
    word_count: int = 0
    start_marker: str = ""

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "word_count": self.word_count,
            "start_marker": self.start_marker,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterInfo":
        return cls(
            number=d["number"],
            title=d["title"],
            start_line=d["start_line"],
            end_line=d["end_line"],
            word_count=d.get("word_count", 0),
            start_marker=d.get("start_marker", ""),
        )


@dataclass
class CharacterInfo:
    name: str
    aliases: list[str] = field(default_factory=list)
    pronunciation_hints: list[str] = field(default_factory=list)
    first_appearance_chapter: int = 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "pronunciation_hints": self.pronunciation_hints,
            "first_appearance_chapter": self.first_appearance_chapter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterInfo":
        return cls(
            name=d["name"],
            aliases=d.get("aliases", []),
            pronunciation_hints=d.get("pronunciation_hints", []),
            first_appearance_chapter=d.get("first_appearance_chapter", 1),
        )


@dataclass
class ProcessedChapter:
    chapter_number: int
    chapter_title: str
    segments: list[Segment]
    discovered_characters: list[CharacterInfo] = field(default_factory=list)
    word_count: int = 0
    is_segment: bool = False
    segment_index: int | None = None
    total_segments: int | None = None

    def to_dict(self) -> dict:
        return {
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "segments": [s.to_dict() for s in self.segments],
            "discovered_characters": [c.to_dict() for c in self.discovered_characters],
            "word_count": self.word_count,
            "is_segment": self.is_segment,
            "segment_index": self.segment_index,
            "total_segments": self.total_segments,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessedChapter":
        return cls(
            chapter_number=d["chapter_number"],
            chapter_title=d["chapter_title"],
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            discovered_characters=[CharacterInfo.from_dict(c) for c in d.get("discovered_characters", [])],
            word_count=d.get("word_count", 0),
            is_segment=d.get("is_segment", False),
            segment_index=d.get("segment_index"),
            total_segments=d.get("total_segments"),
        )


@dataclass
class CriticReport:
    chapter_number: int
    missing_text: list[str] = field(default_factory=list)
    attribution_issues: list[str] = field(default_factory=list)
    name_inconsistencies: list[str] = field(default_factory=list)
    overall_quality: float = 1.0
    needs_reprocessing: bool = False
    fixed_segments: list[Segment] | None = None

    def to_dict(self) -> dict:
        return {
            "chapter_number": self.chapter_number,
            "missing_text": self.missing_text,
            "attribution_issues": self.attribution_issues,
            "name_inconsistencies": self.name_inconsistencies,
            "overall_quality": self.overall_quality,
            "needs_reprocessing": self.needs_reprocessing,
            "fixed_segments": [s.to_dict() for s in self.fixed_segments] if self.fixed_segments else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CriticReport":
        return cls(
            chapter_number=d["chapter_number"],
            missing_text=d.get("missing_text", []),
            attribution_issues=d.get("attribution_issues", []),
            name_inconsistencies=d.get("name_inconsistencies", []),
            overall_quality=d.get("overall_quality", 1.0),
            needs_reprocessing=d.get("needs_reprocessing", False),
            fixed_segments=[Segment.from_dict(s) for s in d["fixed_segments"]] if d.get("fixed_segments") else None,
        )


@dataclass
class BookMetadata:
    title: str = ""
    author: str = ""
    language: str = "en"
    gutenberg_id: str = ""
    release_date: str = ""
    credits: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "gutenberg_id": self.gutenberg_id,
            "release_date": self.release_date,
            "credits": self.credits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BookMetadata":
        return cls(
            title=d.get("title", ""),
            author=d.get("author", ""),
            language=d.get("language", "en"),
            gutenberg_id=d.get("gutenberg_id", ""),
            release_date=d.get("release_date", ""),
            credits=d.get("credits", ""),
        )


@dataclass
class DiscoveryResult:
    metadata: BookMetadata
    chapters: list[ChapterInfo]
    body_start_line: int = 0
    body_end_line: int = 0

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "chapters": [c.to_dict() for c in self.chapters],
            "body_start_line": self.body_start_line,
            "body_end_line": self.body_end_line,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DiscoveryResult":
        return cls(
            metadata=BookMetadata.from_dict(d["metadata"]),
            chapters=[ChapterInfo.from_dict(c) for c in d["chapters"]],
            body_start_line=d.get("body_start_line", 0),
            body_end_line=d.get("body_end_line", 0),
        )
