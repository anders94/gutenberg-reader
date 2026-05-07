"""Atomic read/write helpers per stage."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text file atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Read a JSON file, raising FileNotFoundError if missing."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    """Read a text file, raising FileNotFoundError if missing."""
    return path.read_text(encoding="utf-8")


def stage_complete(path: Path) -> bool:
    """Return True if the given output path exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


def chapter_file(stage_dir: Path, chapter_num: int, suffix: str = ".json") -> Path:
    """Return the canonical path for a chapter file in a stage directory."""
    return stage_dir / f"chapter-{chapter_num:02d}{suffix}"
