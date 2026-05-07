"""Configuration dataclass for the pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    book_id: str
    ollama_url: str = "http://localhost:11434"
    processing_model: str = "qwen2.5:14b"
    validation_model: str = ""
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    output_file: Path | None = None
    chunk_size: int = 400
    chunk_overlap: int = 150
    max_retries: int = 3
    verbose: bool = False
    no_critic: bool = False
    force_stage: int | None = None
    chapters_only: list[int] | None = None

    def __post_init__(self):
        if not self.validation_model:
            self.validation_model = self.processing_model
        self.cache_dir = Path(self.cache_dir)

    @property
    def book_cache_dir(self) -> Path:
        return self.cache_dir / self.book_id

    @property
    def stage_dirs(self) -> dict[int, Path]:
        base = self.book_cache_dir
        return {
            1: base / "01-raw",
            2: base / "02-discovery",
            3: base / "03-chapters",
            4: base / "04-characters",
            5: base / "05-segments",
            6: base / "06-critic",
            7: base / "07-final",
        }

    def stage_dir(self, stage: int) -> Path:
        return self.stage_dirs[stage]
