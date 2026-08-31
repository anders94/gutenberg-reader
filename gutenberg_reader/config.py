"""Configuration dataclass for the pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    book_id: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    processing_model: str = ""  # empty = auto-detect from the server
    validation_model: str = ""
    # Judgment work — verify, tie-break, critic, structure — can run on another
    # box entirely. Empty inherits from the endpoint above it.
    validator_base_url: str = ""
    structure_base_url: str = ""
    structure_model: str = ""
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    output_file: Path | None = None
    chunk_size: int = 1000
    max_retries: int = 3
    verbose: bool = False
    no_critic: bool = False
    force_stage: int | None = None
    chapters_only: list[int] | None = None
    include_front_matter: bool = False
    include_back_matter: bool = False
    accept_structure_warnings: bool = False
    structure_detector: str = "llm"   # "llm" | "regex"
    # Chain-of-thought is waste on the structure pass and costs an order of
    # magnitude. Measured on goldberry (DeepSeek-V4-Flash) for PG 2641: with
    # thinking 5,131 completion tokens in 157s, without 316 tokens in 8s — the
    # same correct 19 chapters. Identifying a series in a list is recognition,
    # not deduction. Servers ignore template keys they do not declare.
    structure_thinking: bool = False
    processing_timeout: float = 300.0
    judgment_timeout: float = 1800.0

    def __post_init__(self):
        if not self.validation_model:
            self.validation_model = self.processing_model
        if not self.validator_base_url:
            self.validator_base_url = self.base_url
        if not self.structure_base_url:
            self.structure_base_url = self.validator_base_url
        if not self.structure_model:
            self.structure_model = self.validation_model
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
