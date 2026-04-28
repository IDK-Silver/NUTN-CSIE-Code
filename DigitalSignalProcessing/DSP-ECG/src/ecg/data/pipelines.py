"""Dataset pipeline contracts used by CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessRawDatasetOptions:
    raw_dir: Path
    output: Path
    manifest: Path
    limit: int | None
    overwrite: bool
    dry_run: bool


@dataclass(frozen=True)
class RawDatasetPipeline:
    dataset_id: str
    build_download_command: Callable[[str, Path], list[str]]


@dataclass(frozen=True)
class ProcessedDatasetPipeline:
    dataset_id: str
    process_raw_dataset: Callable[[ProcessRawDatasetOptions], None]
