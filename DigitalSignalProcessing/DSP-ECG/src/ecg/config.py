"""Typed project configuration loaded from cfgs/project.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawDatasetConfig(StrictConfig):
    path: Path
    participants_tsv: Path
    eeg_source: Literal["raw", "derivatives"]
    eeg_glob: str


class DatasetConfig(StrictConfig):
    raw: RawDatasetConfig


class RbpConfig(StrictConfig):
    total_power_range_hz: tuple[float, float]
    bands: dict[str, tuple[float, float]]

    @field_validator("total_power_range_hz")
    @classmethod
    def validate_total_power_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low >= high:
            raise ValueError("total_power_range_hz must be [low, high] with low < high")
        return value

    @field_validator("bands")
    @classmethod
    def validate_bands(cls, value: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        if not value:
            raise ValueError("rbp.bands must define at least one band")
        for band_name, (low, high) in value.items():
            if low >= high:
                raise ValueError(f"Band {band_name!r} must be [low, high] with low < high")
        return value


class H5DatasetConfig(StrictConfig):
    dtype: str
    shape: tuple[str, ...]
    dimensions: tuple[str, ...] | None = None
    description: str


class ProcessRawDatasetConfig(StrictConfig):
    rbp_epochs_h5: Path
    manifest_json: Path
    rbp: RbpConfig
    h5_datasets: dict[str, H5DatasetConfig] = Field(min_length=1)


class ProjectConfig(StrictConfig):
    dataset: DatasetConfig
    process_raw_dataset: ProcessRawDatasetConfig


def load_project_config(path: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return ProjectConfig.model_validate(raw_config)
