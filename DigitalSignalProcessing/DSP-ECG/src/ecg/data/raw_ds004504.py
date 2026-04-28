"""OpenNeuro ds004504 raw dataset download helpers."""

from __future__ import annotations

from pathlib import Path


OPENNEURO_DATASET_ID = "ds004504"
OPENNEURO_DATASET_TAG = "1.0.8"


def build_ds004504_download_command(tag: str, target_dir: Path) -> list[str]:
    return [
        "uvx",
        "openneuro-py@latest",
        "download",
        f"--dataset={OPENNEURO_DATASET_ID}",
        f"--tag={tag}",
        f"--target-dir={target_dir}",
    ]

