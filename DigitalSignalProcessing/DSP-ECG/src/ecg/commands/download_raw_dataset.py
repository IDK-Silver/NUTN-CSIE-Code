"""Download a supported raw dataset into its configured raw data path."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from ecg.data.openneuro import format_shell_command
from ecg.data.registry import (
    UnsupportedDatasetError,
    get_raw_dataset_pipeline,
    supported_raw_dataset_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a supported OpenNeuro dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Raw dataset id. Supported datasets: {supported_raw_dataset_text()}.",
    )
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="OpenNeuro version tag.")
    parser.add_argument("--force", action="store_true", help="Run the download even if the target directory exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the wrapped uvx command without running it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pipeline = get_raw_dataset_pipeline(args.dataset)
    except UnsupportedDatasetError as exc:
        raise SystemExit(str(exc)) from exc

    target_dir = args.target_dir
    command = pipeline.build_download_command(args.tag, target_dir)

    if args.dry_run:
        print(format_shell_command(command))
        return

    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        print(f"Raw dataset already exists: {target_dir}")
        print("Use --force to run the OpenNeuro download command anyway.")
        return

    if shutil.which("uvx") is None:
        raise SystemExit("uvx was not found on PATH. Install uv first, then rerun this command.")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(format_shell_command(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
