"""Download the ds004504 OpenNeuro dataset into data/raw."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from ecg.config import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenNeuro ds004504 into the configured raw data path.")
    parser.add_argument("--config", type=Path, default=Path("cfgs/project.yaml"))
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="ds004504")
    parser.add_argument("--tag", default="1.0.8")
    parser.add_argument("--force", action="store_true", help="Run the download even if the target directory exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the wrapped uvx command without running it.")
    return parser.parse_args()


def build_command(*, dataset: str, tag: str, target_dir: Path) -> list[str]:
    return [
        "uvx",
        "openneuro-py@latest",
        "download",
        f"--dataset={dataset}",
        f"--tag={tag}",
        f"--target-dir={target_dir}",
    ]


def shell_command(command: list[str]) -> str:
    return " ".join(command[:3]) + " \\\n  " + " \\\n  ".join(command[3:])


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    target_dir = args.target_dir or config.dataset.raw.path
    command = build_command(dataset=args.dataset, tag=args.tag, target_dir=target_dir)

    if args.dry_run:
        print(shell_command(command))
        return

    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        print(f"Raw dataset already exists: {target_dir}")
        print("Use --force to run the OpenNeuro download command anyway.")
        return

    if shutil.which("uvx") is None:
        raise SystemExit("uvx was not found on PATH. Install uv first, then rerun this command.")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(shell_command(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
