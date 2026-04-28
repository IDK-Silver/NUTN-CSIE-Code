"""Convert a supported raw dataset into the project RBP H5 cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg.data.pipelines import ProcessRawDatasetOptions
from ecg.data.registry import (
    UnsupportedDatasetError,
    get_processed_dataset_pipeline,
    supported_processed_dataset_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a processed RBP H5 cache from a supported dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Processed dataset id. Supported datasets: {supported_processed_dataset_text()}.",
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N EEG files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pipeline = get_processed_dataset_pipeline(args.dataset)
    except UnsupportedDatasetError as exc:
        raise SystemExit(str(exc)) from exc

    options = ProcessRawDatasetOptions(
        raw_dir=args.raw_dir,
        output=args.output,
        manifest=args.manifest,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    pipeline.process_raw_dataset(options)


if __name__ == "__main__":
    main()
