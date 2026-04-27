"""Convert the ds004504 EEGLAB files into an RBP H5 cache.

This script only processes the raw dataset into a complete feature dataset.
It does not create train/validation/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import h5py
import mne
import numpy as np
import numpy.typing as npt
from scipy import signal

from ecg.config import load_project_config


LABEL_MAP = {
    "A": 0,
    "F": 1,
    "C": 2,
}

LABEL_NAMES = {
    "A": "alzheimer",
    "F": "frontotemporal_dementia",
    "C": "healthy_control",
}

SUBJECT_RE = re.compile(r"(sub-\d+)")


def dataset_download_command(target_dir: Path) -> str:
    return (
        "uvx openneuro-py@latest download \\\n"
        "  --dataset=ds004504 \\\n"
        "  --tag=1.0.8 \\\n"
        f"  --target-dir={target_dir}"
    )


def fail_missing_dataset(message: str, raw_dataset_path: Path) -> None:
    print(f"Warning: {message}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Download the dataset first:", file=sys.stderr)
    print("", file=sys.stderr)
    print(dataset_download_command(raw_dataset_path), file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/processed_raw_dataset/rbp_epochs.h5 from ds004504 .set files."
    )
    parser.add_argument("--config", type=Path, default=Path("cfgs/project.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N EEG files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Kept as CLI parameters for now so project.yaml remains a dataset/schema config.
    parser.add_argument("--epoch-sec", type=float, default=6.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--welch-window-sec", type=float, default=2.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    return parser.parse_args()


def load_participants(path: Path) -> dict[str, dict[str, str]]:
    participants: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clean_row = {key: value.strip() for key, value in row.items()}
            participants[clean_row["participant_id"]] = clean_row
    return participants


def subject_id_from_path(path: Path) -> str:
    match = SUBJECT_RE.search(str(path))
    if not match:
        raise ValueError(f"Cannot infer subject id from path: {path}")
    return match.group(1)


def band_mask(freqs: np.ndarray, low: float, high: float, include_high: bool) -> np.ndarray:
    if include_high:
        return (freqs >= low) & (freqs <= high)
    return (freqs >= low) & (freqs < high)


def sum_psd_bins(freqs: np.ndarray, psd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected_freqs = freqs[mask]
    selected_psd = psd[..., mask]
    if selected_freqs.size == 0:
        raise ValueError("Frequency mask selected no Welch bins.")
    return selected_psd.sum(axis=-1)


def compute_rbp_for_epoch(
    epoch_data: np.ndarray,
    *,
    sampling_rate_hz: float,
    bands: dict[str, tuple[float, float]],
    total_power_range_hz: tuple[float, float],
    welch_window_sec: float,
    welch_overlap: float,
) -> np.ndarray:
    nperseg = int(round(welch_window_sec * sampling_rate_hz))
    noverlap = int(round(nperseg * welch_overlap))
    if nperseg <= 0:
        raise ValueError("Welch nperseg must be positive.")
    if noverlap >= nperseg:
        raise ValueError("Welch overlap must be smaller than the Welch window.")

    freqs, psd = signal.welch(
        epoch_data,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=min(nperseg, epoch_data.shape[-1]),
        noverlap=min(noverlap, max(0, epoch_data.shape[-1] - 1)),
        detrend="constant",
        scaling="density",
        axis=-1,
        average="mean",
    )

    total_mask = band_mask(
        freqs,
        total_power_range_hz[0],
        total_power_range_hz[1],
        include_high=True,
    )
    total_power = sum_psd_bins(freqs, psd, total_mask)

    rbp_by_band: list[np.ndarray] = []
    for low, high in bands.values():
        include_high = high == total_power_range_hz[1]
        power = sum_psd_bins(freqs, psd, band_mask(freqs, low, high, include_high))
        rbp = np.divide(
            power,
            total_power,
            out=np.full_like(power, np.nan, dtype=np.float64),
            where=total_power > 0,
        )
        rbp_by_band.append(rbp)

    return np.stack(rbp_by_band, axis=-1).astype(np.float32)


def process_one_file(
    eeg_path: Path,
    *,
    participants: dict[str, dict[str, str]],
    bands: dict[str, tuple[float, float]],
    total_power_range_hz: tuple[float, float],
    epoch_sec: float,
    overlap: float,
    welch_window_sec: float,
    welch_overlap: float,
) -> tuple[np.ndarray, list[str], np.ndarray, list[float], dict[str, Any]]:
    subject_id = subject_id_from_path(eeg_path)
    if subject_id not in participants:
        raise KeyError(f"{subject_id} is missing from participants.tsv")

    group = participants[subject_id]["Group"]
    if group not in LABEL_MAP:
        raise KeyError(f"Unsupported Group value for {subject_id}: {group!r}")

    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    data: npt.NDArray[np.float64] = np.asarray(raw.get_data(), dtype=np.float64)
    sampling_rate_hz = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)

    epoch_samples = int(round(epoch_sec * sampling_rate_hz))
    stride_samples = int(round(epoch_samples * (1.0 - overlap)))
    if epoch_samples <= 0:
        raise ValueError("epoch_sec produced zero samples.")
    if stride_samples <= 0:
        raise ValueError("overlap is too large; stride must be positive.")
    if data.shape[1] < epoch_samples:
        raise ValueError(f"{subject_id} has fewer samples than one epoch.")

    starts = list(range(0, data.shape[1] - epoch_samples + 1, stride_samples))
    rbp_epochs = np.empty((len(starts), data.shape[0], len(bands)), dtype=np.float32)

    for epoch_idx, start in enumerate(starts):
        epoch = data[:, start : start + epoch_samples]
        rbp_epochs[epoch_idx] = compute_rbp_for_epoch(
            epoch,
            sampling_rate_hz=sampling_rate_hz,
            bands=bands,
            total_power_range_hz=total_power_range_hz,
            welch_window_sec=welch_window_sec,
            welch_overlap=welch_overlap,
        )

    labels = np.full(len(starts), LABEL_MAP[group], dtype=np.int64)
    epoch_start_sec = [start / sampling_rate_hz for start in starts]
    file_record = {
        "subject_id": subject_id,
        "group": group,
        "label_name": LABEL_NAMES[group],
        "path": str(eeg_path),
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "sampling_rate_hz": sampling_rate_hz,
        "duration_sec": float(data.shape[1] / sampling_rate_hz),
        "n_epochs": len(starts),
    }
    return rbp_epochs, channel_names, labels, epoch_start_sec, file_record


def write_h5(
    output_path: Path,
    *,
    x_rbp_channel: np.ndarray,
    labels: np.ndarray,
    subject_ids: list[str],
    epoch_start_sec: np.ndarray,
    channel_names: list[str],
    band_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    x_rbp_mean = x_rbp_channel.mean(axis=1, dtype=np.float32)

    with NamedTemporaryFile(dir=output_path.parent, suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("X_rbp_mean", data=x_rbp_mean, compression="gzip", shuffle=True)
            h5.create_dataset("X_rbp_channel", data=x_rbp_channel, compression="gzip", shuffle=True)
            h5.create_dataset("y", data=labels)
            h5.create_dataset("subject_id", data=np.asarray(subject_ids, dtype=object), dtype=string_dtype)
            h5.create_dataset("epoch_start_sec", data=epoch_start_sec)
            h5.create_dataset("channel_names", data=np.asarray(channel_names, dtype=object), dtype=string_dtype)
            h5.create_dataset("band_names", data=np.asarray(band_names, dtype=object), dtype=string_dtype)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=manifest_path.parent, suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(manifest, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(manifest_path)


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    raw_config = config.dataset.raw
    processed_config = config.process_raw_dataset
    rbp_config = processed_config.rbp

    raw_dataset_path = raw_config.path
    if not raw_dataset_path.exists():
        fail_missing_dataset(f"raw dataset path does not exist: {raw_dataset_path}", raw_dataset_path)
    if not raw_dataset_path.is_dir():
        fail_missing_dataset(f"raw dataset path is not a directory: {raw_dataset_path}", raw_dataset_path)

    output_path = args.output or processed_config.rbp_epochs_h5
    manifest_path = args.manifest or processed_config.manifest_json
    if not args.overwrite and not args.dry_run:
        existing = [path for path in (output_path, manifest_path) if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing output(s): {names}")

    participants_path = raw_config.participants_tsv
    if not participants_path.exists():
        fail_missing_dataset(f"participants.tsv is missing: {participants_path}", raw_dataset_path)

    participants = load_participants(participants_path)
    eeg_paths = sorted(Path().glob(raw_config.eeg_glob))
    if args.limit is not None:
        eeg_paths = eeg_paths[: args.limit]
    if not eeg_paths:
        fail_missing_dataset(f"No EEG files matched: {raw_config.eeg_glob}", raw_dataset_path)

    if args.dry_run:
        print(f"Matched EEG files: {len(eeg_paths)}")
        print(f"Output H5: {output_path}")
        print(f"Manifest JSON: {manifest_path}")
        return

    all_rbp_channel: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subject_ids: list[str] = []
    all_epoch_start_sec: list[float] = []
    file_records: list[dict[str, Any]] = []
    reference_channels: list[str] | None = None

    for index, eeg_path in enumerate(eeg_paths, start=1):
        print(f"[{index}/{len(eeg_paths)}] processing {eeg_path}")
        rbp_channel, channel_names, labels, epoch_start_sec, file_record = process_one_file(
            eeg_path,
            participants=participants,
            bands=rbp_config.bands,
            total_power_range_hz=rbp_config.total_power_range_hz,
            epoch_sec=args.epoch_sec,
            overlap=args.overlap,
            welch_window_sec=args.welch_window_sec,
            welch_overlap=args.welch_overlap,
        )
        if reference_channels is None:
            reference_channels = channel_names
        elif channel_names != reference_channels:
            raise ValueError(f"Channel names differ for {eeg_path}")

        all_rbp_channel.append(rbp_channel)
        all_labels.append(labels)
        all_subject_ids.extend([file_record["subject_id"]] * len(labels))
        all_epoch_start_sec.extend(epoch_start_sec)
        file_records.append(file_record)

    if reference_channels is None:
        raise RuntimeError("No files were processed.")

    x_rbp_channel = np.concatenate(all_rbp_channel, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    epoch_start_sec = np.asarray(all_epoch_start_sec, dtype=np.float32)
    band_names = list(rbp_config.bands.keys())

    write_h5(
        output_path,
        x_rbp_channel=x_rbp_channel,
        labels=labels,
        subject_ids=all_subject_ids,
        epoch_start_sec=epoch_start_sec,
        channel_names=reference_channels,
        band_names=band_names,
    )

    label_counts = Counter(int(label) for label in labels)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "output_h5": str(output_path),
        "dataset": {
            "path": str(raw_config.path),
            "participants_tsv": str(raw_config.participants_tsv),
            "eeg_source": raw_config.eeg_source,
            "eeg_glob": raw_config.eeg_glob,
        },
        "processing": {
            "epoch_sec": args.epoch_sec,
            "overlap": args.overlap,
            "welch_window_sec": args.welch_window_sec,
            "welch_overlap": args.welch_overlap,
            "rbp_total_power_range_hz": rbp_config.total_power_range_hz,
            "rbp_bands": rbp_config.bands,
        },
        "h5_datasets": {
            name: dataset.model_dump(mode="json") for name, dataset in processed_config.h5_datasets.items()
        },
        "label_map": LABEL_MAP,
        "label_names": LABEL_NAMES,
        "summary": {
            "n_source_files": len(file_records),
            "n_subjects": len({record["subject_id"] for record in file_records}),
            "n_epochs": int(x_rbp_channel.shape[0]),
            "n_channels": int(x_rbp_channel.shape[1]),
            "n_bands": int(x_rbp_channel.shape[2]),
            "epoch_counts_by_label": {str(label): count for label, count in sorted(label_counts.items())},
        },
        "source_files": file_records,
    }
    write_manifest(manifest_path, manifest)

    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")
    print(f"Shape X_rbp_channel: {tuple(x_rbp_channel.shape)}")


if __name__ == "__main__":
    main()
