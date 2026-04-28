"""Paper-style RBP processing for OpenNeuro ds004504."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import h5py
import mne
import numpy as np
import numpy.typing as npt

from ecg.data.openneuro import format_shell_command
from ecg.data.pipelines import ProcessRawDatasetOptions
from ecg.data.raw_ds004504 import OPENNEURO_DATASET_ID, OPENNEURO_DATASET_TAG, build_ds004504_download_command
from ecg.data.rbp import compute_rbp_for_epoch


PROCESSED_DATASET_ID = "ds004504_rbp_paper"
PARTICIPANTS_TSV = "participants.tsv"
EEG_SOURCE = "derivatives"
EEG_GLOB = "derivatives/sub-*/eeg/*_task-eyesclosed_eeg.set"
TOTAL_POWER_RANGE_HZ = (0.5, 45.0)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 16.0),
    "zaeta": (16.0, 24.0),
    "beta": (24.0, 30.0),
    "gamma": (30.0, 45.0),
}
EPOCH_SEC = 6.0
OVERLAP = 0.5
WELCH_WINDOW_SEC = 2.0
WELCH_OVERLAP = 0.5
SUBJECT_RE = re.compile(r"(sub-\d+)")

LABEL_MAP: dict[str, int] = {
    "A": 0,
    "F": 1,
    "C": 2,
}

LABEL_NAMES: dict[str, str] = {
    "A": "alzheimer",
    "F": "frontotemporal_dementia",
    "C": "healthy_control",
}

ParticipantRows = dict[str, dict[str, str]]


def ds004504_download_command(target_dir: Path) -> str:
    command = build_ds004504_download_command(OPENNEURO_DATASET_TAG, target_dir)
    return format_shell_command(command)


def fail_missing_ds004504(message: str, raw_dataset_path: Path) -> None:
    print(f"Warning: {message}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Download the ds004504 dataset first:", file=sys.stderr)
    print("", file=sys.stderr)
    print(ds004504_download_command(raw_dataset_path), file=sys.stderr)
    raise SystemExit(1)


def load_ds004504_participants(path: Path) -> ParticipantRows:
    participants: ParticipantRows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clean_row = {key: value.strip() for key, value in row.items()}
            participants[clean_row["participant_id"]] = clean_row
    return participants


def subject_id_from_ds004504_path(path: Path) -> str:
    match = SUBJECT_RE.search(str(path))
    if not match:
        raise ValueError(f"Cannot infer subject id from ds004504 path: {path}")
    return match.group(1)


def process_one_ds004504_eeg_file(
    eeg_path: Path,
    *,
    participants: ParticipantRows,
    bands: dict[str, tuple[float, float]],
    total_power_range_hz: tuple[float, float],
    epoch_sec: float,
    overlap: float,
    welch_window_sec: float,
    welch_overlap: float,
) -> tuple[np.ndarray, list[str], np.ndarray, list[float], dict[str, Any]]:
    subject_id = subject_id_from_ds004504_path(eeg_path)
    if subject_id not in participants:
        raise KeyError(f"{subject_id} is missing from participants.tsv")

    group = participants[subject_id]["Group"]
    if group not in LABEL_MAP:
        raise KeyError(f"Unsupported ds004504 Group value for {subject_id}: {group!r}")

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


def write_ds004504_rbp_h5(
    output_path: Path,
    *,
    x_rbp_channel: np.ndarray,
    labels: np.ndarray,
    subject_ids: Sequence[str],
    epoch_start_sec: np.ndarray,
    channel_names: Sequence[str],
    band_names: Sequence[str],
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


def write_ds004504_manifest_json(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=manifest_path.parent, suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(manifest, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(manifest_path)


def rbp_h5_dataset_schema() -> dict[str, dict[str, Any]]:
    return {
        "X_rbp_mean": {
            "dtype": "float32",
            "shape": ["n_epochs", "n_bands"],
            "dimensions": ["epoch", "band"],
            "description": "channel-averaged RBP features for each epoch",
        },
        "X_rbp_channel": {
            "dtype": "float32",
            "shape": ["n_epochs", "n_channels", "n_bands"],
            "dimensions": ["epoch", "channel", "band"],
            "description": "per-channel RBP features for each epoch",
        },
        "y": {
            "dtype": "int64",
            "shape": ["n_epochs"],
            "description": "encoded class label",
        },
        "subject_id": {
            "dtype": "string",
            "shape": ["n_epochs"],
            "description": "BIDS participant id for each epoch",
        },
        "epoch_start_sec": {
            "dtype": "float32",
            "shape": ["n_epochs"],
            "description": "start time of each epoch in the source EEG recording",
        },
        "channel_names": {
            "dtype": "string",
            "shape": ["n_channels"],
            "description": "EEG channel names stored in the channel axis order",
        },
        "band_names": {
            "dtype": "string",
            "shape": ["n_bands"],
            "description": "RBP band names stored in the band axis order",
        },
    }


def process_raw_dataset(options: ProcessRawDatasetOptions) -> None:
    raw_dataset_path = options.raw_dir
    if not raw_dataset_path.exists():
        fail_missing_ds004504(f"raw dataset path does not exist: {raw_dataset_path}", raw_dataset_path)
    if not raw_dataset_path.is_dir():
        fail_missing_ds004504(f"raw dataset path is not a directory: {raw_dataset_path}", raw_dataset_path)

    output_path = options.output
    manifest_path = options.manifest
    if not options.overwrite and not options.dry_run:
        existing = [path for path in (output_path, manifest_path) if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing output(s): {names}")

    participants_path = raw_dataset_path / PARTICIPANTS_TSV
    if not participants_path.exists():
        fail_missing_ds004504(f"participants.tsv is missing: {participants_path}", raw_dataset_path)

    participants = load_ds004504_participants(participants_path)
    eeg_paths = sorted(raw_dataset_path.glob(EEG_GLOB))
    if options.limit is not None:
        eeg_paths = eeg_paths[: options.limit]
    if not eeg_paths:
        fail_missing_ds004504(f"No EEG files matched: {raw_dataset_path / EEG_GLOB}", raw_dataset_path)

    if options.dry_run:
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
        rbp_channel, channel_names, labels, epoch_start_sec, file_record = process_one_ds004504_eeg_file(
            eeg_path,
            participants=participants,
            bands=BANDS,
            total_power_range_hz=TOTAL_POWER_RANGE_HZ,
            epoch_sec=EPOCH_SEC,
            overlap=OVERLAP,
            welch_window_sec=WELCH_WINDOW_SEC,
            welch_overlap=WELCH_OVERLAP,
        )
        if reference_channels is None:
            reference_channels = channel_names
        elif channel_names != reference_channels:
            raise ValueError(f"Channel names differ for {eeg_path}")

        all_rbp_channel.append(rbp_channel)
        all_labels.append(labels)
        all_subject_ids.extend([str(file_record["subject_id"])] * len(labels))
        all_epoch_start_sec.extend(epoch_start_sec)
        file_records.append(file_record)

    if reference_channels is None:
        raise RuntimeError("No files were processed.")

    x_rbp_channel = np.concatenate(all_rbp_channel, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    epoch_start_sec = np.asarray(all_epoch_start_sec, dtype=np.float32)
    band_names = list(BANDS.keys())

    write_ds004504_rbp_h5(
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
        "output_h5": str(output_path),
        "dataset": {
            "id": PROCESSED_DATASET_ID,
            "raw_dataset_id": OPENNEURO_DATASET_ID,
            "path": str(options.raw_dir),
            "participants_tsv": str(participants_path),
            "eeg_source": EEG_SOURCE,
            "eeg_glob": str(raw_dataset_path / EEG_GLOB),
        },
        "processing": {
            "method": "rbp_paper",
            "epoch_sec": EPOCH_SEC,
            "overlap": OVERLAP,
            "welch_window_sec": WELCH_WINDOW_SEC,
            "welch_overlap": WELCH_OVERLAP,
            "rbp_total_power_range_hz": TOTAL_POWER_RANGE_HZ,
            "rbp_bands": BANDS,
        },
        "h5_datasets": rbp_h5_dataset_schema(),
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
    write_ds004504_manifest_json(manifest_path, manifest)

    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")
    print(f"Shape X_rbp_channel: {tuple(x_rbp_channel.shape)}")

