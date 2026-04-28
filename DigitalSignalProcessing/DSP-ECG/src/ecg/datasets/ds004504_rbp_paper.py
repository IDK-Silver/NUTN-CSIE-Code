"""PyTorch Dataset for the ds004504_rbp_paper H5 artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from ecg.data.ds004504_rbp_paper import PROCESSED_DATASET_ID
from ecg.datasets.h5 import H5FeatureLabelReader


class Ds004504RbpPaperDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load paper-style channel-averaged RBP features for ds004504."""

    def __init__(self, h5_path: Path, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest JSON does not exist: {manifest_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest path is not a file: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest: Any = json.load(f)
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest JSON must contain an object: {manifest_path}")

        dataset_record = manifest.get("dataset")
        if not isinstance(dataset_record, dict):
            raise ValueError("Manifest JSON is missing dataset metadata.")
        dataset_id = dataset_record.get("id")
        if dataset_id != PROCESSED_DATASET_ID:
            raise ValueError(f"Expected manifest dataset id {PROCESSED_DATASET_ID!r}, got {dataset_id!r}.")

        h5_datasets = manifest.get("h5_datasets")
        if not isinstance(h5_datasets, dict):
            raise ValueError("Manifest JSON is missing h5_datasets metadata.")
        if "X_rbp_mean" not in h5_datasets:
            raise ValueError("Manifest JSON does not describe X_rbp_mean.")
        if "y" not in h5_datasets:
            raise ValueError("Manifest JSON does not describe y.")

        label_map_record = manifest.get("label_map")
        if not isinstance(label_map_record, dict):
            raise ValueError("Manifest JSON is missing label_map metadata.")
        label_map: dict[str, int] = {}
        for label_name, label_value in label_map_record.items():
            if not isinstance(label_name, str):
                raise ValueError(f"Manifest label_map key must be str, got {label_name!r}.")
            if not isinstance(label_value, int):
                raise ValueError(f"Manifest label_map value must be int for {label_name!r}, got {label_value!r}.")
            label_map[label_name] = label_value

        reader = H5FeatureLabelReader(h5_path, feature_key="X_rbp_mean", label_key="y")
        if len(reader.feature_shape) != 1:
            raise ValueError(f"X_rbp_mean must have shape [n_epochs, n_bands], got sample shape {reader.feature_shape}.")
        if len(reader.label_shape) != 0:
            raise ValueError(f"y must have shape [n_epochs], got sample shape {reader.label_shape}.")

        with h5py.File(h5_path, "r") as h5:
            if "band_names" not in h5:
                raise KeyError("H5 dataset is missing band_names.")
            band_names_dataset = h5["band_names"]
            if not isinstance(band_names_dataset, h5py.Dataset):
                raise TypeError("H5 band_names key is not a dataset.")
            band_names = tuple(
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in band_names_dataset[:]
            )

        if len(band_names) != reader.feature_shape[0]:
            raise ValueError(f"band_names length {len(band_names)} does not match X_rbp_mean bands {reader.feature_shape[0]}.")

        self.reader = reader
        self.band_names = band_names
        self.label_map = label_map
        self.sequence_length = reader.feature_shape[0]
        self.input_dim = 1
        self.num_classes = len(set(label_map.values()))

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        feature, label = self.reader.read_pair(index)

        x = Tensor(np.asarray(feature, dtype=np.float32))
        if x.ndim != 1:
            raise ValueError(f"X_rbp_mean sample must be 1D [n_bands], got shape {tuple(x.shape)}.")

        if label.ndim != 0:
            raise ValueError(f"y sample must be scalar, got shape {tuple(label.shape)}.")
        y = Tensor([int(label.item())]).long().squeeze(0)

        return x.unsqueeze(-1), y
