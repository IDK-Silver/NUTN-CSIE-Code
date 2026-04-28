"""Generic H5 readers for row-aligned feature and label datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt


class H5FeatureLabelReader:
    """Read row-aligned feature and label arrays from an H5 file."""

    def __init__(self, h5_path: Path, *, feature_key: str, label_key: str) -> None:
        self.h5_path = h5_path
        self.feature_key = feature_key
        self.label_key = label_key
        self._h5: h5py.File | None = None
        self._owner_pid: int | None = None

        if not feature_key:
            raise ValueError("feature_key must be a non-empty H5 dataset key.")
        if not label_key:
            raise ValueError("label_key must be a non-empty H5 dataset key.")
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file does not exist: {h5_path}")
        if not h5_path.is_file():
            raise FileNotFoundError(f"H5 path is not a file: {h5_path}")

        with h5py.File(h5_path, "r") as h5:
            if feature_key not in h5:
                raise KeyError(f"H5 feature dataset is missing: {feature_key}")
            if label_key not in h5:
                raise KeyError(f"H5 label dataset is missing: {label_key}")

            feature_dataset = h5[feature_key]
            label_dataset = h5[label_key]
            if not isinstance(feature_dataset, h5py.Dataset):
                raise TypeError(f"H5 feature key is not a dataset: {feature_key}")
            if not isinstance(label_dataset, h5py.Dataset):
                raise TypeError(f"H5 label key is not a dataset: {label_key}")
            if len(feature_dataset.shape) == 0:
                raise ValueError(f"H5 feature dataset must have a sample axis: {feature_key}")
            if len(label_dataset.shape) == 0:
                raise ValueError(f"H5 label dataset must have a sample axis: {label_key}")
            if feature_dataset.shape[0] != label_dataset.shape[0]:
                raise ValueError(
                    f"H5 feature and label lengths differ: {feature_key} has {feature_dataset.shape[0]}, "
                    f"{label_key} has {label_dataset.shape[0]}."
                )

            self.length = int(feature_dataset.shape[0])
            self.feature_shape = tuple(int(dim) for dim in feature_dataset.shape[1:])
            self.label_shape = tuple(int(dim) for dim in label_dataset.shape[1:])

    def __len__(self) -> int:
        return self.length

    def _open_h5(self) -> h5py.File:
        current_pid = os.getpid()
        if self._h5 is None or self._owner_pid != current_pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.h5_path, "r")
            self._owner_pid = current_pid
        return self._h5

    def read_pair(self, index: int) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        if index < 0 or index >= self.length:
            raise IndexError(f"H5 sample index out of range: {index}")

        h5 = self._open_h5()
        feature_dataset = h5[self.feature_key]
        label_dataset = h5[self.label_key]
        if not isinstance(feature_dataset, h5py.Dataset):
            raise TypeError(f"H5 feature key is not a dataset: {self.feature_key}")
        if not isinstance(label_dataset, h5py.Dataset):
            raise TypeError(f"H5 label key is not a dataset: {self.label_key}")
        feature = np.asarray(feature_dataset[index])
        label = np.asarray(label_dataset[index])

        return feature, label

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._owner_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        self.close()
