"""Paper-style epoch-level splits for ds004504_rbp_paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PaperSplitIndices:
    train: npt.NDArray[np.int64]
    val: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }


def make_stratified_epoch_split(
    *,
    labels: npt.NDArray[np.integer],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> PaperSplitIndices:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}.")
    if labels.shape[0] <= 0:
        raise ValueError("labels must contain at least one sample.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must satisfy 0 < value < 1, got {train_fraction}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must satisfy 0 < value < 1, got {val_fraction}.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must satisfy 0 <= value < 1, got {test_fraction}.")
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-9:
        raise ValueError("train_fraction, val_fraction, and test_fraction must sum to 1.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(int(value) for value in np.unique(labels)):
        class_indices = np.flatnonzero(labels == label).astype(np.int64)
        required_partitions = 3 if test_fraction > 0.0 else 2
        if class_indices.shape[0] < required_partitions:
            raise ValueError(
                f"Class {label} has fewer than {required_partitions} samples; cannot create requested split."
            )

        shuffled = rng.permutation(class_indices)
        train_count = int(shuffled.shape[0] * train_fraction)
        if test_fraction == 0.0:
            val_count = shuffled.shape[0] - train_count
            test_count = 0
        else:
            val_count = int(shuffled.shape[0] * val_fraction)
            test_count = shuffled.shape[0] - train_count - val_count
        if train_count <= 0 or val_count <= 0 or (test_fraction > 0.0 and test_count <= 0):
            raise ValueError(f"Class {label} split produced an empty partition.")

        train_indices.extend(int(value) for value in shuffled[:train_count])
        val_indices.extend(int(value) for value in shuffled[train_count : train_count + val_count])
        test_indices.extend(int(value) for value in shuffled[train_count + val_count :])

    return PaperSplitIndices(
        train=np.asarray(rng.permutation(train_indices), dtype=np.int64),
        val=np.asarray(rng.permutation(val_indices), dtype=np.int64),
        test=np.asarray(rng.permutation(test_indices), dtype=np.int64),
    )
