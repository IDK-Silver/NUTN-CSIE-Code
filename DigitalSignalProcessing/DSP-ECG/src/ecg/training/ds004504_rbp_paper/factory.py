"""Experiment factory for ds004504_rbp_paper reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from ecg.datasets import Ds004504RbpPaperDataset
from ecg.models import TcnLstmClassifier
from ecg.training.ds004504_rbp_paper.normalization import MinMaxParams, fit_paper_min_max, transform_min_max
from ecg.training.ds004504_rbp_paper.splits import PaperSplitIndices, make_stratified_epoch_split
from ecg.training.ds004504_rbp_paper.tasks import PaperClassificationTask, apply_task_to_labels, get_paper_classification_task


PAPER_TCN_CHANNELS = 32
PAPER_TCN_KERNEL_SIZE = 7
PAPER_TCN_DILATIONS = (1, 1)
PAPER_TCN_DROPOUT = 0.3
PAPER_LSTM_HIDDEN_DIM = 64
PAPER_DENSE_HIDDEN_DIMS = (128, 192, 256)
PAPER_DENSE_DROPOUT = 0.2
PAPER_BATCH_SIZE = 32
PAPER_LEARNING_RATE = 0.0001


@dataclass(frozen=True)
class PaperHyperparameters:
    input_dim: int
    num_classes: int
    tcn_channels: int
    tcn_kernel_size: int
    tcn_dilations: tuple[int, ...]
    tcn_dropout: float
    lstm_hidden_dim: int
    dense_hidden_dims: tuple[int, ...]
    dense_dropout: float
    batch_size: int
    learning_rate: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "tcn_channels": self.tcn_channels,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_dilations": list(self.tcn_dilations),
            "tcn_dropout": self.tcn_dropout,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "dense_hidden_dims": list(self.dense_hidden_dims),
            "dense_dropout": self.dense_dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True)
class PaperExperiment:
    task: PaperClassificationTask
    hyperparameters: PaperHyperparameters
    normalization: MinMaxParams
    split: PaperSplitIndices
    selected_source_indices: npt.NDArray[np.int64]
    selected_subject_ids: tuple[str, ...]
    selected_epoch_start_sec: npt.NDArray[np.float32]
    class_names: tuple[str, ...]
    model: nn.Module
    train_loader: DataLoader[tuple[Tensor, Tensor]]
    val_loader: DataLoader[tuple[Tensor, Tensor]]
    test_loader: DataLoader[tuple[Tensor, Tensor]]


class TensorPairDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y lengths differ: {x.shape[0]} != {y.shape[0]}.")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]


def load_ds004504_rbp_paper_arrays(
    *,
    h5_path: Path,
    manifest_path: Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], tuple[str, ...], tuple[str, ...], npt.NDArray[np.float32]]:
    metadata_dataset = Ds004504RbpPaperDataset(h5_path=h5_path, manifest_path=manifest_path)
    try:
        with h5py.File(h5_path, "r") as h5:
            features = np.asarray(h5["X_rbp_mean"], dtype=np.float32)
            labels = np.asarray(h5["y"], dtype=np.int64)
            subject_id_dataset = h5["subject_id"]
            epoch_start_sec_dataset = h5["epoch_start_sec"]
            if not isinstance(subject_id_dataset, h5py.Dataset):
                raise TypeError("H5 subject_id key is not a dataset.")
            if not isinstance(epoch_start_sec_dataset, h5py.Dataset):
                raise TypeError("H5 epoch_start_sec key is not a dataset.")
            subject_ids = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in subject_id_dataset[:]
            )
            epoch_start_sec = np.asarray(epoch_start_sec_dataset[:], dtype=np.float32)
    finally:
        metadata_dataset.reader.close()

    if features.ndim != 2:
        raise ValueError(f"X_rbp_mean must have shape [n_epochs, n_bands], got {features.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"y must have shape [n_epochs], got {labels.shape}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"X_rbp_mean and y lengths differ: {features.shape[0]} != {labels.shape[0]}.")
    if len(subject_ids) != features.shape[0]:
        raise ValueError(f"subject_id length {len(subject_ids)} does not match X_rbp_mean length {features.shape[0]}.")
    if epoch_start_sec.ndim != 1:
        raise ValueError(f"epoch_start_sec must have shape [n_epochs], got {epoch_start_sec.shape}.")
    if epoch_start_sec.shape[0] != features.shape[0]:
        raise ValueError(
            f"epoch_start_sec length {epoch_start_sec.shape[0]} does not match X_rbp_mean length {features.shape[0]}."
        )

    return features, labels, metadata_dataset.band_names, subject_ids, epoch_start_sec


def build_paper_hyperparameters(*, num_classes: int) -> PaperHyperparameters:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")

    return PaperHyperparameters(
        input_dim=1,
        num_classes=num_classes,
        tcn_channels=PAPER_TCN_CHANNELS,
        tcn_kernel_size=PAPER_TCN_KERNEL_SIZE,
        tcn_dilations=PAPER_TCN_DILATIONS,
        tcn_dropout=PAPER_TCN_DROPOUT,
        lstm_hidden_dim=PAPER_LSTM_HIDDEN_DIM,
        dense_hidden_dims=PAPER_DENSE_HIDDEN_DIMS,
        dense_dropout=PAPER_DENSE_DROPOUT,
        batch_size=PAPER_BATCH_SIZE,
        learning_rate=PAPER_LEARNING_RATE,
    )


def build_paper_model(*, hyperparameters: PaperHyperparameters) -> nn.Module:
    return TcnLstmClassifier(
        input_dim=hyperparameters.input_dim,
        num_classes=hyperparameters.num_classes,
        tcn_channels=hyperparameters.tcn_channels,
        tcn_kernel_size=hyperparameters.tcn_kernel_size,
        tcn_dilations=hyperparameters.tcn_dilations,
        tcn_dropout=hyperparameters.tcn_dropout,
        lstm_hidden_dim=hyperparameters.lstm_hidden_dim,
        dense_hidden_dims=hyperparameters.dense_hidden_dims,
        dense_dropout=hyperparameters.dense_dropout,
    )


def build_paper_experiment(
    *,
    h5_path: Path,
    manifest_path: Path,
    task_id: str,
    seed: int,
    num_workers: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> PaperExperiment:
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")

    task = get_paper_classification_task(task_id)
    source_features, source_labels, _, source_subject_ids, source_epoch_start_sec = load_ds004504_rbp_paper_arrays(
        h5_path=h5_path,
        manifest_path=manifest_path,
    )
    selected_source_indices, task_labels = apply_task_to_labels(source_labels, task)
    task_features = source_features[selected_source_indices]
    selected_subject_ids = tuple(source_subject_ids[int(index)] for index in selected_source_indices)
    selected_epoch_start_sec = source_epoch_start_sec[selected_source_indices]

    normalization = fit_paper_min_max(task_features)
    normalized_features = transform_min_max(task_features, normalization)

    split = make_stratified_epoch_split(
        labels=task_labels,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    x = Tensor(normalized_features[:, :, np.newaxis])
    y = Tensor(task_labels).long()
    tensor_dataset = TensorPairDataset(x, y)

    hyperparameters = build_paper_hyperparameters(num_classes=len(task.class_names))
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        Subset(tensor_dataset, split.train.tolist()),
        batch_size=hyperparameters.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        Subset(tensor_dataset, split.val.tolist()),
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        Subset(tensor_dataset, split.test.tolist()),
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return PaperExperiment(
        task=task,
        hyperparameters=hyperparameters,
        normalization=normalization,
        split=split,
        selected_source_indices=selected_source_indices,
        selected_subject_ids=selected_subject_ids,
        selected_epoch_start_sec=selected_epoch_start_sec,
        class_names=task.class_names,
        model=build_paper_model(hyperparameters=hyperparameters),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
