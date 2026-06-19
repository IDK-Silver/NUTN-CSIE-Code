"""Simple explicit SMOTE helper for paper-style reproduction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SimpleSmoteResult:
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    anchor_indices: npt.NDArray[np.int64]
    neighbor_indices: npt.NDArray[np.int64]
    synthetic_mask: npt.NDArray[np.bool_]

    def class_counts(self) -> dict[int, int]:
        return {int(label): int((self.labels == label).sum()) for label in sorted(np.unique(self.labels))}


def make_max_class_target_counts(labels: npt.NDArray[np.integer]) -> dict[int, int]:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}.")
    if labels.shape[0] <= 0:
        raise ValueError("labels must contain at least one sample.")
    counts = {int(label): int((labels == label).sum()) for label in sorted(np.unique(labels))}
    target_count = max(counts.values())
    return {label: target_count for label in counts}


def balance_with_simple_smote(
    *,
    features: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    target_count_by_class: Mapping[int, int],
    k_neighbors: int,
    seed: int,
) -> SimpleSmoteResult:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [n_samples, n_features], got {features.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"features and labels lengths differ: {features.shape[0]} != {labels.shape[0]}.")
    if features.shape[0] <= 0:
        raise ValueError("features must contain at least one sample.")
    if features.shape[1] <= 0:
        raise ValueError("features must contain at least one feature.")
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be positive, got {k_neighbors}.")
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must satisfy 0 <= value < 2**32, got {seed}.")

    labels_int = np.asarray(labels, dtype=np.int64)
    features_float = np.asarray(features, dtype=np.float32)
    observed_labels = {int(label) for label in np.unique(labels_int)}
    target_labels = {int(label) for label in target_count_by_class}
    if observed_labels != target_labels:
        raise ValueError(f"target_count_by_class labels {sorted(target_labels)} do not match observed labels {sorted(observed_labels)}.")

    rng = np.random.default_rng(seed)
    feature_rows: list[npt.NDArray[np.float32]] = [features_float]
    label_rows: list[npt.NDArray[np.int64]] = [labels_int]
    anchor_rows: list[npt.NDArray[np.int64]] = [np.arange(labels_int.shape[0], dtype=np.int64)]
    neighbor_rows: list[npt.NDArray[np.int64]] = [np.full(labels_int.shape[0], -1, dtype=np.int64)]
    synthetic_rows: list[npt.NDArray[np.bool_]] = [np.zeros(labels_int.shape[0], dtype=np.bool_)]

    for label in sorted(observed_labels):
        class_indices = np.flatnonzero(labels_int == label).astype(np.int64)
        current_count = int(class_indices.shape[0])
        target_count = int(target_count_by_class[label])
        if target_count < current_count:
            raise ValueError(
                f"target_count_by_class[{label}]={target_count} is smaller than current count {current_count}."
            )
        required_count = target_count - current_count
        if required_count == 0:
            continue
        if current_count <= 1:
            raise ValueError(f"Class {label} has only {current_count} sample; cannot synthesize SMOTE neighbors.")

        class_features = features_float[class_indices]
        effective_k = min(k_neighbors, current_count - 1)
        differences = class_features[:, np.newaxis, :] - class_features[np.newaxis, :, :]
        distances = np.sum(differences * differences, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest_positions = np.argsort(distances, axis=1)[:, :effective_k]

        synthetic_features = np.empty((required_count, features_float.shape[1]), dtype=np.float32)
        synthetic_labels = np.full(required_count, label, dtype=np.int64)
        synthetic_anchors = np.empty(required_count, dtype=np.int64)
        synthetic_neighbors = np.empty(required_count, dtype=np.int64)
        for synthetic_index in range(required_count):
            anchor_position = int(rng.integers(0, current_count))
            neighbor_position = int(rng.choice(nearest_positions[anchor_position]))
            interpolation = float(rng.random())
            anchor_feature = class_features[anchor_position]
            neighbor_feature = class_features[neighbor_position]
            synthetic_features[synthetic_index] = anchor_feature + interpolation * (neighbor_feature - anchor_feature)
            synthetic_anchors[synthetic_index] = int(class_indices[anchor_position])
            synthetic_neighbors[synthetic_index] = int(class_indices[neighbor_position])

        feature_rows.append(synthetic_features)
        label_rows.append(synthetic_labels)
        anchor_rows.append(synthetic_anchors)
        neighbor_rows.append(synthetic_neighbors)
        synthetic_rows.append(np.ones(required_count, dtype=np.bool_))

    balanced_features = np.concatenate(feature_rows, axis=0)
    balanced_labels = np.concatenate(label_rows, axis=0)
    anchor_indices = np.concatenate(anchor_rows, axis=0)
    neighbor_indices = np.concatenate(neighbor_rows, axis=0)
    synthetic_mask = np.concatenate(synthetic_rows, axis=0)
    permutation = rng.permutation(balanced_labels.shape[0])

    return SimpleSmoteResult(
        features=np.asarray(balanced_features[permutation], dtype=np.float32),
        labels=np.asarray(balanced_labels[permutation], dtype=np.int64),
        anchor_indices=np.asarray(anchor_indices[permutation], dtype=np.int64),
        neighbor_indices=np.asarray(neighbor_indices[permutation], dtype=np.int64),
        synthetic_mask=np.asarray(synthetic_mask[permutation], dtype=np.bool_),
    )
