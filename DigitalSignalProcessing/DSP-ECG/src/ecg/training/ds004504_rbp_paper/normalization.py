"""Paper-style min-max normalization for ds004504_rbp_paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MinMaxParams:
    feature_min: npt.NDArray[np.float32]
    feature_max: npt.NDArray[np.float32]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
        }


def fit_paper_min_max(features: npt.NDArray[np.floating]) -> MinMaxParams:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [n_samples, n_features], got {features.shape}.")
    if features.shape[0] <= 0:
        raise ValueError("features must contain at least one sample.")
    if features.shape[1] <= 0:
        raise ValueError("features must contain at least one feature.")

    feature_min = np.asarray(features.min(axis=0), dtype=np.float32)
    feature_max = np.asarray(features.max(axis=0), dtype=np.float32)
    if np.any(feature_max <= feature_min):
        raise ValueError("paper min-max normalization cannot handle constant feature columns.")

    return MinMaxParams(feature_min=feature_min, feature_max=feature_max)


def transform_min_max(
    features: npt.NDArray[np.floating],
    params: MinMaxParams,
) -> npt.NDArray[np.float32]:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [n_samples, n_features], got {features.shape}.")
    if features.shape[1] != params.feature_min.shape[0]:
        raise ValueError(f"features width {features.shape[1]} does not match min-max width {params.feature_min.shape[0]}.")
    denominator = params.feature_max - params.feature_min
    if np.any(denominator <= 0.0):
        raise ValueError("paper min-max normalization parameters contain non-positive denominators.")

    return np.asarray((features - params.feature_min) / denominator, dtype=np.float32)
