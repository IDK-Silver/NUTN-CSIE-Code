from pathlib import Path

import h5py
import numpy as np
import pytest

from ecg.datasets import H5FeatureLabelReader


def write_h5(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X", data=features)
        h5.create_dataset("y", data=labels)


def test_h5_feature_label_reader_reads_aligned_rows(tmp_path: Path) -> None:
    h5_path = tmp_path / "features.h5"
    write_h5(
        h5_path,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
    )

    reader = H5FeatureLabelReader(h5_path, feature_key="X", label_key="y")

    assert len(reader) == 2
    assert reader.feature_shape == (2,)
    assert reader.label_shape == ()

    feature, label = reader.read_pair(1)
    np.testing.assert_array_equal(feature, np.asarray([3.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(label, np.asarray(1, dtype=np.int64))

    reader.close()


def test_h5_feature_label_reader_rejects_missing_key(tmp_path: Path) -> None:
    h5_path = tmp_path / "features.h5"
    write_h5(h5_path, np.asarray([[1.0]], dtype=np.float32), np.asarray([0], dtype=np.int64))

    with pytest.raises(KeyError, match="feature dataset is missing"):
        H5FeatureLabelReader(h5_path, feature_key="missing", label_key="y")


def test_h5_feature_label_reader_rejects_mismatched_lengths(tmp_path: Path) -> None:
    h5_path = tmp_path / "features.h5"
    write_h5(
        h5_path,
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="feature and label lengths differ"):
        H5FeatureLabelReader(h5_path, feature_key="X", label_key="y")


def test_h5_feature_label_reader_rejects_out_of_range_index(tmp_path: Path) -> None:
    h5_path = tmp_path / "features.h5"
    write_h5(h5_path, np.asarray([[1.0]], dtype=np.float32), np.asarray([0], dtype=np.int64))
    reader = H5FeatureLabelReader(h5_path, feature_key="X", label_key="y")

    with pytest.raises(IndexError, match="sample index out of range"):
        reader.read_pair(1)

    reader.close()
