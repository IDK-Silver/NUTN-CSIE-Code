import json
from pathlib import Path

import h5py
import numpy as np

from ecg.datasets import Ds004504RbpPaperDataset


def write_ds004504_rbp_paper_fixture(tmp_path: Path, *, dataset_id: str) -> tuple[Path, Path]:
    h5_path = tmp_path / "ds004504_rbp_paper.h5"
    manifest_path = tmp_path / "ds004504_rbp_paper_manifest.json"

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(
            "X_rbp_mean",
            data=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        )
        h5.create_dataset("y", data=np.asarray([0, 2], dtype=np.int64))
        h5.create_dataset(
            "band_names",
            data=np.asarray(["delta", "theta", "alpha"], dtype=h5py.string_dtype("utf-8")),
        )

    manifest_path.write_text(
        json.dumps(
            {
                "dataset": {"id": dataset_id},
                "h5_datasets": {"X_rbp_mean": {}, "y": {}},
                "label_map": {"A": 0, "F": 1, "C": 2},
            }
        ),
        encoding="utf-8",
    )
    return h5_path, manifest_path


def test_ds004504_rbp_paper_dataset_returns_paper_input_shape(tmp_path: Path) -> None:
    h5_path, manifest_path = write_ds004504_rbp_paper_fixture(tmp_path, dataset_id="ds004504_rbp_paper")

    dataset = Ds004504RbpPaperDataset(h5_path=h5_path, manifest_path=manifest_path)
    x, y = dataset[1]

    assert len(dataset) == 2
    assert tuple(x.shape) == (3, 1)
    assert tuple(y.shape) == ()
    assert str(x.dtype) == "torch.float32"
    assert str(y.dtype) == "torch.int64"
    assert int(y.item()) == 2
    assert dataset.band_names == ("delta", "theta", "alpha")
    assert dataset.sequence_length == 3
    assert dataset.input_dim == 1
    assert dataset.num_classes == 3
    dataset.reader.close()


def test_ds004504_rbp_paper_dataset_rejects_wrong_manifest_id(tmp_path: Path) -> None:
    h5_path, manifest_path = write_ds004504_rbp_paper_fixture(tmp_path, dataset_id="other_dataset")

    try:
        Ds004504RbpPaperDataset(h5_path=h5_path, manifest_path=manifest_path)
    except ValueError as exc:
        assert "Expected manifest dataset id 'ds004504_rbp_paper'" in str(exc)
    else:
        raise AssertionError("Expected wrong manifest id to fail.")
