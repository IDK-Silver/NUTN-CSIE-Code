import json
from pathlib import Path

import h5py
import numpy as np

from ecg.training.ds004504_rbp_paper import build_paper_experiment


def write_paper_cache(tmp_path: Path) -> tuple[Path, Path]:
    h5_path = tmp_path / "ds004504_rbp_paper.h5"
    manifest_path = tmp_path / "ds004504_rbp_paper_manifest.json"
    labels = np.asarray([0] * 12 + [1] * 12 + [2] * 12, dtype=np.int64)
    features = np.arange(labels.shape[0] * 6, dtype=np.float32).reshape(labels.shape[0], 6)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("X_rbp_mean", data=features)
        h5.create_dataset("y", data=labels)
        h5.create_dataset(
            "band_names",
            data=np.asarray(["delta", "theta", "alpha", "zaeta", "beta", "gamma"], dtype=h5py.string_dtype("utf-8")),
        )
        h5.create_dataset(
            "subject_id",
            data=np.asarray([f"sub-{index:03d}" for index in range(labels.shape[0])], dtype=h5py.string_dtype("utf-8")),
        )
        h5.create_dataset("epoch_start_sec", data=np.arange(labels.shape[0], dtype=np.float32) * 3.0)

    manifest_path.write_text(
        json.dumps(
            {
                "dataset": {"id": "ds004504_rbp_paper"},
                "h5_datasets": {"X_rbp_mean": {}, "y": {}},
                "label_map": {"A": 0, "F": 1, "C": 2},
            }
        ),
        encoding="utf-8",
    )
    return h5_path, manifest_path


def test_build_paper_experiment_assembles_multiclass_training_objects(tmp_path: Path) -> None:
    h5_path, manifest_path = write_paper_cache(tmp_path)

    experiment = build_paper_experiment(
        h5_path=h5_path,
        manifest_path=manifest_path,
        task_id="multiclass",
        seed=7,
        num_workers=0,
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    assert experiment.class_names == ("alzheimer", "frontotemporal_dementia", "healthy_control")
    assert experiment.hyperparameters.batch_size == 32
    assert experiment.hyperparameters.learning_rate == 0.0001
    assert sum(parameter.numel() for parameter in experiment.model.parameters()) == 131587
    assert len(experiment.train_loader.dataset) == 27
    assert len(experiment.val_loader.dataset) == 3
    assert len(experiment.test_loader.dataset) == 6
    assert len(experiment.selected_subject_ids) == 36
    assert experiment.selected_epoch_start_sec.shape == (36,)

    x, y = next(iter(experiment.train_loader))
    assert tuple(x.shape[1:]) == (6, 1)
    assert y.ndim == 1


def test_build_paper_experiment_allows_no_test_split(tmp_path: Path) -> None:
    h5_path, manifest_path = write_paper_cache(tmp_path)

    experiment = build_paper_experiment(
        h5_path=h5_path,
        manifest_path=manifest_path,
        task_id="multiclass",
        seed=7,
        num_workers=0,
        train_fraction=0.8,
        val_fraction=0.2,
        test_fraction=0.0,
    )

    assert len(experiment.train_loader.dataset) == 27
    assert len(experiment.val_loader.dataset) == 9
    assert len(experiment.test_loader.dataset) == 0
