import csv
import importlib.util
from pathlib import Path
from types import ModuleType


def load_train_module() -> ModuleType:
    script_path = Path("scripts/ds004504_rbp_paper/train.py")
    spec = importlib.util.spec_from_file_location("ds004504_rbp_paper_train", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_training_config_merges_extends() -> None:
    train_module = load_train_module()

    config = train_module.load_training_config(Path("cfgs/ds004504_rbp_paper/multiclass.yaml"), seen=frozenset())

    assert config["experiment_group"] == "ds004504_rbp_paper"
    assert config["experiment_id"] == "ds004504_rbp_paper_multiclass"
    assert config["dataset"]["h5_path"] == "data/processed_raw_dataset/ds004504_rbp_paper.h5"
    assert config["dataset"]["manifest_path"] == "data/processed_raw_dataset/ds004504_rbp_paper_manifest.json"
    assert config["task"]["id"] == "multiclass"
    assert config["training"]["epochs"] == 100
    assert config["training"]["seed"] is None
    assert config["training"]["num_workers"] == 4
    assert config["split"] == {"train_fraction": 0.8, "val_fraction": 0.1, "test_fraction": 0.1}
    assert config["evaluation"]["test_source"] == "split"
    assert config["runtime"]["device"] == "cuda"
    assert config["output"]["dir"] == "data/runs/ds004504_rbp_paper/multiclass"
    assert config["output"]["overwrite"] is False


def test_task_configs_override_base_output_dir() -> None:
    train_module = load_train_module()

    config = train_module.load_training_config(Path("cfgs/ds004504_rbp_paper/ad_vs_healthy.yaml"), seen=frozenset())

    assert config["task"]["id"] == "ad_vs_healthy"
    assert config["output"]["dir"] == "data/runs/ds004504_rbp_paper/ad_vs_healthy"
    assert config["output"]["overwrite"] is False


def test_val_as_test_80_20_config_overrides_split_and_output_dir() -> None:
    train_module = load_train_module()

    config = train_module.load_training_config(
        Path("cfgs/ds004504_rbp_paper/val_as_test_80_20/multiclass.yaml"),
        seen=frozenset(),
    )

    assert config["experiment_id"] == "ds004504_rbp_paper_val_as_test_80_20_multiclass"
    assert config["task"]["id"] == "multiclass"
    assert config["split"] == {"train_fraction": 0.8, "val_fraction": 0.2, "test_fraction": 0.0}
    assert config["evaluation"]["test_source"] == "val"
    assert config["output"]["dir"] == "data/runs/ds004504_rbp_paper/val_as_test_80_20/multiclass"
    assert config["output"]["overwrite"] is False


def test_write_predictions_csv_includes_scores_and_source_metadata(tmp_path: Path) -> None:
    train_module = load_train_module()
    output_path = tmp_path / "predictions.csv"

    train_module.write_predictions_csv(
        output_path,
        source_indices=(10,),
        subject_ids=("sub-001",),
        epoch_start_sec=(3.0,),
        y_true=(1,),
        y_pred=(0,),
        logits=((2.0, 1.0),),
        probabilities=((0.75, 0.25),),
        class_names=("alzheimer", "healthy_control"),
    )

    with output_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "index": "0",
            "source_index": "10",
            "subject_id": "sub-001",
            "epoch_start_sec": "3.0",
            "y_true": "1",
            "y_pred": "0",
            "logit_alzheimer": "2.0",
            "logit_healthy_control": "1.0",
            "prob_alzheimer": "0.75",
            "prob_healthy_control": "0.25",
        }
    ]
