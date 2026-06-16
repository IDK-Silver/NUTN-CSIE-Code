from __future__ import annotations

import argparse
import csv
import json
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Adam

from ecg.training import compute_classification_metrics, evaluate_one_epoch, train_one_epoch
from ecg.training.ds004504_rbp_paper import build_paper_experiment


JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ds004504_rbp_paper reproduction config.")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def merge_config(base: JsonObject, override: JsonObject) -> JsonObject:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml_mapping(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return dict(payload)


def load_training_config(path: Path, *, seen: frozenset[Path]) -> JsonObject:
    resolved_path = path.resolve()
    if resolved_path in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved_path))
        raise ValueError(f"YAML extends cycle detected: {chain}")
    config = read_yaml_mapping(resolved_path)

    extends = config.pop("extends", [])
    if isinstance(extends, str):
        raise ValueError(f"extends must be a list, got string in {resolved_path}.")
    if not isinstance(extends, list):
        raise ValueError(f"extends must be a list in {resolved_path}.")

    merged: JsonObject = {}
    for parent in extends:
        if not isinstance(parent, str):
            raise ValueError(f"extends entries must be strings in {resolved_path}.")
        parent_path = (resolved_path.parent / parent).resolve()
        parent_config = load_training_config(parent_path, seen=seen | {resolved_path})
        merged = merge_config(merged, parent_config)

    return merge_config(merged, config)


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be a mapping.")
    return value


def require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config field {key!r} must be a non-empty string.")
    return value


def require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Config field {key!r} must be an integer.")
    return value


def require_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Config field {key!r} must be a boolean.")
    return value


def require_path(config: Mapping[str, Any], key: str) -> Path:
    return Path(require_str(config, key))


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_predictions_csv(
    path: Path,
    *,
    source_indices: tuple[int, ...],
    subject_ids: tuple[str, ...],
    epoch_start_sec: tuple[float, ...],
    y_true: tuple[int, ...],
    y_pred: tuple[int, ...],
    logits: tuple[tuple[float, ...], ...],
    probabilities: tuple[tuple[float, ...], ...],
    class_names: tuple[str, ...],
) -> None:
    sample_count = len(y_true)
    if not (
        len(source_indices)
        == len(subject_ids)
        == len(epoch_start_sec)
        == len(y_pred)
        == len(logits)
        == len(probabilities)
        == sample_count
    ):
        raise ValueError("Prediction CSV inputs must have the same sample count.")

    fieldnames = ["index", "source_index", "subject_id", "epoch_start_sec", "y_true", "y_pred"]
    fieldnames.extend(f"logit_{class_name}" for class_name in class_names)
    fieldnames.extend(f"prob_{class_name}" for class_name in class_names)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, (
            source_index,
            subject_id,
            epoch_start,
            true_label,
            pred_label,
            sample_logits,
            sample_probabilities,
        ) in enumerate(
            zip(source_indices, subject_ids, epoch_start_sec, y_true, y_pred, logits, probabilities, strict=True)
        ):
            if len(sample_logits) != len(class_names):
                raise ValueError(f"Logit width {len(sample_logits)} does not match class count {len(class_names)}.")
            if len(sample_probabilities) != len(class_names):
                raise ValueError(
                    f"Probability width {len(sample_probabilities)} does not match class count {len(class_names)}."
                )

            row: dict[str, object] = {
                "index": index,
                "source_index": source_index,
                "subject_id": subject_id,
                "epoch_start_sec": epoch_start,
                "y_true": true_label,
                "y_pred": pred_label,
            }
            for class_name, value in zip(class_names, sample_logits, strict=True):
                row[f"logit_{class_name}"] = value
            for class_name, value in zip(class_names, sample_probabilities, strict=True):
                row[f"prob_{class_name}"] = value
            writer.writerow(row)


def run_training(config_path: Path) -> None:
    config = load_training_config(config_path, seen=frozenset())

    experiment_id = require_str(config, "experiment_id")
    dataset_config = require_mapping(config, "dataset")
    task_config = require_mapping(config, "task")
    training_config = require_mapping(config, "training")
    split_config = require_mapping(config, "split")
    evaluation_config = require_mapping(config, "evaluation")
    runtime_config = require_mapping(config, "runtime")
    output_config = require_mapping(config, "output")

    h5_path = require_path(dataset_config, "h5_path")
    manifest_path = require_path(dataset_config, "manifest_path")
    task_id = require_str(task_config, "id")
    epochs = require_int(training_config, "epochs")
    requested_seed = training_config.get("seed")
    if requested_seed is None:
        seed = secrets.randbelow(2**32)
    elif isinstance(requested_seed, int) and not isinstance(requested_seed, bool):
        seed = requested_seed
    else:
        raise ValueError("Config field 'training.seed' must be an integer or null.")
    num_workers = require_int(training_config, "num_workers")
    split_values: dict[str, float] = {}
    for key in ("train_fraction", "val_fraction", "test_fraction"):
        value = split_config.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"Config field 'split.{key}' must be a number.")
        split_values[key] = float(value)
    train_fraction = split_values["train_fraction"]
    val_fraction = split_values["val_fraction"]
    test_fraction = split_values["test_fraction"]
    test_source = require_str(evaluation_config, "test_source")
    device = require_str(runtime_config, "device")
    output_dir = require_path(output_config, "dir")
    overwrite = require_bool(output_config, "overwrite")

    if epochs <= 0:
        raise ValueError(f"training.epochs must be positive, got {epochs}.")
    if not 0 <= seed < 2**32:
        raise ValueError(f"training.seed must satisfy 0 <= value < 2**32, got {seed}.")
    if num_workers < 0:
        raise ValueError(f"training.num_workers must be non-negative, got {num_workers}.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"split.train_fraction must satisfy 0 < value < 1, got {train_fraction}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"split.val_fraction must satisfy 0 < value < 1, got {val_fraction}.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"split.test_fraction must satisfy 0 <= value < 1, got {test_fraction}.")
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-9:
        raise ValueError("split.train_fraction, split.val_fraction, and split.test_fraction must sum to 1.")
    if test_source not in {"split", "val", "none"}:
        raise ValueError("evaluation.test_source must be one of: split, val, none.")
    if test_source == "split" and test_fraction <= 0.0:
        raise ValueError("evaluation.test_source='split' requires split.test_fraction > 0.")
    if test_source in {"val", "none"} and test_fraction != 0.0:
        raise ValueError(f"evaluation.test_source={test_source!r} requires split.test_fraction == 0.0.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    experiment = build_paper_experiment(
        h5_path=h5_path,
        manifest_path=manifest_path,
        task_id=task_id,
        seed=seed,
        num_workers=num_workers,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    model = experiment.model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=experiment.hyperparameters.learning_rate)

    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_output = train_one_epoch(
            model=model,
            batches=experiment.train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_output = evaluate_one_epoch(
            model=model,
            batches=experiment.val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        train_metrics = compute_classification_metrics(
            y_true=train_output.y_true,
            y_pred=train_output.y_pred,
            class_names=experiment.class_names,
        )
        val_metrics = compute_classification_metrics(
            y_true=val_output.y_true,
            y_pred=val_output.y_pred,
            class_names=experiment.class_names,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_output.loss,
                "val_loss": val_output.loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_output.loss:.6f} "
            f"train_acc={train_metrics['accuracy']:.6f} "
            f"val_loss={val_output.loss:.6f} "
            f"val_acc={val_metrics['accuracy']:.6f}"
        )

    if test_source == "split":
        final_batches = experiment.test_loader
        final_split_indices = tuple(int(value) for value in experiment.split.test.tolist())
    elif test_source == "val":
        final_batches = experiment.val_loader
        final_split_indices = tuple(int(value) for value in experiment.split.val.tolist())
    else:
        final_batches = None
        final_split_indices = None

    final_output = None
    final_metrics = None
    if final_batches is not None:
        final_output = evaluate_one_epoch(
            model=model,
            batches=final_batches,
            loss_fn=loss_fn,
            device=device,
        )
        final_metrics = compute_classification_metrics(
            y_true=final_output.y_true,
            y_pred=final_output.y_pred,
            class_names=experiment.class_names,
        )

    write_json(output_dir / "history.json", history)
    if final_output is not None and final_metrics is not None:
        if final_split_indices is None:
            raise ValueError("Final split indices are missing for prediction export.")
        source_indices = tuple(int(experiment.selected_source_indices[index]) for index in final_split_indices)
        subject_ids = tuple(experiment.selected_subject_ids[index] for index in final_split_indices)
        epoch_start_sec = tuple(float(experiment.selected_epoch_start_sec[index]) for index in final_split_indices)
        write_json(
            output_dir / "test_metrics.json",
            {"source": test_source, "loss": final_output.loss, "metrics": final_metrics},
        )
        write_predictions_csv(
            output_dir / "test_predictions.csv",
            source_indices=source_indices,
            subject_ids=subject_ids,
            epoch_start_sec=epoch_start_sec,
            y_true=final_output.y_true,
            y_pred=final_output.y_pred,
            logits=final_output.logits,
            probabilities=final_output.probabilities,
            class_names=experiment.class_names,
        )
    else:
        for filename in ("test_metrics.json", "test_predictions.csv"):
            stale_path = output_dir / filename
            if stale_path.exists():
                stale_path.unlink()
    write_json(
        output_dir / "run.json",
        {
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "merged_config": config,
            "task_id": task_id,
            "h5_path": str(h5_path),
            "manifest_path": str(manifest_path),
            "epochs": epochs,
            "seed": seed,
            "requested_seed": requested_seed,
            "resolved_seed": seed,
            "device": device,
            "num_workers": num_workers,
            "protocol": (
                "ds004504_rbp_paper reproduction: paper-style full-task min-max normalization "
                "before epoch-level split."
            ),
            "class_names": list(experiment.class_names),
            "hyperparameters": experiment.hyperparameters.to_json_dict(),
            "normalization": experiment.normalization.to_json_dict(),
            "split_fractions": {
                "train": train_fraction,
                "val": val_fraction,
                "test": test_fraction,
            },
            "split": experiment.split.to_json_dict(),
            "evaluation": {"test_source": test_source},
            "selected_source_indices": experiment.selected_source_indices.tolist(),
        },
    )
    torch.save(
        {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "model_state_dict": model.state_dict(),
            "hyperparameters": experiment.hyperparameters.to_json_dict(),
            "class_names": list(experiment.class_names),
        },
        output_dir / "model.pt",
    )

    if final_output is not None and final_metrics is not None:
        print(
            f"test_source={test_source} "
            f"test_loss={final_output.loss:.6f} "
            f"test_acc={final_metrics['accuracy']:.6f}"
        )
    else:
        print("test_source=none; skipped final test metrics")
    print(f"Wrote {output_dir}")


def main() -> None:
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
