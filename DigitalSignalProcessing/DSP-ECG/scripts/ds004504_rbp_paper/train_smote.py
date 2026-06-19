from __future__ import annotations

import argparse
import csv
import json
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ecg.training import compute_classification_metrics, evaluate_one_epoch, train_one_epoch
from ecg.training.ds004504_rbp_paper.factory import (
    TensorPairDataset,
    build_paper_hyperparameters,
    build_paper_model,
    load_ds004504_rbp_paper_arrays,
)
from ecg.training.ds004504_rbp_paper.normalization import fit_paper_min_max, transform_min_max
from ecg.training.ds004504_rbp_paper.smote import (
    SimpleSmoteResult,
    balance_with_simple_smote,
    make_max_class_target_counts,
)
from ecg.training.ds004504_rbp_paper.splits import make_stratified_epoch_split
from ecg.training.ds004504_rbp_paper.tasks import apply_task_to_labels, get_paper_classification_task
from train import (
    load_training_config,
    require_bool,
    require_int,
    require_mapping,
    require_path,
    require_str,
    write_json,
    write_predictions_csv,
)


JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper-style ds004504_rbp_paper SMOTE config.")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def resolve_seed(training_config: Mapping[str, Any]) -> tuple[int, int | None]:
    requested_seed = training_config.get("seed")
    if requested_seed is None:
        seed = secrets.randbelow(2**32)
    elif isinstance(requested_seed, int) and not isinstance(requested_seed, bool):
        seed = requested_seed
    else:
        raise ValueError("Config field 'training.seed' must be an integer or null.")
    if not 0 <= seed < 2**32:
        raise ValueError(f"training.seed must satisfy 0 <= value < 2**32, got {seed}.")
    return seed, requested_seed


def require_partition_names(config: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config field {key!r} must be a non-empty list.")
    partitions: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Config field {key!r} must contain only strings.")
        if item not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported partition name in {key!r}: {item!r}.")
        partitions.append(item)
    return tuple(partitions)


def require_float_fraction(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Config field {key!r} must be a number.")
    return float(value)


def target_counts_for_strategy(strategy: str, labels: np.ndarray) -> dict[int, int]:
    if strategy != "max_class_count":
        raise ValueError("smote.target_strategy must be 'max_class_count'.")
    return make_max_class_target_counts(labels)


def build_partition_smote_result(
    *,
    partition_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    subject_ids: Sequence[str],
    epoch_start_sec: np.ndarray,
    smote_partitions: tuple[str, ...],
    target_strategy: str,
    k_neighbors: int,
    seed: int,
) -> tuple[SimpleSmoteResult, tuple[int, ...], tuple[str, ...], tuple[float, ...], dict[str, object]]:
    if partition_name in smote_partitions:
        result = balance_with_simple_smote(
            features=features,
            labels=labels,
            target_count_by_class=target_counts_for_strategy(target_strategy, labels),
            k_neighbors=k_neighbors,
            seed=seed,
        )
    else:
        result = SimpleSmoteResult(
            features=np.asarray(features, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            anchor_indices=np.arange(labels.shape[0], dtype=np.int64),
            neighbor_indices=np.full(labels.shape[0], -1, dtype=np.int64),
            synthetic_mask=np.zeros(labels.shape[0], dtype=np.bool_),
        )

    resolved_source_indices = tuple(int(source_indices[int(index)]) for index in result.anchor_indices)
    resolved_subject_ids = tuple(subject_ids[int(index)] for index in result.anchor_indices)
    resolved_epoch_start_sec = tuple(float(epoch_start_sec[int(index)]) for index in result.anchor_indices)
    metadata = {
        "partition": partition_name,
        "input_count": int(labels.shape[0]),
        "output_count": int(result.labels.shape[0]),
        "synthetic_count": int(result.synthetic_mask.sum()),
        "class_counts": {str(label): count for label, count in result.class_counts().items()},
    }
    return result, resolved_source_indices, resolved_subject_ids, resolved_epoch_start_sec, metadata


def write_partition_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ("partition", "input_count", "output_count", "synthetic_count", "class_counts")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            next_row = dict(row)
            next_row["class_counts"] = json.dumps(next_row["class_counts"], ensure_ascii=False)
            writer.writerow(next_row)


def make_loader(result: SimpleSmoteResult, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    x = torch.as_tensor(result.features[:, :, np.newaxis], dtype=torch.float32)
    y = torch.as_tensor(result.labels, dtype=torch.long)
    return DataLoader(
        TensorPairDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def run_smote_training(config_path: Path) -> None:
    config = load_training_config(config_path, seen=frozenset())

    experiment_id = require_str(config, "experiment_id")
    dataset_config = require_mapping(config, "dataset")
    task_config = require_mapping(config, "task")
    training_config = require_mapping(config, "training")
    split_config = require_mapping(config, "split")
    evaluation_config = require_mapping(config, "evaluation")
    smote_config = require_mapping(config, "smote")
    runtime_config = require_mapping(config, "runtime")
    output_config = require_mapping(config, "output")

    h5_path = require_path(dataset_config, "h5_path")
    manifest_path = require_path(dataset_config, "manifest_path")
    task_id = require_str(task_config, "id")
    epochs = require_int(training_config, "epochs")
    seed, requested_seed = resolve_seed(training_config)
    num_workers = require_int(training_config, "num_workers")
    train_fraction = require_float_fraction(split_config, "train_fraction")
    val_fraction = require_float_fraction(split_config, "val_fraction")
    test_fraction = require_float_fraction(split_config, "test_fraction")
    test_source = require_str(evaluation_config, "test_source")
    target_strategy = require_str(smote_config, "target_strategy")
    k_neighbors = require_int(smote_config, "k_neighbors")
    smote_partitions = require_partition_names(smote_config, "apply_to_partitions")
    device = require_str(runtime_config, "device")
    output_dir = require_path(output_config, "dir")
    overwrite = require_bool(output_config, "overwrite")

    if epochs <= 0:
        raise ValueError(f"training.epochs must be positive, got {epochs}.")
    if num_workers < 0:
        raise ValueError(f"training.num_workers must be non-negative, got {num_workers}.")
    if k_neighbors <= 0:
        raise ValueError(f"smote.k_neighbors must be positive, got {k_neighbors}.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"split.train_fraction must satisfy 0 < value < 1, got {train_fraction}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"split.val_fraction must satisfy 0 < value < 1, got {val_fraction}.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"split.test_fraction must satisfy 0 <= value < 1, got {test_fraction}.")
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-9:
        raise ValueError("split.train_fraction, split.val_fraction, and split.test_fraction must sum to 1.")
    if test_source not in {"split", "val"}:
        raise ValueError("evaluation.test_source must be one of: split, val.")
    if test_source == "split" and test_fraction <= 0.0:
        raise ValueError("evaluation.test_source='split' requires split.test_fraction > 0.")
    if test_source == "val" and test_fraction != 0.0:
        raise ValueError("evaluation.test_source='val' requires split.test_fraction == 0.0.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    split_indices = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }
    partition_results: dict[str, SimpleSmoteResult] = {}
    partition_source_indices: dict[str, tuple[int, ...]] = {}
    partition_subject_ids: dict[str, tuple[str, ...]] = {}
    partition_epoch_start_sec: dict[str, tuple[float, ...]] = {}
    partition_summary_rows: list[dict[str, object]] = []
    for offset, partition_name in enumerate(("train", "val", "test")):
        indices = split_indices[partition_name]
        if indices.shape[0] == 0:
            continue
        result, source_indices, subject_ids, epoch_start_sec, summary = build_partition_smote_result(
            partition_name=partition_name,
            features=normalized_features[indices],
            labels=task_labels[indices],
            source_indices=selected_source_indices[indices],
            subject_ids=tuple(selected_subject_ids[int(index)] for index in indices),
            epoch_start_sec=selected_epoch_start_sec[indices],
            smote_partitions=smote_partitions,
            target_strategy=target_strategy,
            k_neighbors=k_neighbors,
            seed=(seed + offset) % (2**32),
        )
        partition_results[partition_name] = result
        partition_source_indices[partition_name] = source_indices
        partition_subject_ids[partition_name] = subject_ids
        partition_epoch_start_sec[partition_name] = epoch_start_sec
        partition_summary_rows.append(summary)

    train_result = partition_results["train"]
    val_result = partition_results["val"]
    final_partition = "test" if test_source == "split" else "val"
    final_result = partition_results[final_partition]
    hyperparameters = build_paper_hyperparameters(num_classes=len(task.class_names))
    train_loader = make_loader(
        train_result,
        batch_size=hyperparameters.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_result,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    final_loader = make_loader(
        final_result,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = build_paper_model(hyperparameters=hyperparameters).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=hyperparameters.learning_rate)

    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_output = train_one_epoch(
            model=model,
            batches=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_output = evaluate_one_epoch(
            model=model,
            batches=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        train_metrics = compute_classification_metrics(
            y_true=train_output.y_true,
            y_pred=train_output.y_pred,
            class_names=task.class_names,
        )
        val_metrics = compute_classification_metrics(
            y_true=val_output.y_true,
            y_pred=val_output.y_pred,
            class_names=task.class_names,
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

    final_output = evaluate_one_epoch(
        model=model,
        batches=final_loader,
        loss_fn=loss_fn,
        device=device,
    )
    final_metrics = compute_classification_metrics(
        y_true=final_output.y_true,
        y_pred=final_output.y_pred,
        class_names=task.class_names,
    )

    write_json(output_dir / "history.json", history)
    write_json(
        output_dir / "test_metrics.json",
        {"source": f"smote_{test_source}", "loss": final_output.loss, "metrics": final_metrics},
    )
    write_partition_summary_csv(output_dir / "smote_partitions.csv", partition_summary_rows)
    write_predictions_csv(
        output_dir / "test_predictions.csv",
        source_indices=partition_source_indices[final_partition],
        subject_ids=partition_subject_ids[final_partition],
        epoch_start_sec=partition_epoch_start_sec[final_partition],
        y_true=final_output.y_true,
        y_pred=final_output.y_pred,
        logits=final_output.logits,
        probabilities=final_output.probabilities,
        class_names=task.class_names,
    )
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
                "ds004504_rbp_paper SMOTE reproduction: paper-style full-task min-max normalization, "
                "epoch-level split, then simple SMOTE applied to explicit partitions."
            ),
            "class_names": list(task.class_names),
            "hyperparameters": hyperparameters.to_json_dict(),
            "normalization": normalization.to_json_dict(),
            "split_fractions": {
                "train": train_fraction,
                "val": val_fraction,
                "test": test_fraction,
            },
            "split": split.to_json_dict(),
            "evaluation": {"test_source": test_source, "final_partition": final_partition},
            "smote": {
                "target_strategy": target_strategy,
                "k_neighbors": k_neighbors,
                "apply_to_partitions": list(smote_partitions),
                "partition_summary": partition_summary_rows,
            },
            "selected_source_indices": selected_source_indices.tolist(),
        },
    )
    torch.save(
        {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters.to_json_dict(),
            "class_names": list(task.class_names),
        },
        output_dir / "model.pt",
    )
    print(f"test_source=smote_{test_source} test_loss={final_output.loss:.6f} test_acc={final_metrics['accuracy']:.6f}")
    print(f"Wrote {output_dir}")


def main() -> None:
    args = parse_args()
    run_smote_training(args.config)


if __name__ == "__main__":
    main()
