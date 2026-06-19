from __future__ import annotations

import argparse
import csv
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from ecg.training import compute_classification_metrics, evaluate_one_epoch, train_one_epoch
from ecg.training.ds004504_rbp_paper.factory import (
    TensorPairDataset,
    build_paper_hyperparameters,
    build_paper_model,
    load_ds004504_rbp_paper_arrays,
)
from ecg.training.ds004504_rbp_paper.normalization import fit_paper_min_max, transform_min_max
from ecg.training.ds004504_rbp_paper.splits import make_stratified_epoch_kfolds
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
    parser = argparse.ArgumentParser(description="Run paper-style k-fold training for ds004504_rbp_paper.")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = require_int(config, key)
    if value <= 0:
        raise ValueError(f"Config field {key!r} must be positive, got {value}.")
    return value


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


def write_fold_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "fold",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "train_macro_f1",
        "test_macro_f1",
        "train_support",
        "test_support",
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_kfold_training(config_path: Path) -> None:
    config = load_training_config(config_path, seen=frozenset())

    experiment_id = require_str(config, "experiment_id")
    dataset_config = require_mapping(config, "dataset")
    task_config = require_mapping(config, "task")
    training_config = require_mapping(config, "training")
    kfold_config = require_mapping(config, "kfold")
    runtime_config = require_mapping(config, "runtime")
    output_config = require_mapping(config, "output")

    h5_path = require_path(dataset_config, "h5_path")
    manifest_path = require_path(dataset_config, "manifest_path")
    task_id = require_str(task_config, "id")
    epochs = require_positive_int(training_config, "epochs")
    seed, requested_seed = resolve_seed(training_config)
    num_workers = require_int(training_config, "num_workers")
    n_splits = require_positive_int(kfold_config, "n_splits")
    device = require_str(runtime_config, "device")
    output_dir = require_path(output_config, "dir")
    overwrite = require_bool(output_config, "overwrite")

    if n_splits <= 1:
        raise ValueError(f"kfold.n_splits must be greater than 1, got {n_splits}.")
    if num_workers < 0:
        raise ValueError(f"training.num_workers must be non-negative, got {num_workers}.")
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
    x = torch.as_tensor(normalized_features[:, :, np.newaxis], dtype=torch.float32)
    y = torch.as_tensor(task_labels, dtype=torch.long)
    tensor_dataset = TensorPairDataset(x, y)

    folds = make_stratified_epoch_kfolds(labels=task_labels, n_splits=n_splits, seed=seed)
    hyperparameters = build_paper_hyperparameters(num_classes=len(task.class_names))
    loss_fn = nn.CrossEntropyLoss()

    fold_summary_rows: list[dict[str, object]] = []
    for fold in folds:
        fold_seed = (seed + fold.fold_index - 1) % (2**32)
        torch.manual_seed(fold_seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(fold_seed)

        fold_dir = output_dir / f"fold_{fold.fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model = build_paper_model(hyperparameters=hyperparameters).to(device)
        optimizer = Adam(model.parameters(), lr=hyperparameters.learning_rate)
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            Subset(tensor_dataset, fold.train.tolist()),
            batch_size=hyperparameters.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            Subset(tensor_dataset, fold.test.tolist()),
            batch_size=hyperparameters.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        history: list[dict[str, object]] = []
        final_train_output = None
        final_test_output = None
        final_train_metrics = None
        final_test_metrics = None
        for epoch in range(1, epochs + 1):
            train_output = train_one_epoch(
                model=model,
                batches=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )
            test_output = evaluate_one_epoch(
                model=model,
                batches=test_loader,
                loss_fn=loss_fn,
                device=device,
            )
            train_metrics = compute_classification_metrics(
                y_true=train_output.y_true,
                y_pred=train_output.y_pred,
                class_names=task.class_names,
            )
            test_metrics = compute_classification_metrics(
                y_true=test_output.y_true,
                y_pred=test_output.y_pred,
                class_names=task.class_names,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_output.loss,
                    "test_loss": test_output.loss,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                }
            )
            final_train_output = train_output
            final_test_output = test_output
            final_train_metrics = train_metrics
            final_test_metrics = test_metrics
            print(
                f"fold={fold.fold_index} "
                f"epoch={epoch} "
                f"train_loss={train_output.loss:.6f} "
                f"train_acc={train_metrics['accuracy']:.6f} "
                f"test_loss={test_output.loss:.6f} "
                f"test_acc={test_metrics['accuracy']:.6f}"
            )

        if (
            final_train_output is None
            or final_test_output is None
            or final_train_metrics is None
            or final_test_metrics is None
        ):
            raise RuntimeError(f"Fold {fold.fold_index} did not produce final metrics.")

        test_indices = tuple(int(value) for value in fold.test.tolist())
        source_indices = tuple(int(selected_source_indices[index]) for index in test_indices)
        subject_ids = tuple(selected_subject_ids[index] for index in test_indices)
        epoch_start_sec = tuple(float(selected_epoch_start_sec[index]) for index in test_indices)

        write_json(fold_dir / "history.json", history)
        write_json(
            fold_dir / "train_metrics.json",
            {"source": "kfold_train", "loss": final_train_output.loss, "metrics": final_train_metrics},
        )
        write_json(
            fold_dir / "test_metrics.json",
            {"source": "kfold_test", "loss": final_test_output.loss, "metrics": final_test_metrics},
        )
        write_predictions_csv(
            fold_dir / "test_predictions.csv",
            source_indices=source_indices,
            subject_ids=subject_ids,
            epoch_start_sec=epoch_start_sec,
            y_true=final_test_output.y_true,
            y_pred=final_test_output.y_pred,
            logits=final_test_output.logits,
            probabilities=final_test_output.probabilities,
            class_names=task.class_names,
        )
        torch.save(
            {
                "experiment_id": experiment_id,
                "task_id": task_id,
                "fold_index": fold.fold_index,
                "model_state_dict": model.state_dict(),
                "hyperparameters": hyperparameters.to_json_dict(),
                "class_names": list(task.class_names),
            },
            fold_dir / "model.pt",
        )
        fold_summary_rows.append(
            {
                "fold": fold.fold_index,
                "train_loss": final_train_output.loss,
                "test_loss": final_test_output.loss,
                "train_accuracy": final_train_metrics["accuracy"],
                "test_accuracy": final_test_metrics["accuracy"],
                "train_macro_f1": final_train_metrics["macro_f1"],
                "test_macro_f1": final_test_metrics["macro_f1"],
                "train_support": final_train_metrics["support"],
                "test_support": final_test_metrics["support"],
            }
        )

    write_fold_summary_csv(output_dir / "fold_summary.csv", fold_summary_rows)
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
                "ds004504_rbp_paper k-fold reproduction: paper-style full-task min-max normalization "
                "before epoch-level stratified k-fold split."
            ),
            "class_names": list(task.class_names),
            "hyperparameters": hyperparameters.to_json_dict(),
            "normalization": normalization.to_json_dict(),
            "kfold": {
                "n_splits": n_splits,
                "folds": [fold.to_json_dict() for fold in folds],
            },
            "selected_source_indices": selected_source_indices.tolist(),
        },
    )
    print(f"Wrote {output_dir}")


def main() -> None:
    args = parse_args()
    run_kfold_training(args.config)


if __name__ == "__main__":
    main()
