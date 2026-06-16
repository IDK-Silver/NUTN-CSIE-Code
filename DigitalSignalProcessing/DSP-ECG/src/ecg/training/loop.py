"""Generic PyTorch training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import Tensor, nn
from torch.autograd import no_grad
from torch.optim.optimizer import Optimizer


@dataclass(frozen=True)
class EpochOutput:
    loss: float
    y_true: tuple[int, ...]
    y_pred: tuple[int, ...]
    logits: tuple[tuple[float, ...], ...]
    probabilities: tuple[tuple[float, ...], ...]


def train_one_epoch(
    *,
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> EpochOutput:
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_logits: list[tuple[float, ...]] = []
    y_probabilities: list[tuple[float, ...]] = []

    for x, y in batches:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        y_true.extend(int(value) for value in y.detach().cpu().numpy().tolist())
        y_pred.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().numpy().tolist())
        y_logits.extend(tuple(float(item) for item in row) for row in logits.detach().cpu().numpy().tolist())
        y_probabilities.extend(
            tuple(float(item) for item in row)
            for row in logits.softmax(dim=1).detach().cpu().numpy().tolist()
        )

    if total_samples <= 0:
        raise ValueError("Cannot train on an empty batch iterator.")

    return EpochOutput(
        loss=total_loss / total_samples,
        y_true=tuple(y_true),
        y_pred=tuple(y_pred),
        logits=tuple(y_logits),
        probabilities=tuple(y_probabilities),
    )


def evaluate_one_epoch(
    *,
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    device: str,
) -> EpochOutput:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_logits: list[tuple[float, ...]] = []
    y_probabilities: list[tuple[float, ...]] = []

    with no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = int(y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true.extend(int(value) for value in y.detach().cpu().numpy().tolist())
            y_pred.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().numpy().tolist())
            y_logits.extend(tuple(float(item) for item in row) for row in logits.detach().cpu().numpy().tolist())
            y_probabilities.extend(
                tuple(float(item) for item in row)
                for row in logits.softmax(dim=1).detach().cpu().numpy().tolist()
            )

    if total_samples <= 0:
        raise ValueError("Cannot evaluate on an empty batch iterator.")

    return EpochOutput(
        loss=total_loss / total_samples,
        y_true=tuple(y_true),
        y_pred=tuple(y_pred),
        logits=tuple(y_logits),
        probabilities=tuple(y_probabilities),
    )
