import pytest
import torch
from torch import nn

from ecg.training import evaluate_one_epoch


class IdentityLogitModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_evaluate_one_epoch_returns_logits_and_probabilities() -> None:
    output = evaluate_one_epoch(
        model=IdentityLogitModel(),
        batches=[(torch.Tensor([[2.0, 0.0], [0.0, 1.0]]), torch.Tensor([0, 1]).long())],
        loss_fn=nn.CrossEntropyLoss(),
        device="cpu",
    )

    assert output.y_true == (0, 1)
    assert output.y_pred == (0, 1)
    assert output.logits == ((2.0, 0.0), (0.0, 1.0))
    assert output.probabilities[0] == pytest.approx((0.880797, 0.119203), abs=1e-6)
    assert output.probabilities[1] == pytest.approx((0.268941, 0.731059), abs=1e-6)
