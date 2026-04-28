import pytest
import torch

from ecg.models import TcnLstmClassifier


def test_tcn_lstm_classifier_forward_does_not_fix_sequence_length() -> None:
    model = TcnLstmClassifier(
        input_dim=1,
        num_classes=3,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=(1, 2),
        tcn_dropout=0.0,
        lstm_hidden_dim=5,
        dense_hidden_dims=(6,),
        dense_dropout=0.0,
    )
    model.eval()

    logits_short = model(torch.randn(2, 3, 1))
    logits_long = model(torch.randn(2, 7, 1))

    assert tuple(logits_short.shape) == (2, 3)
    assert tuple(logits_long.shape) == (2, 3)


def test_tcn_lstm_classifier_rejects_wrong_input_dim() -> None:
    model = TcnLstmClassifier(
        input_dim=1,
        num_classes=3,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=(1,),
        tcn_dropout=0.0,
        lstm_hidden_dim=5,
        dense_hidden_dims=(6,),
        dense_dropout=0.0,
    )

    with pytest.raises(ValueError, match="Expected input_dim=1"):
        model(torch.randn(2, 3, 2))
