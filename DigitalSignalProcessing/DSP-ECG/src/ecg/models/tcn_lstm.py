"""TCN-LSTM classifiers for sequence features."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """Causal 1D convolution that preserves sequence length."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}.")

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TcnResidualBlock(nn.Module):
    """Two-layer temporal convolution block with a residual projection."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= value < 1, got {dropout}.")

        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout1d(dropout)

        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout1d(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.dropout1(out)

        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.dropout2(out)

        return out + residual


class TcnLstmClassifier(nn.Module):
    """TCN-LSTM classifier for tensors shaped ``(batch, seq_len, input_dim)``."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        tcn_channels: int,
        tcn_kernel_size: int,
        tcn_dilations: Sequence[int],
        tcn_dropout: float,
        lstm_hidden_dim: int,
        dense_hidden_dims: Sequence[int],
        dense_dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if tcn_channels <= 0:
            raise ValueError(f"tcn_channels must be positive, got {tcn_channels}.")
        if tcn_kernel_size <= 0:
            raise ValueError(f"tcn_kernel_size must be positive, got {tcn_kernel_size}.")
        if lstm_hidden_dim <= 0:
            raise ValueError(f"lstm_hidden_dim must be positive, got {lstm_hidden_dim}.")
        if not 0.0 <= tcn_dropout < 1.0:
            raise ValueError(f"tcn_dropout must satisfy 0 <= value < 1, got {tcn_dropout}.")
        if not 0.0 <= dense_dropout < 1.0:
            raise ValueError(f"dense_dropout must satisfy 0 <= value < 1, got {dense_dropout}.")
        if not tcn_dilations:
            raise ValueError("tcn_dilations must define at least one TCN block.")
        if not dense_hidden_dims:
            raise ValueError("dense_hidden_dims must define at least one hidden layer.")
        for index, dilation in enumerate(tcn_dilations):
            if dilation <= 0:
                raise ValueError(f"tcn_dilations[{index}] must be positive, got {dilation}.")
        for index, hidden_dim in enumerate(dense_hidden_dims):
            if hidden_dim <= 0:
                raise ValueError(f"dense_hidden_dims[{index}] must be positive, got {hidden_dim}.")

        self.input_dim = input_dim

        tcn_blocks: list[nn.Module] = []
        in_channels = input_dim
        for dilation in tcn_dilations:
            tcn_blocks.append(
                TcnResidualBlock(
                    in_channels=in_channels,
                    out_channels=tcn_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                )
            )
            in_channels = tcn_channels
        self.tcn = nn.Sequential(*tcn_blocks)

        self.lstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )

        classifier_layers: list[nn.Module] = []
        classifier_input_dim = lstm_hidden_dim
        for hidden_dim in dense_hidden_dims:
            classifier_layers.append(nn.Linear(classifier_input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dense_dropout))
            classifier_input_dim = hidden_dim
        classifier_layers.append(nn.Linear(classifier_input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (batch, seq_len, input_dim), got {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}.")

        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)
