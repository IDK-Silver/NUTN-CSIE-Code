"""Relative band power feature extraction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import signal


def band_mask(freqs: np.ndarray, low: float, high: float, include_high: bool) -> np.ndarray:
    if include_high:
        return (freqs >= low) & (freqs <= high)
    return (freqs >= low) & (freqs < high)


def sum_psd_bins(freqs: np.ndarray, psd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected_freqs = freqs[mask]
    selected_psd = psd[..., mask]
    if selected_freqs.size == 0:
        raise ValueError("Frequency mask selected no Welch bins.")
    return selected_psd.sum(axis=-1)


def compute_rbp_for_epoch(
    epoch_data: np.ndarray,
    *,
    sampling_rate_hz: float,
    bands: Mapping[str, tuple[float, float]],
    total_power_range_hz: tuple[float, float],
    welch_window_sec: float,
    welch_overlap: float,
) -> np.ndarray:
    nperseg = int(round(welch_window_sec * sampling_rate_hz))
    noverlap = int(round(nperseg * welch_overlap))
    if nperseg <= 0:
        raise ValueError("Welch nperseg must be positive.")
    if noverlap >= nperseg:
        raise ValueError("Welch overlap must be smaller than the Welch window.")

    freqs, psd = signal.welch(
        epoch_data,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=min(nperseg, epoch_data.shape[-1]),
        noverlap=min(noverlap, max(0, epoch_data.shape[-1] - 1)),
        detrend="constant",
        scaling="density",
        axis=-1,
        average="mean",
    )

    total_mask = band_mask(
        freqs,
        total_power_range_hz[0],
        total_power_range_hz[1],
        include_high=True,
    )
    total_power = sum_psd_bins(freqs, psd, total_mask)

    rbp_by_band: list[np.ndarray] = []
    for low, high in bands.values():
        include_high = high == total_power_range_hz[1]
        power = sum_psd_bins(freqs, psd, band_mask(freqs, low, high, include_high))
        rbp = np.divide(
            power,
            total_power,
            out=np.full_like(power, np.nan, dtype=np.float64),
            where=total_power > 0,
        )
        rbp_by_band.append(rbp)

    return np.stack(rbp_by_band, axis=-1).astype(np.float32)

