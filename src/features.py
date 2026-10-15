"""Feature extraction for XJTU-SY vibration recordings."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


EXPECTED_COLUMNS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
CHANNELS = {
    "h": "Horizontal_vibration_signals",
    "v": "Vertical_vibration_signals",
}
FREQUENCY_BANDS_HZ = [
    (0.0, 1000.0),
    (1000.0, 3000.0),
    (3000.0, 6000.0),
    (6000.0, 12800.0),
]


def feature_names() -> list[str]:
    names: list[str] = []
    per_channel = [
        "mean",
        "std",
        "rms",
        "peak_abs",
        "peak_to_peak",
        "crest_factor",
        "kurtosis",
        "skewness",
        "dominant_freq_hz",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rms",
    ]
    band_names = [
        f"band_energy_ratio_{int(low)}_{int(high)}hz" for low, high in FREQUENCY_BANDS_HZ
    ]
    for prefix in CHANNELS:
        names.extend(f"{prefix}_{name}" for name in per_channel)
        names.extend(f"{prefix}_{name}" for name in band_names)
    return names


def validate_recording(frame: pd.DataFrame, expected_rows: int, path: str) -> None:
    columns = list(frame.columns)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"{path}: expected columns {EXPECTED_COLUMNS}, found {columns}")
    if len(frame) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(frame)}")
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: found NaN or Inf values")


def read_recording(path: str, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=EXPECTED_COLUMNS, dtype=np.float64)
    validate_recording(frame, expected_rows=expected_rows, path=path)
    return frame


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def channel_features(signal: np.ndarray, sampling_frequency_hz: float) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - float(np.mean(signal))

    mean_value = float(np.mean(signal))
    std_value = float(np.std(signal, ddof=1))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak_abs = float(np.max(np.abs(signal)))
    peak_to_peak = float(np.ptp(signal))
    crest_factor = _safe_ratio(peak_abs, rms)
    kurtosis_value = float(kurtosis(signal, fisher=True, bias=False))
    skewness_value = float(skew(signal, bias=False))

    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_frequency_hz)
    power = np.square(np.abs(spectrum))
    total_power = float(np.sum(power))

    if total_power > 0.0:
        dominant_freq = float(freqs[int(np.argmax(power))])
        spectral_centroid = float(np.sum(freqs * power) / total_power)
        spectral_bandwidth = float(
            np.sqrt(np.sum(np.square(freqs - spectral_centroid) * power) / total_power)
        )
        spectral_rms = float(np.sqrt(np.mean(power)))
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        spectral_rms = 0.0

    features = {
        "mean": mean_value,
        "std": std_value,
        "rms": rms,
        "peak_abs": peak_abs,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis_value,
        "skewness": skewness_value,
        "dominant_freq_hz": dominant_freq,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_rms": spectral_rms,
    }

    for low, high in FREQUENCY_BANDS_HZ:
        if high >= freqs[-1]:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)
        band_power = float(np.sum(power[mask]))
        features[f"band_energy_ratio_{int(low)}_{int(high)}hz"] = _safe_ratio(
            band_power, total_power
        )

    return features


def extract_recording_features(frame: pd.DataFrame, sampling_frequency_hz: float) -> dict[str, float]:
    output: dict[str, float] = {}
    for prefix, column in CHANNELS.items():
        channel_output = channel_features(frame[column].to_numpy(), sampling_frequency_hz)
        output.update({f"{prefix}_{name}": value for name, value in channel_output.items()})
    return output


def assert_feature_vector_complete(row: dict[str, float], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in row]
    if missing:
        raise ValueError(f"Feature row is missing {missing}")
    values = np.asarray([row[name] for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Feature row contains NaN or Inf")

