from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import signal
from scipy.io import wavfile


EPSILON = 1e-10


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    clip_duration_ms: int = 1000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7600.0
    normalize: bool = True

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration_ms / 1000)


DEFAULT_CONFIG = FeatureConfig()
NOISE_LABELS = {"background_noise", "demand_noise"}


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(config: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    fft_bins = config.n_fft // 2 + 1
    mels = np.linspace(
        hz_to_mel(config.fmin),
        hz_to_mel(config.fmax),
        num=config.n_mels + 2,
        dtype=np.float64,
    )
    hz_points = mel_to_hz(mels)
    bins = np.floor((config.n_fft + 1) * hz_points / config.sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_bins - 1)

    filterbank = np.zeros((config.n_mels, fft_bins), dtype=np.float32)
    for idx in range(config.n_mels):
        left, center, right = bins[idx : idx + 3]
        if center == left:
            center = min(left + 1, fft_bins - 1)
        if right == center:
            right = min(center + 1, fft_bins - 1)
        if center > left:
            filterbank[idx, left:center] = (
                np.arange(left, center, dtype=np.float32) - left
            ) / max(center - left, 1)
        if right > center:
            filterbank[idx, center:right] = (
                right - np.arange(center, right, dtype=np.float32)
            ) / max(right - center, 1)
    return filterbank


def _to_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        return audio.astype(np.float32) / float(max_val)
    return audio.astype(np.float32, copy=False)


def load_audio(path: str | Path, target_sr: int = DEFAULT_CONFIG.sample_rate) -> np.ndarray:
    sample_rate, audio = wavfile.read(Path(path))
    audio = _to_float32(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != target_sr:
        gcd = np.gcd(sample_rate, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sample_rate // gcd)
    return np.asarray(audio, dtype=np.float32)


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == length:
        return audio
    if audio.size > length:
        return audio[:length]
    padded = np.zeros(length, dtype=np.float32)
    padded[: audio.size] = audio
    return padded


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < EPSILON:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio / peak, dtype=np.float32)


def mix_with_noise(
    audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float = 10.0,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    noise_audio = np.asarray(noise_audio, dtype=np.float32)
    if noise_audio.size < audio.size:
        repeats = int(np.ceil(audio.size / max(noise_audio.size, 1)))
        noise_audio = np.tile(noise_audio, repeats)
    noise_audio = noise_audio[: audio.size]

    signal_power = np.mean(audio**2) + EPSILON
    noise_power = np.mean(noise_audio**2) + EPSILON
    desired_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(desired_noise_power / noise_power)
    mixed = audio + noise_audio * scale
    return normalize_audio(mixed)


def frames_from_audio(audio: np.ndarray, config: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    audio = pad_or_trim(audio, config.clip_samples)
    if config.normalize:
        audio = normalize_audio(audio)
    _, _, spectrum = signal.stft(
        audio,
        fs=config.sample_rate,
        window="hann",
        nperseg=config.win_length,
        noverlap=config.win_length - config.hop_length,
        nfft=config.n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    mel_filterbank = build_mel_filterbank(config)
    mel_spec = mel_filterbank @ power
    log_mel = np.log(mel_spec + EPSILON)
    return np.asarray(log_mel, dtype=np.float32)


def standardize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / max(std, EPSILON)


def extract_features(
    audio: np.ndarray,
    config: FeatureConfig = DEFAULT_CONFIG,
    standardize: bool = True,
) -> np.ndarray:
    features = frames_from_audio(audio, config=config)
    if standardize:
        features = standardize_features(features)
    return features.astype(np.float32, copy=False)


def extract_features_from_file(
    path: str | Path,
    config: FeatureConfig = DEFAULT_CONFIG,
    standardize: bool = True,
) -> np.ndarray:
    audio = load_audio(path, target_sr=config.sample_rate)
    return extract_features(audio, config=config, standardize=standardize)


def list_wav_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).rglob("*.wav"))


def scan_dataset(
    dataset_root: str | Path,
    include_noise: bool = True,
) -> dict[str, list[Path]]:
    dataset_root = Path(dataset_root)
    class_map: dict[str, list[Path]] = {}
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if not include_noise and child.name in NOISE_LABELS:
            continue
        files = list_wav_files(child)
        if files:
            class_map[child.name] = files
    return class_map


def create_feature_table(
    dataset_root: str | Path,
    config: FeatureConfig = DEFAULT_CONFIG,
    include_noise: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, files in scan_dataset(dataset_root, include_noise=include_noise).items():
        for file_path in files:
            features = extract_features_from_file(file_path, config=config)
            rows.append(
                {
                    "path": str(file_path),
                    "label": label,
                    "shape": list(features.shape),
                    "feature_mean": float(np.mean(features)),
                    "feature_std": float(np.std(features)),
                }
            )
    return rows


def stack_feature_batch(
    files: Sequence[str | Path],
    config: FeatureConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    batch = [extract_features_from_file(path, config=config) for path in files]
    return np.stack(batch).astype(np.float32)


def save_feature_bundle(
    destination: str | Path,
    files: Sequence[str | Path],
    labels: Sequence[str],
    config: FeatureConfig = DEFAULT_CONFIG,
) -> Path:
    features = stack_feature_batch(files, config=config)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=features,
        labels=np.asarray(labels),
        files=np.asarray([str(path) for path in files]),
        config=json.dumps(asdict(config)),
    )
    return destination


def summarize_dataset(dataset_root: str | Path) -> dict[str, int]:
    return {
        label: len(files)
        for label, files in scan_dataset(dataset_root=dataset_root, include_noise=True).items()
    }


__all__ = [
    "DEFAULT_CONFIG",
    "FeatureConfig",
    "create_feature_table",
    "extract_features",
    "extract_features_from_file",
    "frames_from_audio",
    "list_wav_files",
    "load_audio",
    "mix_with_noise",
    "normalize_audio",
    "pad_or_trim",
    "save_feature_bundle",
    "scan_dataset",
    "stack_feature_batch",
    "standardize_features",
    "summarize_dataset",
]
