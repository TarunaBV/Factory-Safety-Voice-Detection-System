from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from program.feature_extraction import (
        DEFAULT_CONFIG,
        FeatureConfig,
        extract_features,
        extract_features_from_file,
        list_wav_files,
        load_audio,
        pad_or_trim,
    )
except ModuleNotFoundError:
    from feature_extraction import (
        DEFAULT_CONFIG,
        FeatureConfig,
        extract_features,
        extract_features_from_file,
        list_wav_files,
        load_audio,
        pad_or_trim,
    )


EPSILON = 1e-10
NEGATIVE_FOLDERS = ("background_noise", "demand_noise")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = (np.linalg.norm(left) * np.linalg.norm(right)) + EPSILON
    return float(np.dot(left, right) / denominator)


def iter_negative_files(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root)
    files: list[Path] = []
    for folder in NEGATIVE_FOLDERS:
        path = dataset_root / folder
        if path.exists():
            files.extend(list_wav_files(path))
    return sorted(files)


def _flatten_feature_batch(feature_batch: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in feature_batch])


def _best_threshold(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> tuple[float, float]:
    candidates = np.unique(np.concatenate([positive_scores, negative_scores]))
    if candidates.size == 0:
        return 0.5, 0.0

    best_threshold = float(candidates[0])
    best_accuracy = -1.0
    for threshold in candidates:
        true_positive = np.mean(positive_scores >= threshold) if positive_scores.size else 0.0
        true_negative = np.mean(negative_scores < threshold) if negative_scores.size else 0.0
        accuracy = (true_positive + true_negative) / 2.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


@dataclass
class KeywordPrediction:
    detected: bool
    label: str
    score: float
    threshold: float
    backend: str
    best_window_start_ms: int = 0


@dataclass
class TemplateKeywordSpotter:
    keyword: str
    prototype: np.ndarray
    threshold: float
    config: FeatureConfig = DEFAULT_CONFIG
    backend: str = "template"

    def score_features(self, features: np.ndarray) -> float:
        return cosine_similarity(self.prototype, np.asarray(features, dtype=np.float32))

    def predict_features(self, features: np.ndarray) -> KeywordPrediction:
        score = self.score_features(features)
        return KeywordPrediction(
            detected=score >= self.threshold,
            label=self.keyword if score >= self.threshold else "unknown",
            score=score,
            threshold=self.threshold,
            backend=self.backend,
        )

    def predict_file(self, path: str | Path) -> KeywordPrediction:
        audio = load_audio(path, target_sr=self.config.sample_rate)
        return self.predict_audio(audio)

    def predict_audio(self, audio: np.ndarray, window_hop_ms: int = 250) -> KeywordPrediction:
        window_samples = self.config.clip_samples
        hop_samples = max(1, int(self.config.sample_rate * window_hop_ms / 1000))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size <= window_samples:
            features = extract_features(pad_or_trim(audio, window_samples), config=self.config)
            prediction = self.predict_features(features)
            prediction.best_window_start_ms = 0
            return prediction

        best_prediction: KeywordPrediction | None = None
        for start in range(0, audio.size - window_samples + 1, hop_samples):
            window = audio[start : start + window_samples]
            features = extract_features(window, config=self.config)
            prediction = self.predict_features(features)
            prediction.best_window_start_ms = int(start * 1000 / self.config.sample_rate)
            if best_prediction is None or prediction.score > best_prediction.score:
                best_prediction = prediction

        assert best_prediction is not None
        return best_prediction

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keyword=self.keyword,
            prototype=self.prototype.astype(np.float32),
            threshold=np.float32(self.threshold),
            config=json.dumps(asdict(self.config)),
            backend=self.backend,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TemplateKeywordSpotter":
        payload = np.load(Path(path), allow_pickle=False)
        config = FeatureConfig(**json.loads(str(payload["config"])))
        return cls(
            keyword=str(payload["keyword"]),
            prototype=np.asarray(payload["prototype"], dtype=np.float32),
            threshold=float(payload["threshold"]),
            config=config,
            backend=str(payload["backend"]),
        )


def train_template_spotter(
    dataset_root: str | Path,
    keyword: str = "stop",
    config: FeatureConfig = DEFAULT_CONFIG,
    max_positive: int | None = None,
    max_negative: int | None = None,
) -> tuple[TemplateKeywordSpotter, dict[str, float]]:
    dataset_root = Path(dataset_root)
    positive_files = list_wav_files(dataset_root / keyword)
    negative_files = iter_negative_files(dataset_root)

    if not positive_files:
        raise FileNotFoundError(f"No WAV files found for keyword '{keyword}' in {dataset_root}")

    if max_positive is not None:
        positive_files = positive_files[:max_positive]
    if max_negative is not None:
        negative_files = negative_files[:max_negative]

    positive_features = [extract_features_from_file(path, config=config) for path in positive_files]
    prototype = np.mean(_flatten_feature_batch(positive_features), axis=0).astype(np.float32)
    prototype_2d = prototype.reshape(positive_features[0].shape)

    positive_scores = np.asarray(
        [cosine_similarity(prototype_2d, features) for features in positive_features],
        dtype=np.float32,
    )

    negative_scores = np.asarray(
        [cosine_similarity(prototype_2d, extract_features_from_file(path, config=config)) for path in negative_files],
        dtype=np.float32,
    )

    threshold, balanced_accuracy = _best_threshold(positive_scores, negative_scores)
    model = TemplateKeywordSpotter(
        keyword=keyword,
        prototype=prototype_2d,
        threshold=threshold,
        config=config,
    )

    metrics = {
        "positive_count": float(len(positive_files)),
        "negative_count": float(len(negative_files)),
        "positive_score_mean": float(np.mean(positive_scores)),
        "negative_score_mean": float(np.mean(negative_scores)) if negative_scores.size else 0.0,
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy),
    }
    return model, metrics


def load_spotter(model_path: str | Path) -> TemplateKeywordSpotter:
    model_path = Path(model_path)
    if model_path.suffix.lower() == ".npz":
        return TemplateKeywordSpotter.load(model_path)
    if model_path.suffix.lower() == ".pth":
        raise RuntimeError(
            "PyTorch model loading is not available in this environment because 'torch' is not installed. "
            "Train or use a template '.npz' spotter for now."
        )
    raise ValueError(f"Unsupported model format: {model_path.suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword spotting utilities for the factory safety project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-template", help="Train a template spotter from raw dataset folders.")
    train_parser.add_argument("--dataset-root", default="dataset/raw")
    train_parser.add_argument("--keyword", default="stop")
    train_parser.add_argument("--output", default="models/stop_template_spotter.npz")
    train_parser.add_argument("--max-positive", type=int, default=None)
    train_parser.add_argument("--max-negative", type=int, default=512)

    predict_parser = subparsers.add_parser("predict", help="Run keyword spotting on a WAV file.")
    predict_parser.add_argument("--audio", required=True)
    predict_parser.add_argument("--model", default="models/stop_template_spotter.npz")

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Show raw dataset counts used by the spotter.")
    inspect_parser.add_argument("--dataset-root", default="dataset/raw")
    inspect_parser.add_argument("--keyword", default="stop")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "train-template":
        spotter, metrics = train_template_spotter(
            dataset_root=args.dataset_root,
            keyword=args.keyword,
            max_positive=args.max_positive,
            max_negative=args.max_negative,
        )
        output = spotter.save(args.output)
        print(json.dumps({"saved_to": str(output), **metrics}, indent=2))
        return 0

    if args.command == "predict":
        spotter = load_spotter(args.model)
        prediction = spotter.predict_file(args.audio)
        print(json.dumps(asdict(prediction), indent=2))
        return 0

    if args.command == "inspect-dataset":
        dataset_root = Path(args.dataset_root)
        payload = {
            "keyword": args.keyword,
            "positive_files": len(list_wav_files(dataset_root / args.keyword)),
            "negative_files": len(iter_negative_files(dataset_root)),
            "negative_folders": list(NEGATIVE_FOLDERS),
        }
        print(json.dumps(payload, indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
