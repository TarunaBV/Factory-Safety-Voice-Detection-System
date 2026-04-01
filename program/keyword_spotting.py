# Generalized keyword analysis supports any dataset keyword folder.

from __future__ import annotations

from program.confidence_gate import apply_confidence_gate

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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


def discover_keywords(dataset_root: str | Path) -> list[str]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return []

    keywords: list[str] = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and child.name not in NEGATIVE_FOLDERS:
            if list_wav_files(child):
                keywords.append(child.name)
    return keywords


def resolve_keyword(dataset_root: str | Path, keyword: str) -> str:
    keywords = discover_keywords(dataset_root)
    if keyword in keywords:
        return keyword

    lookup = {item.lower(): item for item in keywords}
    resolved = lookup.get(keyword.lower())
    if resolved is not None:
        return resolved

    available = ", ".join(keywords) if keywords else "none"
    raise FileNotFoundError(
        f"Keyword '{keyword}' was not found in {Path(dataset_root)}. Available keywords: {available}"
    )


def sanitize_keyword_name(keyword: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", keyword.strip().lower()).strip("_")
    return cleaned or "keyword"


def default_model_path(keyword: str, output_dir: str | Path = "models") -> Path:
    return Path(output_dir) / f"{sanitize_keyword_name(keyword)}_template_spotter.npz"


def _collect_label_files(dataset_root: Path, label: str) -> list[Path]:
    label_dir = dataset_root / label
    if not label_dir.exists():
        return []
    return list_wav_files(label_dir)


def iter_negative_files(
    dataset_root: str | Path,
    keyword: str | None = None,
    include_other_keywords: bool = True,
) -> list[Path]:
    dataset_root = Path(dataset_root)
    files: list[Path] = []

    for folder in NEGATIVE_FOLDERS:
        files.extend(_collect_label_files(dataset_root, folder))

    if include_other_keywords and keyword is not None:
        resolved_keyword = resolve_keyword(dataset_root, keyword)
        for label in discover_keywords(dataset_root):
            if label != resolved_keyword:
                files.extend(_collect_label_files(dataset_root, label))

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
class MultiKeywordPrediction(KeywordPrediction):
    runner_up_label: str = "unknown"
    runner_up_score: float = 0.0


@dataclass
class KeywordAnalysis:
    keyword: str
    positive_count: int
    negative_count: int
    threshold: float
    balanced_accuracy: float
    positive_score_mean: float
    negative_score_mean: float
    score_margin: float
    positive_score_min: float
    negative_score_max: float
    output_model_path: str


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


@dataclass
class MultiKeywordTemplateSpotter:
    spotters: list[TemplateKeywordSpotter]
    backend: str = "template-ensemble"

    def __post_init__(self) -> None:
        if not self.spotters:
            raise ValueError("At least one keyword spotter is required.")

    def predict_audio(self, audio: np.ndarray, window_hop_ms: int = 250) -> MultiKeywordPrediction:
        candidates = [spotter.predict_audio(audio, window_hop_ms=window_hop_ms) for spotter in self.spotters]
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        return MultiKeywordPrediction(
            detected=best.detected,
            label=best.label if best.detected else "unknown",
            score=best.score,
            threshold=best.threshold,
            backend=self.backend,
            best_window_start_ms=best.best_window_start_ms,
            runner_up_label=runner_up.label if runner_up is not None else "unknown",
            runner_up_score=runner_up.score if runner_up is not None else 0.0,
        )

    def predict_file(self, path: str | Path) -> MultiKeywordPrediction:
        audio = load_audio(path, target_sr=self.spotters[0].config.sample_rate)
        return self.predict_audio(audio)

    @classmethod
    def from_paths(cls, model_paths: Sequence[str | Path]) -> "MultiKeywordTemplateSpotter":
        return cls([TemplateKeywordSpotter.load(path) for path in model_paths])


def analyze_template_keyword(
    dataset_root: str | Path,
    keyword: str,
    config: FeatureConfig = DEFAULT_CONFIG,
    max_positive: int | None = None,
    max_negative: int | None = None,
    output_dir: str | Path = "models",
) -> tuple[TemplateKeywordSpotter, KeywordAnalysis]:
    dataset_root = Path(dataset_root)
    resolved_keyword = resolve_keyword(dataset_root, keyword)
    positive_files = list_wav_files(dataset_root / resolved_keyword)
    negative_files = iter_negative_files(dataset_root, resolved_keyword, include_other_keywords=True)

    if not positive_files:
        raise FileNotFoundError(
            f"No WAV files found for keyword '{resolved_keyword}' in {dataset_root}"
        )

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
        keyword=resolved_keyword,
        prototype=prototype_2d,
        threshold=threshold,
        config=config,
    )
    analysis = KeywordAnalysis(
        keyword=resolved_keyword,
        positive_count=len(positive_files),
        negative_count=len(negative_files),
        threshold=float(threshold),
        balanced_accuracy=float(balanced_accuracy),
        positive_score_mean=float(np.mean(positive_scores)),
        negative_score_mean=float(np.mean(negative_scores)) if negative_scores.size else 0.0,
        score_margin=float(np.mean(positive_scores) - np.mean(negative_scores)) if negative_scores.size else float(np.mean(positive_scores)),
        positive_score_min=float(np.min(positive_scores)),
        negative_score_max=float(np.max(negative_scores)) if negative_scores.size else 0.0,
        output_model_path=str(default_model_path(resolved_keyword, output_dir=output_dir)),
    )
    return model, analysis


def train_template_spotter(
    dataset_root: str | Path,
    keyword: str = "stop",
    config: FeatureConfig = DEFAULT_CONFIG,
    max_positive: int | None = None,
    max_negative: int | None = None,
) -> tuple[TemplateKeywordSpotter, dict[str, float]]:
    model, analysis = analyze_template_keyword(
        dataset_root=dataset_root,
        keyword=keyword,
        config=config,
        max_positive=max_positive,
        max_negative=max_negative,
    )
    return model, {
        "positive_count": float(analysis.positive_count),
        "negative_count": float(analysis.negative_count),
        "positive_score_mean": float(analysis.positive_score_mean),
        "negative_score_mean": float(analysis.negative_score_mean),
        "threshold": float(analysis.threshold),
        "balanced_accuracy": float(analysis.balanced_accuracy),
    }


def train_keyword_collection(
    dataset_root: str | Path,
    keywords: Sequence[str] | None = None,
    output_dir: str | Path = "models",
    config: FeatureConfig = DEFAULT_CONFIG,
    max_positive: int | None = None,
    max_negative: int | None = None,
) -> list[KeywordAnalysis]:
    dataset_root = Path(dataset_root)
    selected_keywords = list(keywords) if keywords else discover_keywords(dataset_root)
    analyses: list[KeywordAnalysis] = []

    for keyword in selected_keywords:
        model, analysis = analyze_template_keyword(
            dataset_root=dataset_root,
            keyword=keyword,
            config=config,
            max_positive=max_positive,
            max_negative=max_negative,
            output_dir=output_dir,
        )
        model.save(analysis.output_model_path)
        analyses.append(analysis)

    return analyses


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


def load_spotters_from_directory(models_dir: str | Path) -> list[TemplateKeywordSpotter]:
    paths = sorted(Path(models_dir).glob("*_template_spotter.npz"))
    if not paths:
        raise FileNotFoundError(f"No template spotters were found in {Path(models_dir)}")
    return [load_spotter(path) for path in paths]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General keyword spotting utilities for the factory safety project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-keywords", help="List all trainable keyword folders discovered in the dataset.")
    list_parser.add_argument("--dataset-root", default="dataset/raw")

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Show discovered keywords and available file counts.")
    inspect_parser.add_argument("--dataset-root", default="dataset/raw")
    inspect_parser.add_argument("--keyword", default=None)

    analyze_parser = subparsers.add_parser("analyze-keyword", help="Analyze how well a keyword separates from all other sounds.")
    analyze_parser.add_argument("--dataset-root", default="dataset/raw")
    analyze_parser.add_argument("--keyword", required=True)
    analyze_parser.add_argument("--max-positive", type=int, default=None)
    analyze_parser.add_argument("--max-negative", type=int, default=512)
    analyze_parser.add_argument("--output-dir", default="models")

    train_parser = subparsers.add_parser("train-template", help="Train a template spotter for one keyword.")
    train_parser.add_argument("--dataset-root", default="dataset/raw")
    train_parser.add_argument("--keyword", default="stop")
    train_parser.add_argument("--output", default=None)
    train_parser.add_argument("--max-positive", type=int, default=None)
    train_parser.add_argument("--max-negative", type=int, default=512)

    train_all_parser = subparsers.add_parser("train-all", help="Train one template spotter per discovered keyword.")
    train_all_parser.add_argument("--dataset-root", default="dataset/raw")
    train_all_parser.add_argument("--output-dir", default="models")
    train_all_parser.add_argument("--max-positive", type=int, default=None)
    train_all_parser.add_argument("--max-negative", type=int, default=512)

    predict_parser = subparsers.add_parser("predict", help="Run keyword spotting with a single model on a WAV file.")
    predict_parser.add_argument("--audio", required=True)
    predict_parser.add_argument("--model", default="models/stop_template_spotter.npz")

    predict_any_parser = subparsers.add_parser("predict-any", help="Run all template models in a directory and return the best keyword match.")
    predict_any_parser.add_argument("--audio", required=True)
    predict_any_parser.add_argument("--models-dir", default="models")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "list-keywords":
        payload = {
            "dataset_root": args.dataset_root,
            "keywords": discover_keywords(args.dataset_root),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "inspect-dataset":
        keywords = [resolve_keyword(args.dataset_root, args.keyword)] if args.keyword else discover_keywords(args.dataset_root)
        payload = {
            "dataset_root": args.dataset_root,
            "keywords": keywords,
            "keyword_file_counts": {
                keyword: len(list_wav_files(Path(args.dataset_root) / keyword)) for keyword in keywords
            },
            "background_negative_files": len(iter_negative_files(args.dataset_root, include_other_keywords=False)),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "analyze-keyword":
        _, analysis = analyze_template_keyword(
            dataset_root=args.dataset_root,
            keyword=args.keyword,
            max_positive=args.max_positive,
            max_negative=args.max_negative,
            output_dir=args.output_dir,
        )
        print(json.dumps(asdict(analysis), indent=2))
        return 0

    if args.command == "train-template":
        spotter, analysis = analyze_template_keyword(
            dataset_root=args.dataset_root,
            keyword=args.keyword,
            max_positive=args.max_positive,
            max_negative=args.max_negative,
        )
        output_path = Path(args.output) if args.output else default_model_path(spotter.keyword)
        output = spotter.save(output_path)
        payload = asdict(analysis)
        payload["saved_to"] = str(output)
        print(json.dumps(payload, indent=2))
        return 0

       # ---- TRAIN ALL ----
    if args.command == "train-all":
        analyses = train_keyword_collection(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            max_positive=args.max_positive,
            max_negative=args.max_negative,
        )
        print(json.dumps([asdict(analysis) for analysis in analyses], indent=2))
        return 0


    # ---- PREDICT ----
    if args.command == "predict":
        spotter = load_spotter(args.model)
        prediction = spotter.predict_file(args.audio)

        # 🔥 Apply your confidence gate
        result = apply_confidence_gate(
            prediction.label,
            prediction.score,
            prediction.threshold
        )

        print("\n--- Model Output ---")
        print(json.dumps(asdict(prediction), indent=2))

        print("\n--- Confidence Gate Output ---")
        print(result)

        if result["status"] == "VALID":
            print("🚨 ALERT TRIGGERED")
        else:
            print("Ignored")

        return 0


    # ---- PREDICT ANY ----
    if args.command == "predict-any":
        ensemble = MultiKeywordTemplateSpotter(
            load_spotters_from_directory(args.models_dir)
        )
        prediction = ensemble.predict_file(args.audio)
        print(json.dumps(asdict(prediction), indent=2))
        return 0


    # ---- ERROR ----
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())