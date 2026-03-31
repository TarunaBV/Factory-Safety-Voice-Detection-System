from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from program.keyword_spotting import (
    MultiKeywordTemplateSpotter,
    discover_keywords,
    load_spotter,
    load_spotters_from_directory,
    resolve_keyword,
)


def _evaluate_single_model(dataset_root: Path, keyword: str, model_path: Path) -> dict[str, object]:
    resolved_keyword = resolve_keyword(dataset_root, keyword)
    spotter = load_spotter(model_path)

    positive_files = sorted((dataset_root / resolved_keyword).rglob("*.wav"))
    positive_predictions = [spotter.predict_file(path) for path in positive_files]

    true_positive = sum(1 for prediction in positive_predictions if prediction.detected)
    false_negative = len(positive_predictions) - true_positive

    return {
        "mode": "single-model-evaluation",
        "keyword": resolved_keyword,
        "model_path": str(model_path),
        "positive_samples": len(positive_predictions),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "recall": (true_positive / len(positive_predictions)) if positive_predictions else 0.0,
        "score_mean": (
            sum(prediction.score for prediction in positive_predictions) / len(positive_predictions)
            if positive_predictions
            else 0.0
        ),
    }


def _evaluate_model_directory(dataset_root: Path, models_dir: Path) -> dict[str, object]:
    spotters = load_spotters_from_directory(models_dir)
    ensemble = MultiKeywordTemplateSpotter(spotters)
    keywords = discover_keywords(dataset_root)

    results: list[dict[str, object]] = []
    total = 0
    correct = 0

    for keyword in keywords:
        files = sorted((dataset_root / keyword).rglob("*.wav"))
        keyword_total = 0
        keyword_correct = 0
        for path in files:
            prediction = ensemble.predict_file(path)
            keyword_total += 1
            total += 1
            if prediction.label == keyword:
                keyword_correct += 1
                correct += 1

        results.append(
            {
                "keyword": keyword,
                "samples": keyword_total,
                "correct_predictions": keyword_correct,
                "accuracy": (keyword_correct / keyword_total) if keyword_total else 0.0,
            }
        )

    return {
        "mode": "multi-model-evaluation",
        "dataset_root": str(dataset_root),
        "models_dir": str(models_dir),
        "overall_accuracy": (correct / total) if total else 0.0,
        "total_samples": total,
        "per_keyword": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained keyword spotting models on files or dataset folders."
    )
    parser.add_argument("--dataset-root", default="dataset/raw", help="Dataset root used for evaluation.")
    parser.add_argument("--keyword", default=None, help="Keyword to evaluate when using a single model.")
    parser.add_argument("--model", default=None, help="Path to one trained model for single-keyword evaluation.")
    parser.add_argument("--models-dir", default=None, help="Directory of trained models for multi-keyword evaluation.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if args.model:
        if not args.keyword:
            parser.error("--keyword is required when --model is used.")
        payload = _evaluate_single_model(dataset_root, args.keyword, Path(args.model))
        print(json.dumps(payload, indent=2))
        return 0

    if args.models_dir:
        payload = _evaluate_model_directory(dataset_root, Path(args.models_dir))
        print(json.dumps(payload, indent=2))
        return 0

    parser.error("Provide either --model for single-keyword evaluation or --models-dir for multi-keyword evaluation.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
