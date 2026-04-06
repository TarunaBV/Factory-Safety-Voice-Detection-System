from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import LABELS, load_model, predict
from program.feature_extraction import FeatureConfig, extract_features_from_file


def iter_dataset_files(dataset_root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for label in LABELS:
        label_dir = dataset_root / label
        if not label_dir.exists():
            continue
        wav_files = sorted(label_dir.rglob("*.wav"))
        for wav_path in wav_files:
            samples.append((wav_path, label))
    return samples


def _sample_balanced(
    dataset_root: Path, max_per_label: int | None, seed: int
) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    for label in LABELS:
        label_dir = dataset_root / label
        if not label_dir.exists():
            continue
        wav_files = list(label_dir.rglob("*.wav"))
        if max_per_label is not None and len(wav_files) > max_per_label:
            wav_files = rng.sample(wav_files, max_per_label)
        for wav_path in wav_files:
            samples.append((wav_path, label))
    return samples


def evaluate_model(
    dataset_root: Path,
    model_path: Path,
    max_per_label: int | None,
    seed: int,
) -> dict[str, object]:
    model = load_model(model_path)
    feature_config = FeatureConfig(pre_emphasis=0.97)
    samples = _sample_balanced(dataset_root, max_per_label=max_per_label, seed=seed)

    if not samples:
        expected = ", ".join(LABELS)
        raise RuntimeError(
            f"No evaluation files found in {dataset_root}. Expected folders: {expected}."
        )

    total = 0
    correct = 0
    per_label: dict[str, dict[str, float]] = {
        label: {"total": 0, "correct": 0, "confidence_sum": 0.0} for label in LABELS
    }

    for wav_path, true_label in samples:
        features = extract_features_from_file(wav_path, config=feature_config)
        _, predicted_label, confidence = predict(model, features)

        total += 1
        per_label[true_label]["total"] += 1
        per_label[true_label]["confidence_sum"] += float(confidence)

        if predicted_label == true_label:
            correct += 1
            per_label[true_label]["correct"] += 1

    per_label_payload = []
    for label in LABELS:
        label_total = int(per_label[label]["total"])
        label_correct = int(per_label[label]["correct"])
        confidence_sum = float(per_label[label]["confidence_sum"])
        per_label_payload.append(
            {
                "label": label,
                "samples": label_total,
                "correct_predictions": label_correct,
                "accuracy": (label_correct / label_total) if label_total else 0.0,
                "mean_confidence": (confidence_sum / label_total) if label_total else 0.0,
            }
        )

    return {
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "labels": LABELS,
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "per_label": per_label_payload,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DS-CNN model on the configured labels.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset folder containing the label subfolders.")
    parser.add_argument("--model", default="models/ds_cnn_model.pth", help="Path to the trained DS-CNN model.")
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Optional cap on number of wav files evaluated per label.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    payload = evaluate_model(
        Path(args.dataset_root),
        Path(args.model),
        max_per_label=args.max_per_label,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
