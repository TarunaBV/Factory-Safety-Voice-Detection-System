from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm  # Added for a progress bar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import LABELS, load_model, predict
from program.feature_extraction import extract_features_from_file

def iter_dataset_files(dataset_root: Path) -> list[tuple[Path, str]]:
    """
    Finds wav files and maps folder names to model labels.
    Folder '_background_noise_' -> 'not_stop'
    Folder 'other_speech' -> 'not_stop'
    Folder 'stop' -> 'stop'
    """
    samples: list[tuple[Path, str]] = []
    
    # Mapping based on your screenshot
    folder_mapping = {
        "stop": "stop",
        "other_speech": "not_stop",
        "_background_noise_": "not_stop"
    }

    print(f"Scanning directory: {dataset_root}")
    for folder_name, target_label in folder_mapping.items():
        label_dir = dataset_root / folder_name
        if not label_dir.exists():
            print(f"  [!] Skipping {folder_name} (folder not found)")
            continue
        
        found_files = list(label_dir.rglob("*.wav"))
        print(f"  [+] Found {len(found_files)} files in {folder_name} (Label: {target_label})")
        
        for wav_path in found_files:
            samples.append((wav_path, target_label))
            
    return samples

def evaluate_model(dataset_root: Path, model_path: Path) -> dict[str, object]:
    model = load_model(model_path)
    samples = iter_dataset_files(dataset_root)

    if not samples:
        raise RuntimeError(f"No .wav files found in {dataset_root}")

    total = 0
    correct = 0
    # Initialize stats for 'stop' and 'not_stop'
    per_label: dict[str, dict[str, float]] = {
        label: {"total": 0, "correct": 0, "confidence_sum": 0.0} for label in ["stop", "not_stop"]
    }

    # Added tqdm so you can see progress and don't feel the need to interrupt
    for wav_path, true_label in tqdm(samples, desc="Evaluating"):
        try:
            features = extract_features_from_file(wav_path)
            _, predicted_label, confidence = predict(model, features)

            total += 1
            per_label[true_label]["total"] += 1
            per_label[true_label]["confidence_sum"] += float(confidence)

            if predicted_label == true_label:
                correct += 1
                per_label[true_label]["correct"] += 1
        except Exception as e:
            print(f"\nError processing {wav_path.name}: {e}")
            continue

    per_label_payload = []
    for label in ["stop", "not_stop"]:
        label_total = int(per_label[label]["total"])
        label_correct = int(per_label[label]["correct"])
        confidence_sum = float(per_label[label]["confidence_sum"])
        
        accuracy = (label_correct / label_total) if label_total else 0.0
        mean_conf = (confidence_sum / label_total) if label_total else 0.0
        
        per_label_payload.append({
            "label": label,
            "samples": label_total,
            "correct_predictions": label_correct,
            "accuracy": accuracy,
            "mean_confidence": mean_conf,
        })

    return {
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "per_label": per_label_payload,
    }

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DS-CNN model on Factory Safety data.")
    parser.add_argument("--dataset-root", default="dataset/final", help="Dataset folder.")
    parser.add_argument("--model", default="models/ds_cnn_model.pth", help="Path to trained model.")
    return parser

def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Create Path objects
    ds_path = Path(args.dataset_root)
    model_path = Path(args.model)

    payload = evaluate_model(ds_path, model_path)
    
    print("\n--- EVALUATION RESULTS ---")
    print(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
        sys.exit(1)