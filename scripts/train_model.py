from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from program.keyword_spotting import analyze_template_keyword, train_keyword_collection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train reusable keyword spotting template models from a dataset."
    )
    parser.add_argument("--dataset-root", default="dataset/raw", help="Root folder containing one subfolder per keyword.")
    parser.add_argument("--keyword", default=None, help="Train only one keyword. Omit to train all discovered keywords.")
    parser.add_argument("--output-dir", default="models", help="Directory where trained models will be saved.")
    parser.add_argument("--output", default=None, help="Optional custom output path when training a single keyword.")
    parser.add_argument("--max-positive", type=int, default=None, help="Optional cap on positive training examples.")
    parser.add_argument("--max-negative", type=int, default=512, help="Optional cap on negative training examples.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if args.keyword:
        spotter, analysis = analyze_template_keyword(
            dataset_root=dataset_root,
            keyword=args.keyword,
            max_positive=args.max_positive,
            max_negative=args.max_negative,
            output_dir=output_dir,
        )
        output_path = Path(args.output) if args.output else Path(analysis.output_model_path)
        saved_to = spotter.save(output_path)
        payload = {
            "mode": "single-keyword-training",
            "saved_to": str(saved_to),
            "analysis": asdict(analysis),
        }
        print(json.dumps(payload, indent=2))
        return 0

    analyses = train_keyword_collection(
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_positive=args.max_positive,
        max_negative=args.max_negative,
    )
    payload = {
        "mode": "multi-keyword-training",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "trained_keywords": len(analyses),
        "keywords": [asdict(analysis) for analysis in analyses],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
