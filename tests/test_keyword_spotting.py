from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from program.keyword_spotting import (
    MultiKeywordTemplateSpotter,
    TemplateKeywordSpotter,
    analyze_template_keyword,
    default_model_path,
    discover_keywords,
    train_keyword_collection,
)


SAMPLE_RATE = 16000


def _write_wave(path: Path, frequency: float, duration_s: float = 1.0, amplitude: float = 0.4) -> None:
    time_axis = np.linspace(0.0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    audio = amplitude * np.sin(2.0 * np.pi * frequency * time_axis)
    wavfile.write(path, SAMPLE_RATE, np.asarray(audio, dtype=np.float32))


def _build_dataset(root: Path) -> Path:
    dataset_root = root / "raw"
    for label in ("stop", "help", "background_noise"):
        (dataset_root / label).mkdir(parents=True, exist_ok=True)

    for index, frequency in enumerate((440.0, 450.0, 460.0)):
        _write_wave(dataset_root / "stop" / f"stop_{index}.wav", frequency=frequency)

    for index, frequency in enumerate((880.0, 890.0, 900.0)):
        _write_wave(dataset_root / "help" / f"help_{index}.wav", frequency=frequency)

    rng = np.random.default_rng(7)
    for index in range(3):
        noise = 0.05 * rng.standard_normal(SAMPLE_RATE).astype(np.float32)
        wavfile.write(dataset_root / "background_noise" / f"noise_{index}.wav", SAMPLE_RATE, noise)

    return dataset_root


def test_discover_and_analyze_keyword_dataset(tmp_path: Path) -> None:
    dataset_root = _build_dataset(tmp_path)

    assert discover_keywords(dataset_root) == ["help", "stop"]

    _, analysis = analyze_template_keyword(dataset_root=dataset_root, keyword="stop", output_dir=tmp_path / "models")

    assert analysis.keyword == "stop"
    assert analysis.positive_count == 3
    assert analysis.negative_count == 6
    assert analysis.positive_score_mean > analysis.negative_score_mean
    assert analysis.output_model_path == str(default_model_path("stop", output_dir=tmp_path / "models"))


def test_train_collection_and_predict_any_keyword(tmp_path: Path) -> None:
    dataset_root = _build_dataset(tmp_path)
    output_dir = tmp_path / "models"

    analyses = train_keyword_collection(dataset_root=dataset_root, output_dir=output_dir)

    assert {analysis.keyword for analysis in analyses} == {"help", "stop"}

    ensemble = MultiKeywordTemplateSpotter.from_paths(
        [output_dir / "help_template_spotter.npz", output_dir / "stop_template_spotter.npz"]
    )
    prediction = ensemble.predict_file(dataset_root / "help" / "help_0.wav")

    assert prediction.detected is True
    assert prediction.label == "help"
    assert prediction.score >= prediction.threshold
    assert prediction.runner_up_score <= prediction.score

    restored = TemplateKeywordSpotter.load(output_dir / "stop_template_spotter.npz")
    stop_prediction = restored.predict_file(dataset_root / "stop" / "stop_0.wav")
    assert stop_prediction.label == "stop"
