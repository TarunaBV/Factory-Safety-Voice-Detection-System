from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from program.keyword_spotting import load_spotter
from program.confidence_gate import apply_confidence_gate

import numpy as np

# Load trained model
MODEL_PATH = "models/stop_template_spotter.npz"
spotter = load_spotter(MODEL_PATH)

THRESHOLD = 0.6  # base threshold


def main():
    print("Listening for STOP command...")

    stream = get_audio_stream()

    try:
        while True:
            audio_chunk = read_audio_chunk(stream)

            # Step 1: VAD
            speech = apply_vad(audio_chunk)
            if speech is None:
                continue

            # Step 2: Feature extraction
            features = extract_features(speech)

            # Step 3: Keyword prediction
            prediction = spotter.predict_features(features)

            # Step 4: Confidence gate
            result = apply_confidence_gate(
                prediction.label,
                prediction.score,
                THRESHOLD
            )

            if result["status"] == "VALID":
                print(f"STOP DETECTED! Confidence: {result['confidence']:.2f}")
            else:
                print(f"Ignored ({result['confidence']:.2f})")

    except KeyboardInterrupt:
        print("Stopped.")
        close_stream(stream)


if __name__ == "__main__":
    main()