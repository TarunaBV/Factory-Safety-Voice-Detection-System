from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from models.model_loader import load_model, predict

import numpy as np
import time

model = load_model("models/ds_cnn_model.pth")

THRESHOLD = 0.8


def main():
    print("🎤 Listening...")

    stream = get_audio_stream()

    BUFFER_SIZE = 16000
    buffer = np.zeros(0, dtype=np.float32)

    COOLDOWN = 2
    last_time = 0

    try:
        while True:
            chunk = read_audio_chunk(stream)
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) > BUFFER_SIZE:
                buffer = buffer[-BUFFER_SIZE:]

            if len(buffer) < BUFFER_SIZE:
                continue

            speech = apply_vad(buffer)

            if speech is None:
                continue

            features = extract_features(buffer)
            label_id, label_name, conf = predict(model, features)

            print(f"{label_name} ({conf:.2f})")

            if label_name == "stop" and conf > THRESHOLD:
                if time.time() - last_time > COOLDOWN:
                    print(f"🚨 STOP DETECTED!")
                    last_time = time.time()

    except KeyboardInterrupt:
        close_stream(stream)
        print("\nStopped")


if __name__ == "__main__":
    main()