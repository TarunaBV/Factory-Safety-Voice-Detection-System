from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from models.model_loader import load_model, predict

import numpy as np
import time
import torch


MODEL_PATH = "models/ds_cnn_model.pth"

# 🔥 DEVICE SUPPORT (CPU/GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(MODEL_PATH, device=DEVICE)

THRESHOLD = 0.75


def main():
    print("🎤 Listening for STOP command...")

    stream = get_audio_stream()

    BUFFER_SIZE = 16000
    audio_buffer = np.zeros(0, dtype=np.float32)

    COOLDOWN_TIME = 2
    last_detected_time = 0

    speech_active = False
    silence_counter = 0
    SILENCE_LIMIT = 8

    try:
        while True:
            audio_chunk = read_audio_chunk(stream)

            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            # 🔥 keep only last 1 second audio
            if len(audio_buffer) > BUFFER_SIZE:
                audio_buffer = audio_buffer[-BUFFER_SIZE:]

            if len(audio_buffer) < BUFFER_SIZE:
                continue

            speech = apply_vad(audio_buffer)

            if speech is not None:
                speech_active = True
                silence_counter = 0
                continue

            else:
                if speech_active:
                    silence_counter += 1

                    if silence_counter < SILENCE_LIMIT:
                        continue

                    print("🔍 Processing speech...")

                    features = extract_features(audio_buffer)

                    # 🔥 FIXED unpacking (3 values now)
                    label_idx, label_name, confidence = predict(
                        model, features, device=DEVICE
                    )

                    print(f"Prediction: {label_name}, Confidence: {confidence:.2f}")

                    current_time = time.time()

                    # 🔥 cleaner logic using label_name instead of index
                    if label_name == "stop" and confidence > THRESHOLD:
                        if current_time - last_detected_time > COOLDOWN_TIME:
                            print(f"🚨 STOP DETECTED! Confidence: {confidence:.2f}")
                            last_detected_time = current_time

                    speech_active = False

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        close_stream(stream)


if __name__ == "__main__":
    main()