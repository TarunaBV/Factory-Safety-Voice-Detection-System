from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from models.model_loader import load_model, predict
import numpy as np
import time


# Load DS-CNN model
MODEL_PATH = "models/ds_cnn_model.pth"
model = load_model(MODEL_PATH)

THRESHOLD = 0.7  # confidence threshold


def main():
    print("🎤 Listening for STOP command...")

    stream = get_audio_stream()

    try:
        BUFFER_SIZE = 16000  # 1 second buffer
        audio_buffer = np.zeros(0, dtype=np.float32)
        COOLDOWN_TIME = 2  # seconds
        last_detected_time = 0
        while True:
            audio_chunk = read_audio_chunk(stream)

            # Append to buffer
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            # Keep last 1 second only
            if len(audio_buffer) > BUFFER_SIZE:
                audio_buffer = audio_buffer[-BUFFER_SIZE:]

            # Run VAD ONLY if buffer is full
            if len(audio_buffer) < BUFFER_SIZE:
                continue

            speech = apply_vad(audio_buffer)

            if speech is None:
                print("No speech detected...")
                continue
            else:
                print("Speech detected!")

            # Feature extraction
            features = extract_features(speech)

            # Prediction
            label, confidence = predict(model, features)

            print(f"Prediction: Label={label}, Confidence={confidence:.2f}")

            current_time = time.time()

            if label == 1 and confidence > 0.7:
                if current_time - last_detected_time > COOLDOWN_TIME:
                    print(f"🚨 STOP DETECTED! Confidence: {confidence:.2f}")
                    last_detected_time = current_time

    except KeyboardInterrupt:
        print("\nStopped.")
        close_stream(stream)


if __name__ == "__main__":
    main()