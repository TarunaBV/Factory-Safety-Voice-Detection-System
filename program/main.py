from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from models.model_loader import load_model, predict

# Load DS-CNN model
MODEL_PATH = "models/ds_cnn_model.pth"
model = load_model(MODEL_PATH)

THRESHOLD = 0.7  # confidence threshold


def main():
    print("🎤 Listening for STOP command...")

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

            # Step 3: DS-CNN prediction
            label, confidence = predict(model, features)

            # Step 4: Decision
            if label == 1 and confidence > THRESHOLD:
                print(f"STOP DETECTED! Confidence: {confidence:.2f}")
            else:
                print(f"Ignored ({confidence:.2f})")

    except KeyboardInterrupt:
        print("\nStopped.")
        close_stream(stream)


if __name__ == "__main__":
    main()