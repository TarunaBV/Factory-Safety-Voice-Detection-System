import os
import librosa
import soundfile as sf
from tqdm import tqdm

# 📁 INPUT & OUTPUT PATHS
MUSAN_NOISE_PATH = r"D:\GITHUB\Factory-Safety-Voice-Detection-System\dataset\raw\noise"
OUTPUT_PATH = r"D:\GITHUB\Factory-Safety-Voice-Detection-System\dataset\raw\background_noise"

# ⚙️ CONFIG
TARGET_SR = 16000
CLIP_DURATION = 1  # seconds
CLIP_SAMPLES = TARGET_SR * CLIP_DURATION

# 📂 CREATE OUTPUT FOLDER
os.makedirs(OUTPUT_PATH, exist_ok=True)

def process_file(file_path, start_index):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        total_samples = len(audio)
        num_clips = total_samples // CLIP_SAMPLES

        saved_count = 0

        for i in range(num_clips):
            start = i * CLIP_SAMPLES
            end = start + CLIP_SAMPLES
            clip = audio[start:end]

            # Ensure exact length
            if len(clip) == CLIP_SAMPLES:
                filename = f"noise_{start_index + saved_count:06d}.wav"
                output_file = os.path.join(OUTPUT_PATH, filename)

                sf.write(output_file, clip, TARGET_SR)
                saved_count += 1

        return saved_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    file_list = []

    # 🔍 Collect all wav files
    for root, _, files in os.walk(MUSAN_NOISE_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_list.append(os.path.join(root, file))

    print(f"Found {len(file_list)} audio files")

    counter = 0

    # 🚀 Process all files
    for file_path in tqdm(file_list):
        count = process_file(file_path, counter)
        counter += count

    print(f"\n✅ Done! Generated {counter} noise clips")


if __name__ == "__main__":
    main()