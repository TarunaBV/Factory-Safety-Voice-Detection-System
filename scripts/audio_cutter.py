import librosa
import soundfile as sf
import os

audio, sr = librosa.load("D:/Intenship/Factory Safety Voice Detection/Factory-Safety-Voice-Detection-System/Recording.wav", sr=16000)

chunk_size = sr  # 1 second
output_dir = "dataset/final/other_speech"
os.makedirs(output_dir, exist_ok=True)

count = 0

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]

    if len(chunk) == chunk_size:
        sf.write(f"{output_dir}/speech_{count}.wav", chunk, sr)
        count += 1

print("Done! Created", count, "samples")