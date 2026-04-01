import os
import librosa
import numpy as np
import soundfile as sf
import random


speech_dir = "dataset/raw/stop"   # STOP data
noise_dir = "dataset/raw/background_noise"     # long noise files

output_stop = "dataset/final/stop"
output_not_stop = "dataset/final/not_stop"

os.makedirs(output_stop, exist_ok=True)
os.makedirs(output_not_stop, exist_ok=True)

#  Parameters

SR = 16000
NOISE_LEVELS = [0.1, 0.2, 0.3]
AUG_PER_FILE = 4          # number of noisy versions per STOP
CHUNK_DURATION = 1.0      # seconds for noise chunks

# Utility Functions

def load_audio(path):
    audio, _ = librosa.load(path, sr=SR)
    audio, _ = librosa.effects.trim(audio)
    return audio

def normalize(audio):
    if np.max(np.abs(audio)) == 0:
        return audio
    return audio / np.max(np.abs(audio))

def get_random_noise_segment(noise, target_len):
    if len(noise) < target_len:
        repeat = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeat)

    if len(noise) == target_len:
        return noise

    max_start = len(noise) - target_len
    start = np.random.randint(0, max_start)

    return noise[start:start + target_len]

def mix_audio(speech, noise, level):
    speech = normalize(speech)
    noise = normalize(noise)

    noise = noise * level
    mixed = speech + noise

    return normalize(mixed)

def split_noise_into_chunks(noise, chunk_size):
    chunks = []
    for i in range(0, len(noise), chunk_size):
        chunk = noise[i:i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks

# CREATE STOP DATASET

speech_files = os.listdir(speech_dir)
noise_files = os.listdir(noise_dir)

print("Creating STOP dataset...")

stop_count = 0

for speech_file in speech_files:
    speech_path = os.path.join(speech_dir, speech_file)
    speech = load_audio(speech_path)

    # Save clean STOP
    sf.write(f"{output_stop}/stop_clean_{stop_count}.wav", speech, SR)
    stop_count += 1

    # Generate noisy versions
    for _ in range(AUG_PER_FILE):
        noise_file = random.choice(noise_files)
        noise_path = os.path.join(noise_dir, noise_file)

        noise = load_audio(noise_path)
        noise_segment = get_random_noise_segment(noise, len(speech))

        level = random.choice(NOISE_LEVELS)

        mixed = mix_audio(speech, noise_segment, level)

        sf.write(f"{output_stop}/stop_aug_{stop_count}.wav", mixed, SR)
        stop_count += 1

print(f"STOP samples created: {stop_count}")

# CREATE NOT_STOP DATASET

print("Creating NOT_STOP dataset...")

chunk_size = int(SR * CHUNK_DURATION)
not_stop_count = 0

for noise_file in noise_files:
    noise_path = os.path.join(noise_dir, noise_file)
    noise = load_audio(noise_path)

    chunks = split_noise_into_chunks(noise, chunk_size)

    for chunk in chunks:
        chunk = normalize(chunk)
        sf.write(f"{output_not_stop}/noise_{not_stop_count}.wav", chunk, SR)
        not_stop_count += 1

print(f"NOT_STOP samples created: {not_stop_count}")


print("Dataset ready for training!")