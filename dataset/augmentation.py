import os
import librosa
import numpy as np
import soundfile as sf
import random


# Existing
speech_dir = "dataset/raw/stop"
output_stop = "dataset/final/stop"

# NEW
help_dir = "dataset/raw/Help"
fire_dir = "dataset/raw/Fire"

output_help = "dataset/final/help"
output_fire = "dataset/final/fire"

noise_dir = "dataset/raw/_background_noise_"

# Create folders
os.makedirs(output_stop, exist_ok=True)
os.makedirs(output_help, exist_ok=True)
os.makedirs(output_fire, exist_ok=True)


SR = 16000
NOISE_LEVELS = [0.1, 0.2, 0.3]
AUG_PER_FILE = 4


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


noise_files = os.listdir(noise_dir)

def create_keyword_dataset(speech_dir, output_dir, label_name):
    speech_files = os.listdir(speech_dir)

    print(f"\n🔥 Creating {label_name.upper()} dataset...")

    count = 0

    for speech_file in speech_files:
        speech_path = os.path.join(speech_dir, speech_file)
        speech = load_audio(speech_path)

        # Save clean
        sf.write(f"{output_dir}/{label_name}_clean_{count}.wav", speech, SR)
        count += 1

        # Augment
        for _ in range(AUG_PER_FILE):
            noise_file = random.choice(noise_files)
            noise_path = os.path.join(noise_dir, noise_file)

            noise = load_audio(noise_path)
            noise_segment = get_random_noise_segment(noise, len(speech))

            level = random.choice(NOISE_LEVELS)

            mixed = mix_audio(speech, noise_segment, level)

            sf.write(f"{output_dir}/{label_name}_aug_{count}.wav", mixed, SR)
            count += 1

    print(f"✅ {label_name.upper()} samples created: {count}")

#create_keyword_dataset(speech_dir, output_stop, "stop")

create_keyword_dataset(help_dir, output_help, "help")
create_keyword_dataset(fire_dir, output_fire, "fire")

print("\n🚀 HELP and FIRE datasets added successfully!")