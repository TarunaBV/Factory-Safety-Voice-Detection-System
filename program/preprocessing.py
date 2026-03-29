import os
import zipfile
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import gdown
from scipy.signal import butter, sosfilt

# Configuration
# Google Drive file ID — taken from your share link:
# https://drive.google.com/file/d/12Q2_SYykHNM1wHo-0cacGAclN-l5hN-n/view
DRIVE_FILE_ID  = "12Q2_SYykHNM1wHo-0cacGAclN-l5hN-n"

# Local folders
DOWNLOAD_PATH  = "final_zip.zip"              # Where the zip is saved after download
RAW_DATASET    = "raw_dataset"                # Where the zip is extracted
PREPROCESSED   = "preprocessed_dataset"       # Where cleaned audio files are saved

# Preprocessing settings
PROP_DECREASE          = 1.0    
THRESHOLD_DB           = -55.0  
GATE_WINDOW_MS         = 50.0   
PRE_EMPHASIS_COEF      = 0.97   
BANDPASS_LOW           = 300    
BANDPASS_HIGH          = 7500  
NOISE_PROFILE_DURATION = 0.5   


# Step 1 — Download dataset from Google Drive

def download_dataset():
    if os.path.exists(DOWNLOAD_PATH):
        print(f"[Download] Already exists: {DOWNLOAD_PATH} — skipping download.")
        return
    print(f"[Download] Downloading dataset from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, DOWNLOAD_PATH, quiet=False)
    print(f"[Download] Saved to: {DOWNLOAD_PATH}")

# Step 2 — Extract zip

def extract_dataset():
    if os.path.exists(RAW_DATASET) and os.listdir(RAW_DATASET):
        print(f"[Extract] Already extracted to: {RAW_DATASET} — skipping.")
        return
    print(f"[Extract] Extracting {DOWNLOAD_PATH} → {RAW_DATASET}/")
    os.makedirs(RAW_DATASET, exist_ok=True)
    with zipfile.ZipFile(DOWNLOAD_PATH, "r") as z:
        z.extractall(RAW_DATASET)
    print(f"[Extract] Done.")

# Step 3 — AudioPreprocessor (two_pass mode)

class AudioPreprocessor:
    def __init__(self):
        self.sr                    = None
        self.gate_window_samples   = None
        self.noise_profile_samples = None
        self.bandpass_high_clamped = None

    def _init_sr_dependent(self):
        self.gate_window_samples   = int(self.sr * GATE_WINDOW_MS / 1000)
        self.noise_profile_samples = int(self.sr * NOISE_PROFILE_DURATION)
        self.bandpass_high_clamped = min(BANDPASS_HIGH, self.sr // 2 - 500)

    def _normalize(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-6)

    def _bandpass_filter(self, audio):
        nyquist = 0.5 * self.sr
        sos = butter(
            5,
            [BANDPASS_LOW / nyquist, self.bandpass_high_clamped / nyquist],
            btype="band",
            output="sos",
        )
        return sosfilt(sos, audio)

    def _pre_emphasis(self, audio):
        return np.append(audio[0], audio[1:] - PRE_EMPHASIS_COEF * audio[:-1])

    def _de_emphasis(self, audio):
        result = np.zeros_like(audio)
        result[0] = audio[0]
        for i in range(1, len(audio)):
            result[i] = audio[i] + PRE_EMPHASIS_COEF * result[i - 1]
        return result

    def _apply_noise_gate(self, audio):
        window   = self.gate_window_samples
        n_frames = len(audio) // window
        mask     = np.ones(len(audio), dtype=bool)

        for i in range(n_frames):
            start = i * window
            end   = start + window
            rms   = np.sqrt(np.mean(audio[start:end] ** 2) + 1e-12)
            if 20 * np.log10(rms) < THRESHOLD_DB:
                mask[start:end] = False

        if len(audio) % window:
            chunk = audio[n_frames * window:]
            rms   = np.sqrt(np.mean(chunk ** 2) + 1e-12)
            if 20 * np.log10(rms) < THRESHOLD_DB:
                mask[n_frames * window:] = False

        return np.where(mask, audio, 0.0)

    def _reduce_noise(self, audio):
        noise_clip = audio[:self.noise_profile_samples]
        if len(noise_clip) == 0:
            print("    Warning: too short for noise profile — using non_stationary fallback.")
            return nr.reduce_noise(y=audio, sr=self.sr, stationary=False, prop_decrease=PROP_DECREASE)
        return nr.reduce_noise(
            y=audio, sr=self.sr,
            y_noise=noise_clip,
            stationary=True,
            prop_decrease=PROP_DECREASE,
        )

    def process(self, input_path, output_path):
        try:
            audio, self.sr = librosa.load(input_path, sr=None)
        except Exception as exc:
            print(f"    ERROR loading {input_path}: {exc}")
            return False

        self._init_sr_dependent()

        audio = self._bandpass_filter(audio)
        audio = self._pre_emphasis(audio)
        audio = self._reduce_noise(audio)
        audio = self._de_emphasis(audio)
        audio = self._apply_noise_gate(audio)
        audio = self._normalize(audio)

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, self.sr)
            return True
        except Exception as exc:
            print(f"    ERROR saving {output_path}: {exc}")
            return False

# Step 4 — Batch process entire dataset

def preprocess_dataset():
    processor = AudioPreprocessor()
    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    total = skipped = errors = 0

    for root, dirs, files in os.walk(RAW_DATASET):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue

            input_path = os.path.join(root, filename)

            relative   = os.path.relpath(input_path, RAW_DATASET)
            output_path = os.path.join(PREPROCESSED, relative)

            if os.path.exists(output_path):
                skipped += 1
                continue

            print(f"  Processing: {relative}")
            success = processor.process(input_path, output_path)
            if success:
                total += 1
            else:
                errors += 1

    print()
    print(f"[Preprocess] Done.")
    print(f"  Processed : {total} files")
    print(f"  Skipped   : {skipped} files (already existed)")
    print(f"  Errors    : {errors} files")
    print(f"  Output    : {os.path.abspath(PREPROCESSED)}/")


# Execution

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    preprocess_dataset()