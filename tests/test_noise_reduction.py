import soundfile as sf
from assets.noise_reduction import clean_audio
import librosa
audio, sr = librosa.load("test_audio2.wav")

clean = clean_audio(audio, sr)

sf.write("clean.wav", clean, sr)

print("Done! Check clean.wav")