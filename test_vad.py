import librosa
import soundfile as sf
from program.vad import apply_vad

audio_path = "test_audio.wav"
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

speech = apply_vad(audio)

if speech is not None:
    print("Speech detected!")
    print("Speech length:", len(speech) / 16000, "seconds")

    output_path = "vad_output.wav"
    sf.write(output_path, speech, 16000)

    print(f"Saved VAD output as {output_path}")

else:
    print("No speech detected")