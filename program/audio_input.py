import numpy as np
import os
from assets.config import SAMPLE_RATE, FRAME_SIZE, CHANNELS

AUDIO_GAIN = float(os.getenv("AUDIO_GAIN", "1.0"))

try:
    import sounddevice as sd
except ModuleNotFoundError:
    sd = None

try:
    import pyaudio
except ModuleNotFoundError:
    pyaudio = None


_pyaudio_instance = None


class SoundDeviceAudioStream:
    def __init__(self):
        device = _resolve_input_device()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=FRAME_SIZE,
            dtype="int16",
            device=device,
        )
        self.stream.start()

    def read(self):
        data, _ = self.stream.read(FRAME_SIZE)
        return np.asarray(data).reshape(-1)

    def close(self):
        self.stream.stop()
        self.stream.close()


def get_audio_stream():
    global _pyaudio_instance

    if sd is not None:
        return SoundDeviceAudioStream()

    if pyaudio is not None:
        _pyaudio_instance = pyaudio.PyAudio()
        return _pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )

    raise ModuleNotFoundError(
        "No microphone backend is installed. Install one with: "
        "python -m pip install sounddevice"
    )


def _resolve_input_device():
    value = os.getenv("AUDIO_INPUT_DEVICE")
    if value is None or value.strip() == "":
        return None

    try:
        return int(value)
    except ValueError:
        return value


def read_audio_chunk(stream):
    if isinstance(stream, SoundDeviceAudioStream):
        audio = stream.read()
    else:
        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)

    # Normalize to float [-1, 1]
    audio = audio.astype(np.float32) / 32768.0
    audio = np.clip(audio * AUDIO_GAIN, -1.0, 1.0)

    return audio


def close_stream(stream):
    global _pyaudio_instance

    if isinstance(stream, SoundDeviceAudioStream):
        stream.close()
        return

    stream.stop_stream()
    stream.close()
    if _pyaudio_instance is not None:
        _pyaudio_instance.terminate()
        _pyaudio_instance = None
