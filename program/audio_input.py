import pyaudio
import numpy as np
from assets.config import SAMPLE_RATE, FRAME_SIZE, CHANNELS

# 🎤 Initialize PyAudio
p = pyaudio.PyAudio()


def get_audio_stream():
    """
    Opens microphone stream
    """
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE
    )
    return stream


def read_audio_chunk(stream):
    """
    Reads one chunk of audio from mic
    """
    data = stream.read(FRAME_SIZE, exception_on_overflow=False)
    
    # Convert bytes → numpy array
    audio = np.frombuffer(data, dtype=np.int16)

    # Normalize to float [-1, 1]
    audio = audio.astype(np.float32) / 32768.0

    return audio


def close_stream(stream):
    """
    Close mic stream safely
    """
    stream.stop_stream()
    stream.close()
    p.terminate()