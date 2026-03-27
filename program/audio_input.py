import pyaudio
import numpy as np
from assets.config import SAMPLE_RATE, FRAME_SIZE, CHANNELS

p = pyaudio.PyAudio()


def get_audio_stream():
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE
    )
    return stream


def read_audio_chunk(stream):
    data = stream.read(FRAME_SIZE, exception_on_overflow=False)
    
    # Convert bytes to numpy array
    audio = np.frombuffer(data, dtype=np.int16)

    # Normalize to float [-1, 1]
    audio = audio.astype(np.float32) / 32768.0

    return audio


def close_stream(stream):
    stream.stop_stream()
    stream.close()
    p.terminate()