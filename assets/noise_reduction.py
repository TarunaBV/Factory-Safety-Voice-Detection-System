import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter


def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)


def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)


def bandpass_filter(audio, sr, low=300, high=3400):
    nyquist = 0.5 * sr
    low = low / nyquist
    high = high / nyquist

    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, audio)
