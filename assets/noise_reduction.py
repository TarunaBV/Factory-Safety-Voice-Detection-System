import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter

def normalize(audio):
    """Brings the peak volume to a standard level."""
    return audio / (np.max(np.abs(audio)) + 1e-6)

def bandpass_filter(audio, sr, low=300, high=4000):
    """Removes frequencies outside the human voice range."""
    nyquist = 0.5 * sr
    low_cut = low / nyquist
    high_cut = high / nyquist
    b, a = butter(5, [low_cut, high_cut], btype='band')
    return lfilter(b, a, audio)

def aggressive_noise_reduce(audio, sr):
    """
    Performs spectral subtraction. 
    prop_decrease=1.0 removes 100% of detected stationary noise.
    """
    return nr.reduce_noise(
        y=audio, 
        sr=sr, 
        stationary=True, 
        prop_decrease=1.0, # Complete removal
        n_fft=4096         # Higher resolution for cleaner separation
    )

def apply_noise_gate(audio, threshold_db=-45):
    """
    Forces absolute silence (0.0) when the signal is below the threshold.
    This eliminates the 'hiss' or 'warble' left between words.
    """
    # We use a small window to avoid clipping individual wave peaks
    audio_abs = np.abs(audio)
    db_level = 20 * np.log10(audio_abs + 1e-6)
    
    # Create a mask: 1.0 (keep) if above threshold, 0.0 (mute) if below
    gate_mask = (db_level > threshold_db).astype(float)
    
    return audio * gate_mask

def super_clean_pipeline(audio, sr):
    """The full chain for maximum background removal."""
    # 1. Filter out rumble and high-end hiss first
    processed = bandpass_filter(audio, sr)
    
    # 2. Subtract the background noise profile
    processed = aggressive_noise_reduce(processed, sr)
    
    # 3. Kill the remaining 'ghost' artifacts with a gate
    processed = apply_noise_gate(processed, threshold_db=-40)
    
    # 4. Bring the voice back to a clear volume
    return normalize(processed)

def clean_audio(audio, sr):
    audio = normalize(audio)
    audio = bandpass_filter(audio, sr)
    audio = apply_noise_gate(audio)
    audio = super_clean_pipeline(audio,sr)
    return audio
