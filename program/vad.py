import webrtcvad
import numpy as np

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20 
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0–3 

def float_to_pcm16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def frame_generator(audio):
    num_samples = len(audio)
    offset = 0

    while offset + FRAME_SIZE <= num_samples:
        yield audio[offset:offset + FRAME_SIZE]
        offset += FRAME_SIZE


def apply_vad(audio):
    speech_frames = []

    for frame in frame_generator(audio):
        pcm_frame = float_to_pcm16(frame)

        is_speech = vad.is_speech(pcm_frame, SAMPLE_RATE)

        if is_speech:
            speech_frames.append(frame)

    if len(speech_frames) == 0:
        return None

    return np.concatenate(speech_frames)