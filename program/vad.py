import webrtcvad
import numpy as np

# 🎯 CONFIG
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20  # 10, 20, or 30 ms only
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# 🔥 Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0–3 (2 or 3 recommended)


# 🧠 Convert float audio → int16 bytes
def float_to_pcm16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


# 🧩 Split audio into frames
def frame_generator(audio):
    num_samples = len(audio)
    offset = 0

    while offset + FRAME_SIZE <= num_samples:
        yield audio[offset:offset + FRAME_SIZE]
        offset += FRAME_SIZE


def apply_vad(audio, energy_threshold=500, min_speech_frames=5):

    speech_frames = []
    speech_buffer = []
    speech_count = 0

    for frame in frame_generator(audio):
        pcm_frame = float_to_pcm16(frame)

        frame_np = np.frombuffer(pcm_frame, dtype=np.int16)
        energy = np.sqrt(np.mean(frame_np**2))

        if energy < energy_threshold:
            is_speech = False
        else:
            is_speech = vad.is_speech(pcm_frame, SAMPLE_RATE)

        if is_speech:
            speech_count += 1
            speech_buffer.append(frame)
        else:
            speech_count = max(0, speech_count - 1)

            if speech_count >= min_speech_frames:
                speech_frames.extend(speech_buffer)

            speech_buffer = []

    if len(speech_frames) == 0:
        return None

    return np.concatenate(speech_frames)