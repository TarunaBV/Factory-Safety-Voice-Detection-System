import torch
import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME_DURATION = 20  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

model, utils = torch.hub.load(
    'snakers4/silero-vad',
    'silero_vad',
    force_reload=False
)

(get_speech_timestamps,
 _, _, _, collect_chunks) = utils

webrtc = webrtcvad.Vad(2)


def apply_vad(audio):

    audio = audio.astype(np.float32)

    frames = []
    for i in range(0, len(audio), FRAME_SIZE):
        frame = audio[i:i + FRAME_SIZE]
        if len(frame) == FRAME_SIZE:
            frames.append(frame)

    speech_frames = []
    for frame in frames:
        pcm16 = (frame * 32767).astype(np.int16)

        if webrtc.is_speech(pcm16.tobytes(), SAMPLE_RATE):
            speech_frames.append(frame)

    if len(speech_frames) == 0:
        return None

    filtered_audio = np.concatenate(speech_frames)

    audio_tensor = torch.from_numpy(filtered_audio)

    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=SAMPLE_RATE
        )

    if len(speech_timestamps) == 0:
        return None

    speech_audio = collect_chunks(speech_timestamps, audio_tensor)

    return speech_audio.numpy()