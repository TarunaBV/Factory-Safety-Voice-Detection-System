import torch
import numpy as np

# Load model once (important)
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

SAMPLE_RATE = 16000


def apply_vad(audio):
    """
    Input: numpy array (float32)
    Output: speech-only numpy array OR None
    """

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio)

    # Get speech segments
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=SAMPLE_RATE
    )

    # If no speech
    if len(speech_timestamps) == 0:
        return None

    # Extract only speech parts
    speech_audio = collect_chunks(speech_timestamps, audio_tensor)

    # Convert back to numpy
    return speech_audio.numpy()