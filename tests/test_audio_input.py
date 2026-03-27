from program.audio_input import get_audio_stream, read_audio_chunk

stream = get_audio_stream()

print("Listening...")

for _ in range(50):
    audio = read_audio_chunk(stream)
    print(len(audio))
