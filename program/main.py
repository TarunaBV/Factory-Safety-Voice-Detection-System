from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import FeatureConfig, extract_features, normalize_audio
from program.agentic_alert import decide_alert, explain_decision, make_detection_context
from program.dashboard import event_to_json, publish_dashboard_event
from program.keyword_spotting import TemplateKeywordSpotter, MultiKeywordTemplateSpotter, load_spotters_from_directory
from models.model_loader import load_model, predict

from collections import deque
import os
import numpy as np
import time
import torch


MODEL_PATH = "models/ds_cnn_model.pth"

# 🔥 DEVICE SUPPORT (CPU/GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(MODEL_PATH, device=DEVICE)

# 🔥 Template spotters — load all *_template_spotter.npz from models/ (stop, fire, help, ...)
try:
    _spotters = load_spotters_from_directory("models")
    template_spotter = MultiKeywordTemplateSpotter(_spotters)
    _kws = [s.keyword for s in _spotters]
    print(f"✅ Template spotters loaded: {_kws}")
except Exception as e:
    template_spotter = None
    print(f"⚠️  Template spotters not loaded: {e}")

THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.60"))
ENERGY_SPEECH_THRESHOLD = float(os.getenv("ENERGY_SPEECH_THRESHOLD", "0.0025"))
ENERGY_SILENCE_THRESHOLD = float(os.getenv("ENERGY_SILENCE_THRESHOLD", "0.0010"))
DEBUG_MIC_LEVELS = os.getenv("DEBUG_MIC_LEVELS", "1") == "1"
MIN_UTTERANCE_SECONDS = float(os.getenv("MIN_UTTERANCE_SECONDS", "0.30"))
DSCNN_FEATURE_CONFIG = FeatureConfig(pre_emphasis=0.97)
TEMPLATE_WINDOW_HOP_MS = int(os.getenv("TEMPLATE_WINDOW_HOP_MS", "50"))


def select_loudest_window(audio: np.ndarray, window_size: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size <= window_size:
        return audio

    hop_size = max(1, window_size // 4)
    best_start = 0
    best_rms = -1.0

    for start in range(0, audio.size - window_size + 1, hop_size):
        window = audio[start : start + window_size]
        rms = float(np.sqrt(np.mean(np.square(window))))
        if rms > best_rms:
            best_rms = rms
            best_start = start

    return audio[best_start : best_start + window_size]


def main():
    print("🎤 Listening for STOP command...")

    stream = get_audio_stream()

    BUFFER_SIZE = 16000
    audio_buffer = np.zeros(0, dtype=np.float32)

    COOLDOWN_TIME = 2
    last_detected_time = 0

    speech_active = False
    silence_counter = 0
    SILENCE_LIMIT = 5
    speech_segments = []
    pre_roll_chunks = deque(maxlen=max(1, BUFFER_SIZE // 320))
    chunk_counter = 0

    try:
        while True:
            audio_chunk = read_audio_chunk(stream)
            chunk_counter += 1
            chunk_rms = float(np.sqrt(np.mean(np.square(audio_chunk))))

            if DEBUG_MIC_LEVELS and chunk_counter % 25 == 0:
                print(f"Mic level rms={chunk_rms:.4f}")

            pre_roll_chunks.append(audio_chunk)

            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            # 🔥 keep only last 1 second audio
            if len(audio_buffer) > BUFFER_SIZE:
                audio_buffer = audio_buffer[-BUFFER_SIZE:]

            if len(audio_buffer) < BUFFER_SIZE:
                continue

            speech = apply_vad(audio_buffer)

            energy_speech = chunk_rms >= ENERGY_SPEECH_THRESHOLD

            if speech is not None or energy_speech:
                if not speech_active:
                    speech_segments = list(pre_roll_chunks)
                speech_active = True
                silence_counter = 0
                speech_segments.append(audio_chunk)
                continue

            else:
                if speech_active:
                    if chunk_rms <= ENERGY_SILENCE_THRESHOLD:
                        silence_counter += 1
                    else:
                        speech_segments.append(audio_chunk)
                        silence_counter = 0

                    if silence_counter < SILENCE_LIMIT:
                        continue

                    print("🔍 Processing speech...")

                    if speech_segments:
                        utterance_audio = np.concatenate(speech_segments).astype(np.float32)
                    else:
                        utterance_audio = audio_buffer

                    utterance_rms = float(np.sqrt(np.mean(np.square(utterance_audio))))
                    utterance_seconds = len(utterance_audio) / 16000
                    if utterance_seconds < MIN_UTTERANCE_SECONDS:
                        print(
                            f"⚠️  Ignored short audio fragment: {utterance_seconds:.2f}s "
                            f"(min={MIN_UTTERANCE_SECONDS:.2f}s), rms={utterance_rms:.4f} — "
                            f"try speaking a bit longer"
                        )
                        speech_active = False
                        speech_segments = []
                        continue

                    keyword_audio = select_loudest_window(utterance_audio, BUFFER_SIZE)
                    keyword_rms = float(np.sqrt(np.mean(np.square(keyword_audio))))
                    print(
                        f"Captured speech: {utterance_seconds:.2f}s, "
                        f"rms={utterance_rms:.4f}, keyword_window_rms={keyword_rms:.4f}"
                    )

                    # --- DS-CNN detector ---
                    # Normalize amplitude before inference so quiet mic audio isn't
                    # confused with background_noise by the model.
                    normalized_audio = normalize_audio(keyword_audio)
                    features = extract_features(normalized_audio, config=DSCNN_FEATURE_CONFIG)

                    label_idx, label_name, confidence = predict(
                        model, features, device=DEVICE
                    )
                    from program.agentic_alert import CRITICAL_KEYWORDS
                    dscnn_is_keyword = label_name.lower() in CRITICAL_KEYWORDS and confidence >= THRESHOLD
                    print(
                        f"[DS-CNN]   '{label_name}' conf={confidence:.2f} "
                        f"(thr={THRESHOLD:.2f}) "
                        f"{'✅' if dscnn_is_keyword else '❌'}"
                    )

                    # --- Template spotter (cosine similarity, amplitude-agnostic) ---
                    template_detected = False
                    template_label = "unknown"
                    template_score = 0.0
                    template_threshold = 0.0
                    if template_spotter is not None:
                        tmpl_pred = template_spotter.predict_audio(
                            utterance_audio,
                            window_hop_ms=TEMPLATE_WINDOW_HOP_MS,
                        )
                        template_detected = tmpl_pred.detected
                        template_label = tmpl_pred.label
                        template_score = tmpl_pred.score
                        template_threshold = tmpl_pred.threshold
                        print(
                            f"[Template] '{tmpl_pred.label}' score={tmpl_pred.score:.3f} "
                            f"(thr={tmpl_pred.threshold:.3f}) "
                            f"{'✅' if template_detected else '❌'}"
                        )

                    # Either detector can trigger — union strategy
                    detected_keyword = dscnn_is_keyword or template_detected
                    if dscnn_is_keyword:
                        effective_label = label_name.lower()
                        effective_confidence = confidence
                        effective_threshold = THRESHOLD
                        detection_source = "DS-CNN"
                    elif template_detected:
                        effective_label = template_label
                        effective_confidence = template_score
                        effective_threshold = template_threshold
                        detection_source = "Template"
                    else:
                        effective_label = label_name
                        effective_confidence = confidence
                        effective_threshold = THRESHOLD
                        detection_source = "none"

                    current_time = time.time()

                    context = make_detection_context(
                        keyword=effective_label,
                        confidence=effective_confidence,
                        threshold=effective_threshold,
                        audio_rms=keyword_rms,
                    )
                    decision = decide_alert(context)

                    print(f"Agent Decision: {decision.action.upper()} ({decision.source})")
                    print(explain_decision(decision))

                    if decision.emergency:
                        if current_time - last_detected_time > COOLDOWN_TIME:
                            dashboard_event = publish_dashboard_event(
                                context,
                                decision,
                                detection_source=detection_source,
                            )
                            print(
                                f"🚨 ALERT TRIGGERED! [via {detection_source}] "
                                f"type={decision.alarm_type} action={decision.action} "
                                f"keyword='{effective_label}' conf={effective_confidence:.2f}"
                            )
                            print(f"Dashboard Event: {event_to_json(dashboard_event)}")
                            last_detected_time = current_time
                        else:
                            remaining = COOLDOWN_TIME - (current_time - last_detected_time)
                            print(f"⏳ Alert suppressed by cooldown ({remaining:.1f}s remaining)")
                    elif not detected_keyword:
                        print(
                            f"   → No safety alert detected "
                            f"(DS-CNN={label_name}, template={template_label if template_detected else 'no match'})"
                        )

                    speech_active = False
                    speech_segments = []

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        close_stream(stream)


if __name__ == "__main__":
    main()
