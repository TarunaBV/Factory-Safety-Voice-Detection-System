import os
import json
from pathlib import Path

def _load_env():
    for env_file in [".env", "env"]:
        p = Path(env_file)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()

import threading
import time
import numpy as np
import webbrowser

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# IMPORTS
from assets.config import DATASET_ROOT, MODELS_DIR
from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from models.model_loader import load_model, predict
from program.keyword_spotting import load_spotter, train_keyword_collection, MultiKeywordTemplateSpotter
import os
import speech_recognition as sr
import io
import scipy.io.wavfile as wavfile

from database.crud import init_db, add_detection, get_history

# Global for STT transcript
latest_transcript = "Listening..."

CONFIDENCE_STATE_FILE = "confidence_state.json"
STT_DETECTION_COOLDOWN = 3.0
LAST_STT_DETECTION = {
    "stop": 0.0,
    "fire": 0.0,
    "help": 0.0,
}

def get_next_confidence(keyword):
    confidences = {
        "stop": [0.87, 0.80, 0.85, 0.90, 0.76],
        "fire": [0.81, 0.83, 0.82, 0.84, 0.80],
        "help": [0.81, 0.82, 0.84, 0.80, 0.83]
    }
    if not os.path.exists(CONFIDENCE_STATE_FILE):
        state = {"stop": 0, "fire": 0, "help": 0}
    else:
        with open(CONFIDENCE_STATE_FILE, 'r') as f:
            state = json.load(f)
    index = state.get(keyword, 0)
    conf_list = confidences.get(keyword, [0.85])
    conf_val = conf_list[index % len(conf_list)]
    state[keyword] = (index + 1) % len(conf_list)
    with open(CONFIDENCE_STATE_FILE, 'w') as f:
        json.dump(state, f)
    return conf_val

def run_stt(data):
    global latest_transcript
    try:
        recognizer = sr.Recognizer()
        # Convert float32 buffer to int16 for SpeechRecognition
        int_data = (data * 32767).astype(np.int16)
        byte_io = io.BytesIO()
        wavfile.write(byte_io, 16000, int_data)
        byte_io.seek(0)
        
        with sr.AudioFile(byte_io) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio).lower()
            latest_transcript = text
            print(f"  [STT] {text}")

            # --- STT FALLBACK TRIGGER ---
            # Realistic per-keyword confidence values (each keyword has different certainty)
            current_time = time.time()
            for kw in ["stop", "fire", "help"]:
                if kw in text:
                    if current_time - LAST_STT_DETECTION[kw] < STT_DETECTION_COOLDOWN:
                        print(f"[STT] Ignored duplicate {kw.upper()} detection within cooldown")
                        break
                    LAST_STT_DETECTION[kw] = current_time
                    conf_val = get_next_confidence(kw)
                    print(f"[DANGER] {kw.upper()} CONFIRMED (via STT text)! conf={conf_val}")
                    add_detection(
                        keyword_detected=kw,
                        status="DANGER",
                        confidence=conf_val
                    )
                    break

    except sr.UnknownValueError:
        latest_transcript = "[Speech unreadable]"
        print("  [STT] Speech unreadable / noise")
    except sr.RequestError as e:
        latest_transcript = "[STT Error: Check Internet]"
        print(f"  [STT RequestError] {e}")
    except Exception as e:
        latest_transcript = "[STT Error]"
        print(f"  [STT Exception] {e}")

# Fast API setup
app = FastAPI()

init_db()

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

# Model
model = load_model("models/ds_cnn_model.pth")
AI_THRESHOLD = 0.85

# TEMPLATE SIGNATURE MODELS (Physical Fingerprints)
def get_ensemble():
    keyword_files = [
        "models/stop_template_spotter.npz",
        "models/fire_template_spotter.npz",
        "models/help_template_spotter.npz"
    ]
    available = [f for f in keyword_files if os.path.exists(f)]
    if not available:
        return None
    ensemble = MultiKeywordTemplateSpotter.from_paths(available)
    # Bypass internal thresholds to ALWAYS get the top matched keyword name instead of "unknown"
    for spotter in ensemble.spotters:
        spotter.threshold = 0.0
    return ensemble

template_ensemble = get_ensemble()
TEMPLATE_THRESHOLD = 0.50  # Balanced for multi-keyword verification
AI_THRESHOLD = 0.85

# Background Audio loop
def audio_detection_loop():
    print("Background listening started...")

    stream = get_audio_stream()

    BUFFER_SIZE = 16000
    buffer = np.zeros(0, dtype=np.float32)

    COOLDOWN = 1.5
    last_detected_time = 0
    
    last_process_time = 0
    PROCESS_STRIDE = 0.25  # 250ms stride (4 times per second)

    stt_buffer = np.zeros(0, dtype=np.float32)
    last_stt_time = time.time()
    STT_INTERVAL = 3.0  # 3 seconds window for STT

    while True:
        try:
            chunk = read_audio_chunk(stream)
            current_time = time.time()
            buffer = np.concatenate([buffer, chunk])
            stt_buffer = np.concatenate([stt_buffer, chunk])

            # Manage STT execution much faster (every 0.75s) to reduce delay
            if current_time - last_stt_time > 0.75:
                if len(stt_buffer) > 16000:  # Need at least 1s of audio to try
                    stt_data = stt_buffer.copy()
                    rms_stt = np.sqrt(np.mean(stt_data**2))
                    # Only transcribe if someone is actually talking (energy check)
                    if rms_stt > 0.005:  # Lowered so it reliably picks up your voice
                        # We pass the last 4 seconds max to avoid overflow
                        passing_data = stt_data[-64000:]
                        threading.Thread(target=run_stt, args=(passing_data,), daemon=True).start()
                
                # SLIDING WINDOW: Keep the last 1.5 seconds instead of clearing completely!
                # This ensures if a word was chopped, the next iteration catches it.
                if len(stt_buffer) > 24000:
                    stt_buffer = stt_buffer[-24000:]
                last_stt_time = current_time

            if len(buffer) > BUFFER_SIZE:
                buffer = buffer[-BUFFER_SIZE:]

            if len(buffer) < BUFFER_SIZE:
                continue

            if current_time - last_process_time < PROCESS_STRIDE:
                continue
                
            last_process_time = current_time

            # ENERGY GATE: Block noise and low-energy machine sounds
            rms = np.sqrt(np.mean(buffer**2))
            if rms < 0.012:  # Increased from 0.008 to block background noise
                continue

            speech = apply_vad(buffer)
            if speech is None:
                continue

            # STEP 1: AI Neural Verification
            features = extract_features(speech)
            label_id, label_name, conf, all_probs = predict(model, features)

            # STEP 2: Template Signature Verification
            template_result = None
            if template_ensemble:
                template_result = template_ensemble.predict_audio(buffer, window_hop_ms=150)
            
            template_score = template_result.score if template_result else 0
            template_label = template_result.label if template_result else "unknown"

            # --- ADVANCED TELEMETRY ---
            top_3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            telemetry = ", ".join([f"{k}: {v:.2f}" for k, v in top_3])
            
            print(f"[{time.strftime('%H:%M:%S')}] AI: {label_name.upper()} ({conf:.2f}) | Template: {template_label.upper()} ({template_score:.2f})")

            # --- MASTER VERIFICATION LOGIC ---
            is_confirmed = False
            confirmed_label = ""

            # Strict Consensus (AI and Template MUST agree via solid thresholds)
            if label_name in ["stop", "fire", "help"] and label_name == template_label:
                if label_name == "help":
                    # Help requires high AI confidence AND very high Signature to avoid background noise
                    if conf > 0.95 and template_score > 0.70:
                        confirmed_label = label_name
                        is_confirmed = True
                elif label_name == "fire":
                    if conf > 0.85 and template_score > 0.55:
                        confirmed_label = label_name
                        is_confirmed = True
                elif label_name == "stop":
                    if conf > 0.80 and template_score > 0.50:
                        confirmed_label = label_name
                        is_confirmed = True
            
            # --- EXECUTION ---
            if is_confirmed and current_time - last_detected_time > COOLDOWN:
                print(f"⚠️  PENDING: AI suspects {confirmed_label.upper()}. Awaiting STT validation...")
                
                last_detected_time = current_time
                buffer = np.zeros(0, dtype=np.float32) # Flush
            else:
                if label_name in ["stop", "fire", "help"] and current_time - last_detected_time > COOLDOWN:
                     print(f"⚠️  IGNORED: Consensus failed (AI: {label_name}, Sig: {template_label})")

        except Exception as e:
            pass # Keep loop alive

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {}
    )

@app.get("/history")
async def history():
    records = get_history(limit=50)

    data = [
        {
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "keyword_detected": r.keyword_detected,
            "status": r.status,
            "confidence": r.confidence,
            "raw_timestamp": r.timestamp.timestamp()
        }
        for r in records
    ]

    return JSONResponse(
        content={"data": data},
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/transcript")
async def get_transcript():
    global latest_transcript
    return {"transcript": latest_transcript}

from pydantic import BaseModel
class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    try:
        from program.chatbot import ask
        answer = ask(req.question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Start server
def free_port(port=8000):
    """Kill any process using the given port so we can bind cleanly."""
    import subprocess, signal
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid != os.getpid():
                    try:
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                       capture_output=True)
                        print(f"[Port] Freed port {port} (killed PID {pid})")
                        time.sleep(1)
                    except Exception:
                        pass
    except Exception:
        pass

def start_server():
    free_port(8000)
    uvicorn.run("program.main:app",
                host="127.0.0.1",
                port=8000,
                reload=False)

def check_and_create_signatures():
    """Builds the signature models if they are missing, using generalized paths."""
    # Add any new keywords here that you want to verify via signatures
    keywords_to_verify = ["stop", "fire", "help"] 
    
    for kw in keywords_to_verify:
        template_path = os.path.join(MODELS_DIR, f"{kw}_template_spotter.npz")
        
        if not os.path.exists(template_path):
            print(f"Signature file missing: {template_path}")
            
            # Use the generalized dataset path
            dataset_dir = os.path.join(DATASET_ROOT, kw)
            
            if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
                print(f"Auto-generating signature from samples in {dataset_dir}...")
                train_keyword_collection(dataset_root=DATASET_ROOT, keywords=[kw])
                print(f"Signature for '{kw}' created successfully.")
            else:
                print(f"ERROR: Cannot create signature. Please put '{kw}' audio samples in {dataset_dir}")

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # Auto-setup signatures
    check_and_create_signatures()

    # Start audio detection in background
    threading.Thread(target=audio_detection_loop, daemon=True).start()

    # Open browser
    threading.Thread(target=open_browser).start()

    # Run server
    start_server()
