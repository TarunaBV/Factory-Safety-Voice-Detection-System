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
from program.keyword_spotting import load_spotter, train_keyword_collection
import os

from database.crud import init_db, add_detection, get_history

# Fast API setup
app = FastAPI()

init_db()

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

# Model
model = load_model("models/ds_cnn_model.pth")
AI_THRESHOLD = 0.85

# TEMPLATE SIGNATURE MODEL (Real-world Fingerprint)
template_spotter = load_spotter("models/stop_template_spotter.npz")
TEMPLATE_THRESHOLD = 0.50  # Increased for stricter precision

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

    while True:
        try:
            chunk = read_audio_chunk(stream)
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) > BUFFER_SIZE:
                buffer = buffer[-BUFFER_SIZE:]

            if len(buffer) < BUFFER_SIZE:
                continue

            current_time = time.time()
            if current_time - last_process_time < PROCESS_STRIDE:
                continue
                
            last_process_time = current_time

            # ENERGY GATE: Filter out quiet background chatter and ambient music
            rms = np.sqrt(np.mean(buffer**2))
            if rms < 0.010:  # Tightened for better chatter rejection
                continue

            speech = apply_vad(buffer)
            if speech is None:
                continue

            # STEP 1: AI Neural Verification
            features = extract_features(speech)
            label_id, label_name, conf = predict(model, features)

            # STEP 2: Template Signature Verification (Searching within the 1s buffer)
            # We use 'predict_audio' here because it 'slides' to find the best match
            template_result = template_spotter.predict_audio(buffer, window_hop_ms=100)
            template_score = template_result.score

            print(f"[{time.strftime('%H:%M:%S')}] {label_name.upper()} (AI: {conf:.2f}, Sig: {template_score:.2f})")

            # MANDATORY DUAL-VERIFICATION LOGIC
            # Both the AI brain AND the physical sound wave MUST agree it is a STOP.
            if (label_name == "stop" and conf > AI_THRESHOLD) and (template_score > TEMPLATE_THRESHOLD):
                if current_time - last_detected_time > COOLDOWN:
                    print(f"🚨DANGER: STOP CONFIRMED! (AI: {conf:.2f}, Sig: {template_score:.2f})")

                    record = add_detection(
                        keyword_detected="stop",
                        status="DANGER",
                        confidence=float((conf + template_score) / 2)
                    )
                    last_detected_time = current_time
            else:
                # Captured by AI but rejected by Template signature (likely casual speech)
                print(f"⚠️  IGNORED: AI detected '{label_name}' but template signature didn't match ({template_score:.2f})")

        except Exception as e:
            print("Error in audio loop:", e)

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

# Start server
def start_server():
    uvicorn.run("program.main:app",
                host="127.0.0.1",
                port=8000,
                reload=False)

def check_and_create_signatures():
    """Builds the signature models if they are missing, using generalized paths."""
    # Add any new keywords here that you want to verify via signatures
    keywords_to_verify = ["stop"] 
    
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