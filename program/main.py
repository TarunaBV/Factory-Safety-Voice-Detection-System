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

# 🔥 YOUR WORKING LOGIC IMPORTS
from program.audio_input import get_audio_stream, read_audio_chunk, close_stream
from program.vad import apply_vad
from program.feature_extraction import extract_features
from models.model_loader import load_model, predict

from database.crud import init_db, add_detection, get_history

# =========================
# 🚀 FASTAPI SETUP
# =========================
app = FastAPI()

init_db()

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

# =========================
# 🧠 MODEL
# =========================
model = load_model("models/ds_cnn_model.pth")

THRESHOLD = 0.8

# =========================
# 🎤 BACKGROUND AUDIO LOOP
# =========================
def audio_detection_loop():
    print("🎤 Background listening started...")

    stream = get_audio_stream()

    BUFFER_SIZE = 16000
    buffer = np.zeros(0, dtype=np.float32)

    COOLDOWN = 2
    last_time = 0

    while True:
        try:
            chunk = read_audio_chunk(stream)
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) > BUFFER_SIZE:
                buffer = buffer[-BUFFER_SIZE:]

            if len(buffer) < BUFFER_SIZE:
                continue

            speech = apply_vad(buffer)
            if speech is None:
                continue

            features = extract_features(buffer)
            label_id, label_name, conf = predict(model, features)

            print(f"{label_name} ({conf:.2f})")

            if label_name == "stop" and conf > THRESHOLD:
                if time.time() - last_time > COOLDOWN:

                    print("💾 SAVING TO DB...")

                    record = add_detection(
                        keyword_detected="stop",
                        status="DANGER",
                        confidence=float(conf)
                    )

                    print("✅ SAVED:", record.id)

                    last_time = time.time()

        except Exception as e:
            print("Error in audio loop:", e)

# =========================
# 🌐 ROUTES
# =========================
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
            "confidence": r.confidence
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

# =========================
# 🚀 START EVERYTHING
# =========================
def start_server():
    uvicorn.run("program.main:app",
                host="127.0.0.1",
                port=8000,
                reload=False)

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # 🎤 Start audio detection in background
    threading.Thread(target=audio_detection_loop, daemon=True).start()

    # 🌐 Open browser
    threading.Thread(target=open_browser).start()

    # 🚀 Run server
    start_server()