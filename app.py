from pathlib import Path
import os


def _load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

from flask import Flask, request, jsonify, render_template, send_from_directory
from program.chatbot import ask
import json

app = Flask(__name__, template_folder="dashboard/templates", static_folder="dashboard/static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history")
def history():
    path = Path(os.getenv("DASHBOARD_EVENTS_PATH", "dashboard_events.jsonl"))
    if not path.exists():
        return jsonify({"data": []})
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            items.append({
                "timestamp": __import__('datetime').datetime.fromtimestamp(e["timestamp"]).strftime("%Y-%m-%d %H:%M:%S") if e.get("timestamp") else "",
                "raw_timestamp": e.get("timestamp", 0),
                "status": "DANGER" if e.get("emergency") else "NORMAL",
                "keyword_detected": e.get("keyword", ""),
                "confidence": e.get("confidence", 0),
            })
        except Exception:
            continue
    return jsonify({"data": list(reversed(items))})


@app.route("/transcript")
def transcript():
    return jsonify({"transcript": ""})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        answer = ask(question)
        return jsonify({"answer": answer})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
