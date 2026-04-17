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

from flask import Flask, request, jsonify, render_template
from program.chatbot import ask

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


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


@app.route("/api/events")
def events():
    path = Path(os.getenv("DASHBOARD_EVENTS_PATH", "dashboard_events.jsonl"))
    if not path.exists():
        return jsonify([])
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            import json
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return jsonify(items)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
