from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DASHBOARD_EVENTS_PATH = os.getenv("DASHBOARD_EVENTS_PATH", "dashboard_events.jsonl")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

SYSTEM_PROMPT = """\
You are a factory safety dashboard assistant. You answer questions about voice-detected \
safety alert events logged from the factory floor.

Each event has these fields:
- alarm_type (STOP / FIRE / HELP)
- keyword (the detected word: stop/fire/help)
- severity (high/medium)
- emergency (true/false)
- zone, device_id
- timestamp (Unix epoch — convert to human-readable time when relevant)

Rules you MUST follow:
- ONLY discuss structured alert events. Do NOT mention speech recognition, \
transcripts, STT, raw audio text, microphone input, or any spoken words that \
are not part of an alert event.
- Be concise and factual. Use bullet points for lists of events.
- If no events match the question, say so clearly.
- Never make up events that are not in the data.
- When counting, be precise (e.g., "3 FIRE alerts were logged").
"""


def _load_events(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _events_to_context(events: list[dict]) -> str:
    if not events:
        return "No events logged yet."
    lines = []
    for e in events:
        ts = e.get("timestamp")
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            dt = "unknown time"
        lines.append(
            f"[{dt}] keyword={e.get('keyword')} alarm={e.get('alarm_type')} "
            f"action={e.get('action')} severity={e.get('severity')} "
            #f"conf={e.get('confidence', 0):.2f} zone={e.get('zone')} "
            f"device={e.get('device_id')} source={e.get('detection_source')}"
        )
    return "\n".join(lines)


def _call_groq(messages: list[dict]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Export it to use the chatbot.")

    # Merge system + user messages into a single user message as fallback
    # to support both chat models and classification models.
    merged = []
    system_text = ""
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            if system_text and m["role"] == "user":
                merged.append({"role": "user", "content": f"{system_text}\n\n{m['content']}"})
                system_text = ""  # Only prepend once
            else:
                merged.append(m)

    body = json.dumps({
        "model": DEFAULT_GROQ_MODEL,
        "messages": merged,
        "temperature": 0.2,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq request failed: {exc} — {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    return payload["choices"][0]["message"]["content"].strip()


def ask(question: str, events_path: str | Path = DASHBOARD_EVENTS_PATH) -> str:
    events = _load_events(events_path)
    events_context = _events_to_context(events)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here are all logged safety alert events:\n\n{events_context}\n\n"
                f"Question: {question}"
            ),
        },
    ]
    return _call_groq(messages)


def run_chat_loop(events_path: str | Path = DASHBOARD_EVENTS_PATH) -> None:
    events = _load_events(events_path)
    print(f"🏭 Factory Safety Dashboard Chatbot")
    print(f"   {len(events)} event(s) loaded from {events_path}")
    print("   Type your question or 'quit' to exit.\n")

    history: list[dict] = []
    events_context = _events_to_context(events)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye.")
            break

        history.append({"role": "user", "content": question})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Here are all logged safety alert events:\n\n{events_context}",
            },
            {"role": "assistant", "content": "Understood. I have the event data. Ask me anything."},
            *history,
        ]

        try:
            answer = _call_groq(messages)
        except RuntimeError as exc:
            print(f"❌ {exc}\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}\n")
