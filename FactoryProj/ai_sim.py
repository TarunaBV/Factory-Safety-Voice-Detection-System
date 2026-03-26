import sqlite3
import datetime
import random
import time

# fake dataset (AI output simulation)
possible_outputs = [
    ("fire", 0.95),
    ("help", 0.75),
    ("emergency", 0.92),

    ("normal", 0.80),
    ("machine noise", 0.70),
    ("background sound", 0.60),
    ("talking", 0.65),
    ("equipment noise", 0.75)
]

def save_to_db(text, confidence):
    conn = sqlite3.connect('database/database.db')
    cursor = conn.cursor()

    # check duplicate (last entry)
    cursor.execute('''
    SELECT detected_text FROM voice_events
    ORDER BY id DESC LIMIT 1
    ''')
    last = cursor.fetchone()

    if last and last[0] == text and confidence < 0.9:
        print("Duplicate low-confidence → skipping")
        conn.close()
        return

    # classify alert
    if text.lower() in ["fire", "emergency"] and confidence > 0.9:
        alert = "DANGER"

    elif text.lower() == "help" and confidence > 0.7:
        alert = "WARNING"

    else:
        alert = "NORMAL"

    cursor.execute('''
    INSERT INTO voice_events (timestamp, detected_text, alert_type, confidence, audio_path)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        str(datetime.datetime.now()),
        text,
        alert,
        confidence,
        ""
    ))

    conn.commit()
    conn.close()

    print(f"Saved: {text} | Confidence: {confidence} | Alert: {alert}")


# simulate AI running continuously
while True:
    text, confidence = random.choice(possible_outputs)

    print(f"AI detected: {text}")
    save_to_db(text, confidence)

    time.sleep(60)  # wait 3 seconds before next detection