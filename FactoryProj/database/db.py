import sqlite3

conn = sqlite3.connect('database/database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS voice_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    detected_text TEXT,
    alert_type TEXT,
    confidence REAL,
    audio_path TEXT
)
''')

conn.commit()
conn.close()

print("Table created successfully")