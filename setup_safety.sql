-- 1. Create the database
CREATE DATABASE IF NOT EXISTS factory_safety;
USE factory_safety;

-- 2. Create the table
CREATE TABLE IF NOT EXISTS voice_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    detected_text VARCHAR(50),
    alert_type VARCHAR(50),
    confidence FLOAT,
    audio_path VARCHAR(255)
);

-- 3. FIX: Change 'detection_history' to 'voice_events'
SELECT * FROM voice_events;