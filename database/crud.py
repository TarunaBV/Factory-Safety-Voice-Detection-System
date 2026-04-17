from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import pymysql
from .config import DATABASE_URL, MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE
from .models import Base, DetectionHistory

engine = create_engine(DATABASE_URL)

# 🔥 THREAD SAFE SESSION
SessionLocal = scoped_session(sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
))

def init_db():
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port=int(MYSQL_PORT)
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        conn.close()
        print(f"DB '{MYSQL_DATABASE}' ready.")
    except Exception as e:
        print("DB error:", e)

    Base.metadata.create_all(bind=engine)


def add_detection(keyword_detected: str, status: str, confidence: float):
    db = SessionLocal()
    try:
        record = DetectionHistory(
            keyword_detected=keyword_detected,
            status=status,
            confidence=confidence
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        print("✅ DB SAVED:", record.id)

        return record
    except Exception as e:
        db.rollback()
        print("DB ERROR:", e)
    finally:
        db.close()


def get_history(limit: int = 50):
    db = SessionLocal()
    try:
        data = db.query(DetectionHistory)\
            .order_by(DetectionHistory.timestamp.desc())\
            .limit(limit)\
            .all()
        return data
    finally:
        db.close()