import sqlite3
from datetime import datetime

DB_PATH = "sensei_memory.db"


def init_db():
    """
    Creates the database and table if they don't already exist.
    Call this once when the app starts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            source_pdf TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_interaction(question, answer, source_pdf="unknown"):
    """
    Saves one Q&A exchange to the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (timestamp, question, answer, source_pdf)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            question,
            answer,
            source_pdf
        )
    )

    conn.commit()
    conn.close()


def get_all_history():
    """
    Returns every past Q&A exchange, most recent first.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT timestamp, question, answer FROM history ORDER BY id DESC"
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_total_questions_asked():
    """
    Returns the total number of questions stored.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM history")

    count = cursor.fetchone()[0]

    conn.close()

    return count



