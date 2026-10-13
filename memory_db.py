"""
SENSEI V2 — Persistent Learning Database

Responsibilities:
- Preserve existing interaction history
- Store documents, concepts, and questions
- Record student attempts and learning signals
- Estimate concept mastery
- Schedule personalized revision

Database: SQLite
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = Path(__file__).resolve().parent / "sensei_memory.db"

DEFAULT_LEARNER_ID = "default_learner"

REVIEW_INTERVALS = {
    "incorrect": 1,
    "low": 1,
    "developing": 3,
    "good": 7,
    "mastered": 14,
}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    """Create a database connection with foreign keys enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _now() -> datetime:
    """Return the current local datetime."""
    return datetime.now()


def _timestamp() -> str:
    return _now().isoformat(timespec="seconds")


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(row) if row is not None else None


# ============================================================
# V1 COMPATIBILITY: INTERACTION HISTORY
# ============================================================

def init_db() -> None:
    """Initialize the existing V1 interaction history table."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_pdf TEXT,
                timestamp TEXT NOT NULL
            )
        """)


def log_interaction(
    question: str,
    answer: str,
    source_pdf: str = "unknown",
) -> int:
    """Save a notes-Q&A interaction and return its history ID."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO history
                (question, answer, source_pdf, timestamp)
            VALUES (?, ?, ?, ?)
        """, (question, answer, source_pdf, _timestamp()))
        return cursor.lastrowid


def get_all_history(limit: Optional[int] = None) -> list[dict]:
    """Return interaction history, newest first."""
    with get_connection() as conn:
        if limit is None:
            rows = conn.execute("""
                SELECT * FROM history
                ORDER BY id DESC
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM history
                ORDER BY id DESC
                LIMIT ?
            """, (max(0, int(limit)),)).fetchall()

        return [dict(row) for row in rows]


def get_total_questions_asked() -> int:
    """Return the total number of saved interactions."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM history"
        ).fetchone()
        return row["total"]


# ============================================================
# V2 SCHEMA
# ============================================================

def init_learning_db() -> None:
    """
    Create the V2 learning tables without deleting V1 data.

    Safe to call repeatedly.
    """
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_documents (
                document_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'registered',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                concept_key TEXT NOT NULL,
                concept_name TEXT NOT NULL,
                source_pages TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES learning_documents(document_id)
                    ON DELETE CASCADE,
                UNIQUE (document_id, concept_key)
            );

            CREATE TABLE IF NOT EXISTS question_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                concept_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                question_type TEXT NOT NULL,
                answer_key TEXT NOT NULL,
                source_reference TEXT NOT NULL DEFAULT '',
                difficulty INTEGER NOT NULL DEFAULT 1
                    CHECK (difficulty BETWEEN 1 AND 5),
                active INTEGER NOT NULL DEFAULT 1
                    CHECK (active IN (0, 1)),
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES learning_documents(document_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (concept_id)
                    REFERENCES concepts(id)
                    ON DELETE CASCADE,
                UNIQUE (concept_id, question)
            );

            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                concept_id INTEGER NOT NULL,
                question_id INTEGER,
                submitted_answer TEXT NOT NULL DEFAULT '',
                is_correct INTEGER NOT NULL
                    CHECK (is_correct IN (0, 1)),
                attempts INTEGER NOT NULL DEFAULT 1
                    CHECK (attempts >= 1),
                hints_used INTEGER NOT NULL DEFAULT 0
                    CHECK (hints_used >= 0),
                response_seconds REAL NOT NULL DEFAULT 0
                    CHECK (response_seconds >= 0),
                answered_at TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES learning_documents(document_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (concept_id)
                    REFERENCES concepts(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (question_id)
                    REFERENCES question_bank(id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS concept_mastery (
                learner_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                concept_id INTEGER NOT NULL,
                mastery_score REAL NOT NULL DEFAULT 0
                    CHECK (mastery_score BETWEEN 0 AND 100),
                attempts INTEGER NOT NULL DEFAULT 0,
                correct_answers INTEGER NOT NULL DEFAULT 0,
                last_tested_at TEXT,
                next_review_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (learner_id, document_id, concept_id),
                FOREIGN KEY (document_id)
                    REFERENCES learning_documents(document_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (concept_id)
                    REFERENCES concepts(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                concept_id INTEGER NOT NULL,
                due_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'completed', 'cancelled')),
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (document_id)
                    REFERENCES learning_documents(document_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (concept_id)
                    REFERENCES concepts(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_document
                ON concepts(document_id);

            CREATE INDEX IF NOT EXISTS idx_questions_concept
                ON question_bank(concept_id, active);

            CREATE INDEX IF NOT EXISTS idx_events_learner_concept
                ON learning_events(learner_id, concept_id, answered_at);

            CREATE INDEX IF NOT EXISTS idx_reviews_due
                ON review_queue(learner_id, status, due_at);

            
            CREATE TABLE IF NOT EXISTS pomodoro_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL DEFAULT 'default_learner',
                session_type TEXT NOT NULL DEFAULT 'focus'
                    CHECK (session_type IN ('focus', 'short_break', 'long_break')),
                planned_minutes INTEGER NOT NULL,
                completed_minutes INTEGER NOT NULL,
                completed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pomodoro_completed
                ON pomodoro_sessions(learner_id, completed_at);

            CREATE TABLE IF NOT EXISTS todo_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL DEFAULT 'default_learner',
                title TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'General',
                priority TEXT NOT NULL DEFAULT 'Medium'
                    CHECK (priority IN ('Low', 'Medium', 'High')),
                is_completed INTEGER NOT NULL DEFAULT 0
                    CHECK (is_completed IN (0, 1)),
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_todo_tasks
                ON todo_tasks(learner_id, is_completed, created_at);
        """)


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

def register_learning_document(
    document_id: str,
    document_name: str,
) -> None:
    """Register or update a document without resetting its progress."""
    now = _timestamp()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO learning_documents (
                document_id, document_name, status, created_at, updated_at
            )
            VALUES (?, ?, 'registered', ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                document_name = excluded.document_name,
                updated_at = excluded.updated_at
        """, (document_id, document_name, now, now))


def update_document_status(document_id: str, status: str) -> None:
    """Update document processing status."""
    allowed = {"registered", "processing", "ready", "failed"}

    if status not in allowed:
        raise ValueError(f"Status must be one of: {sorted(allowed)}")

    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE learning_documents
            SET status = ?, updated_at = ?
            WHERE document_id = ?
        """, (status, _timestamp(), document_id))

        if cursor.rowcount == 0:
            raise ValueError(f"Unknown document: {document_id}")


def get_learning_documents() -> list[dict]:
    """Return registered learning documents."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM learning_documents
            ORDER BY created_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


# ============================================================
# CONCEPT MANAGEMENT
# ============================================================

def save_concept(
    document_id: str,
    concept_key: str,
    concept_name: str,
    source_pages: str = "",
) -> int:
    """Save a concept and return its stable database ID."""
    now = _timestamp()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO concepts (
                document_id, concept_key, concept_name,
                source_pages, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(document_id, concept_key) DO UPDATE SET
                concept_name = excluded.concept_name,
                source_pages = excluded.source_pages
        """, (
            document_id,
            concept_key,
            concept_name,
            source_pages,
            now,
        ))

        row = conn.execute("""
            SELECT id FROM concepts
            WHERE document_id = ? AND concept_key = ?
        """, (document_id, concept_key)).fetchone()

        return row["id"]


def get_document_concepts(document_id: str) -> list[dict]:
    """Return concepts belonging to a document."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM concepts
            WHERE document_id = ?
            ORDER BY id
        """, (document_id,)).fetchall()
        return [dict(row) for row in rows]


def get_concept(concept_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM concepts WHERE id = ?
        """, (concept_id,)).fetchone()
        return _row_to_dict(row)


# ============================================================
# QUESTION BANK
# ============================================================

def save_question(
    document_id: str,
    concept_id: int,
    question: str,
    question_type: str,
    answer_key: Any,
    source_reference: str = "",
    difficulty: int = 1,
) -> Optional[int]:
    """
    Save a question.

    answer_key may be a dictionary or string. Dictionaries are
    serialized as JSON. Returns None for duplicate questions.
    """
    allowed_types = {
        "multiple_choice",
        "short_answer",
        "true_false",
    }

    if question_type not in allowed_types:
        raise ValueError(
            f"question_type must be one of {sorted(allowed_types)}"
        )

    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty must be between 1 and 5")

    if isinstance(answer_key, (dict, list)):
        answer_key = json.dumps(
            answer_key, ensure_ascii=False
        )
    elif not isinstance(answer_key, str):
        raise TypeError("answer_key must be a string, dict, or list")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO question_bank (
                document_id, concept_id, question, question_type,
                answer_key, source_reference, difficulty, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document_id,
            concept_id,
            question.strip(),
            question_type,
            answer_key,
            source_reference,
            difficulty,
            _timestamp(),
        ))

        return cursor.lastrowid if cursor.rowcount else None


def get_question(question_id: int) -> Optional[dict]:
    """Return one question, including its answer key."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM question_bank WHERE id = ?
        """, (question_id,)).fetchone()

        result = _row_to_dict(row)

        if result:
            try:
                result["answer_key"] = json.loads(result["answer_key"])
            except (json.JSONDecodeError, TypeError):
                pass

        return result


def get_questions_for_concept(
    concept_id: int,
    question_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Return active questions for a concept."""
    sql = """
        SELECT id, document_id, concept_id, question,
            question_type, source_reference, difficulty, created_at
        FROM question_bank
        WHERE concept_id = ? AND active = 1
    """
    params: list[Any] = [concept_id]

    if question_type is not None:
        sql += " AND question_type = ?"
        params.append(question_type)

    sql += " ORDER BY difficulty, id"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def get_question_for_evaluation(question_id: int) -> Optional[dict]:
    """
    Internal evaluation API. Use this to retrieve the answer key
    on the backend, not when displaying a question to the learner.
    """
    return get_question(question_id)


# ============================================================
# LEARNING EVENTS AND MASTERY
# ============================================================

def record_learning_event(
    learner_id: str,
    document_id: str,
    concept_id: int,
    question_id: Optional[int],
    is_correct: bool,
    attempts: int = 1,
    hints_used: int = 0,
    response_seconds: float = 0,
    submitted_answer: str = "",
) -> dict:
    """
    Record an answer attempt and update concept mastery.

    Initial mastery model:
    - Correct answers earn credit.
    - Multiple attempts reduce the credit for that event.
    - Hint use reduces the credit for that event.
    - Events are averaged into a running mastery score.

    This is a heuristic, not a trained ML model.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if hints_used < 0:
        raise ValueError("hints_used cannot be negative")
    if response_seconds < 0:
        raise ValueError("response_seconds cannot be negative")

    now_dt = _now()
    now_text = now_dt.isoformat(timespec="seconds")
    correct = bool(is_correct)

    event_credit = 1.0 if correct else 0.0
    event_credit /= attempts
    event_credit *= 0.8 ** hints_used
    event_score = event_credit * 100

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO learning_events (
                learner_id, document_id, concept_id, question_id,
                submitted_answer, is_correct, attempts, hints_used,
                response_seconds, answered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            learner_id,
            document_id,
            concept_id,
            question_id,
            submitted_answer,
            int(correct),
            attempts,
            hints_used,
            float(response_seconds),
            now_text,
        ))

        previous = conn.execute("""
            SELECT mastery_score, attempts, correct_answers
            FROM concept_mastery
            WHERE learner_id = ?
              AND document_id = ?
              AND concept_id = ?
        """, (
            learner_id, document_id, concept_id
        )).fetchone()

        old_attempts = previous["attempts"] if previous else 0
        old_score = previous["mastery_score"] if previous else 0.0
        old_correct = previous["correct_answers"] if previous else 0

        total_attempts = old_attempts + 1
        new_score = (
            old_score * old_attempts + event_score
        ) / total_attempts
        new_correct = old_correct + int(correct)

        if not correct:
            interval_days = REVIEW_INTERVALS["incorrect"]
        elif new_score < 50:
            interval_days = REVIEW_INTERVALS["low"]
        elif new_score < 75:
            interval_days = REVIEW_INTERVALS["developing"]
        elif new_score < 90:
            interval_days = REVIEW_INTERVALS["good"]
        else:
            interval_days = REVIEW_INTERVALS["mastered"]

        due_dt = now_dt + timedelta(days=interval_days)
        due_text = due_dt.isoformat(timespec="seconds")

        conn.execute("""
            INSERT INTO concept_mastery (
                learner_id, document_id, concept_id,
                mastery_score, attempts, correct_answers,
                last_tested_at, next_review_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id, document_id, concept_id)
            DO UPDATE SET
                mastery_score = excluded.mastery_score,
                attempts = excluded.attempts,
                correct_answers = excluded.correct_answers,
                last_tested_at = excluded.last_tested_at,
                next_review_at = excluded.next_review_at,
                updated_at = excluded.updated_at
        """, (
            learner_id,
            document_id,
            concept_id,
            round(new_score, 2),
            total_attempts,
            new_correct,
            now_text,
            due_text,
            now_text,
        ))

        conn.execute("""
            UPDATE review_queue
            SET status = 'cancelled'
            WHERE learner_id = ?
            AND document_id = ?
            AND concept_id = ?
            AND status = 'pending'
        """, (learner_id, document_id, concept_id))

        conn.execute("""
            INSERT INTO review_queue (
                learner_id, document_id, concept_id,
                due_at, status, created_at
            )
            VALUES (?, ?, ?, ?, 'pending', ?)
        """, (
            learner_id,
            document_id,
            concept_id,
            due_text,
            now_text,
        ))

    return {
        "learner_id": learner_id,
        "document_id": document_id,
        "concept_id": concept_id,
        "is_correct": correct,
        "event_score": round(event_score, 2),
        "mastery_score": round(new_score, 2),
        "attempts": total_attempts,
        "correct_answers": new_correct,
        "next_review_at": due_text,
        "review_interval_days": interval_days,
    }


def get_concept_performance(
    learner_id: str,
    concept_id: int,
) -> Optional[dict]:
    """Return mastery and event statistics for one concept."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                c.id AS concept_id,
                c.concept_name,
                c.source_pages,
                m.document_id,
                m.mastery_score,
                m.attempts,
                m.correct_answers,
                m.last_tested_at,
                m.next_review_at,
                m.updated_at
            FROM concept_mastery AS m
            JOIN concepts AS c ON c.id = m.concept_id
            WHERE m.learner_id = ? AND m.concept_id = ?
        """, (learner_id, concept_id)).fetchone()

        return _row_to_dict(row)


def get_document_performance(
    learner_id: str,
    document_id: str,
) -> list[dict]:
    """Return mastery records for all tested concepts in a document."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                c.id AS concept_id,
                c.concept_name,
                c.source_pages,
                COALESCE(m.mastery_score, 0) AS mastery_score,
                COALESCE(m.attempts, 0) AS attempts,
                COALESCE(m.correct_answers, 0) AS correct_answers,
                m.last_tested_at,
                m.next_review_at
            FROM concepts AS c
            LEFT JOIN concept_mastery AS m
            ON m.concept_id = c.id
            AND m.document_id = c.document_id
            AND m.learner_id = ?
        WHERE c.document_id = ?
            ORDER BY mastery_score ASC, c.concept_name
        """, (learner_id, document_id)).fetchall()

        return [dict(row) for row in rows]


# ============================================================
# REVISION QUEUE
# ============================================================

def get_due_reviews(
    learner_id: str = DEFAULT_LEARNER_ID,
    limit: int = 10,
    document_id: Optional[str] = None,
) -> list[dict]:
    """Return pending revisions that are due, earliest first."""
    sql = """
        SELECT
            r.id AS review_id,
            r.learner_id,
            r.document_id,
            d.document_name,
            r.concept_id,
            c.concept_name,
            c.source_pages,
            r.due_at,
            COALESCE(m.mastery_score, 0) AS mastery_score,
            r.created_at
        FROM review_queue AS r
        JOIN concepts AS c ON c.id = r.concept_id
        JOIN learning_documents AS d
        ON d.document_id = r.document_id
        LEFT JOIN concept_mastery AS m
        ON m.learner_id = r.learner_id
        AND m.document_id = r.document_id
    AND m.concept_id = r.concept_id
        WHERE r.learner_id = ?
        AND r.status = 'pending'
        AND r.due_at <= ?
    """
    params: list[Any] = [learner_id, _timestamp()]

    if document_id is not None:
        sql += " AND r.document_id = ?"
        params.append(document_id)

    sql += " ORDER BY mastery_score ASC, r.due_at ASC LIMIT ?"
    params.append(max(0, int(limit)))

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def complete_review(review_id: int, learner_id: str) -> bool:
    """Mark a pending revision as completed for its learner."""
    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE review_queue
            SET status = 'completed', completed_at = ?
            WHERE id = ?
            AND learner_id = ?
            AND status = 'pending'
        """, (_timestamp(), review_id, learner_id))

        return cursor.rowcount > 0


def get_learning_event_history(
    learner_id: str = DEFAULT_LEARNER_ID,
    concept_id: Optional[int] = None,
    limit: int = 50,
) -> list[dict]:
    """Return recent learning events without exposing answer keys."""
    sql = """
        SELECT
            e.id,
            e.learner_id,
            e.document_id,
            d.document_name,
            e.concept_id,
            c.concept_name,
            e.question_id,
            e.submitted_answer,
            e.is_correct,
            e.attempts,
            e.hints_used,
            e.response_seconds,
            e.answered_at
        FROM learning_events AS e
        JOIN concepts AS c ON c.id = e.concept_id
        JOIN learning_documents AS d
        ON d.document_id = e.document_id
        WHERE e.learner_id = ?
    """
    params: list[Any] = [learner_id]

    if concept_id is not None:
        sql += " AND e.concept_id = ?"
        params.append(concept_id)

    sql += " ORDER BY e.answered_at DESC LIMIT ?"
    params.append(max(0, int(limit)))

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]



# ============================================================
# POMODORO — PERSISTENT FOCUS SESSIONS
# ============================================================

def save_pomodoro_session(
    planned_minutes: int,
    completed_minutes: int,
    session_type: str = "focus",
    learner_id: str = DEFAULT_LEARNER_ID,
) -> int:
    """Save a completed focus or break session."""
    if session_type not in {"focus", "short_break", "long_break"}:
        raise ValueError("Invalid Pomodoro session type.")
    if planned_minutes <= 0:
        raise ValueError("planned_minutes must be positive.")
    if not 0 <= completed_minutes <= planned_minutes:
        raise ValueError("Invalid completed duration.")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO pomodoro_sessions (
                learner_id, session_type, planned_minutes,
                completed_minutes, completed_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            learner_id,
            session_type,
            planned_minutes,
            completed_minutes,
            _timestamp(),
        ))
        return cursor.lastrowid


def get_pomodoro_stats(
    learner_id: str = DEFAULT_LEARNER_ID,
) -> dict:
    """Return persistent Pomodoro totals."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_sessions,
                COALESCE(SUM(
                    CASE WHEN session_type = 'focus'
                    THEN 1 ELSE 0 END
                ), 0) AS completed_focus_sessions,
                COALESCE(SUM(
                    CASE WHEN session_type = 'focus'
                    THEN completed_minutes ELSE 0 END
                ), 0) AS focus_minutes,
                COALESCE(SUM(
                    CASE WHEN session_type != 'focus'
                    THEN 1 ELSE 0 END
                ), 0) AS completed_breaks
            FROM pomodoro_sessions
            WHERE learner_id = ?
        """, (learner_id,)).fetchone()

        return dict(row)


def get_recent_pomodoro_sessions(
    learner_id: str = DEFAULT_LEARNER_ID,
    limit: int = 10,
) -> list[dict]:
    """Return recent completed sessions."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM pomodoro_sessions
            WHERE learner_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (learner_id, max(0, int(limit)))).fetchall()

        return [dict(row) for row in rows]


# ============================================================
# TO-DO LIST — PERSISTENT TASKS
# ============================================================

def add_todo_task(
    title: str,
    category: str = "General",
    priority: str = "Medium",
    learner_id: str = DEFAULT_LEARNER_ID,
) -> int:
    """Create a task and return its ID."""
    title = title.strip()

    if not title:
        raise ValueError("Task title cannot be empty.")
    if priority not in {"Low", "Medium", "High"}:
        raise ValueError("Invalid task priority.")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO todo_tasks (
                learner_id, title, category, priority,
                is_completed, created_at
            )
            VALUES (?, ?, ?, ?, 0, ?)
        """, (
            learner_id, title, category.strip() or "General",
            priority, _timestamp(),
        ))
        return cursor.lastrowid


def get_todo_tasks(
    learner_id: str = DEFAULT_LEARNER_ID,
    include_completed: bool = True,
) -> list[dict]:
    """Return tasks, pending first."""
    sql = """
        SELECT *
        FROM todo_tasks
        WHERE learner_id = ?
    """
    params: list[Any] = [learner_id]

    if not include_completed:
        sql += " AND is_completed = 0"

    sql += """
        ORDER BY is_completed ASC,
        CASE priority
            WHEN 'High' THEN 1
            WHEN 'Medium' THEN 2
            ELSE 3
        END,
        id DESC
    """

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def set_todo_completed(
    task_id: int,
    completed: bool,
    learner_id: str = DEFAULT_LEARNER_ID,
) -> bool:
    """Mark a task complete or reopen it."""
    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE todo_tasks
            SET is_completed = ?,
                completed_at = ?
            WHERE id = ? AND learner_id = ?
        """, (
            int(completed),
            _timestamp() if completed else None,
            task_id,
            learner_id,
        ))
        return cursor.rowcount > 0


def delete_todo_task(
    task_id: int,
    learner_id: str = DEFAULT_LEARNER_ID,
) -> bool:
    """Delete one task belonging to this learner."""
    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM todo_tasks
            WHERE id = ? AND learner_id = ?
        """, (task_id, learner_id))
        return cursor.rowcount > 0
