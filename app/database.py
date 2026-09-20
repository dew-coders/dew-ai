"""SQLite persistence layer — every conversation, message, feedback event,
search log and training run is stored here so the model can keep learning.

Uses one connection per operation plus a global lock; WAL mode keeps concurrent
reads by the API responsive while the trainer writes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Dict, List, Optional

from app import config

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,          -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    source          TEXT,                   -- neural | search | template | fallback
    confidence      REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    rating      INTEGER NOT NULL,           -- +1 / -1
    comment     TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS training_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt        TEXT NOT NULL,
    response      TEXT NOT NULL,
    source        TEXT,
    quality       INTEGER DEFAULT 0,        -- -1 (downvoted) .. +1 (upvoted)
    used_in_runs  INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS search_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    results_json TEXT,
    used         INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS training_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger        TEXT,                    -- bootstrap | auto | manual
    steps          INTEGER,
    samples        INTEGER,
    max_sample_id  INTEGER,
    loss_before    REAL,
    loss_after     REAL,
    status         TEXT DEFAULT 'running',  -- running | done | error
    error          TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    finished_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_samples_created ON training_samples(created_at);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _LOCK, _connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Conversations & messages
# --------------------------------------------------------------------------- #
def create_conversation(title: str = "") -> int:
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title[:80],))
        return int(cur.lastrowid)


def add_message(conversation_id: int, role: str, content: str,
                source: Optional[str] = None,
                confidence: Optional[float] = None) -> int:
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, source, confidence)"
            " VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, source, confidence))
        return int(cur.lastrowid)


def get_history(conversation_id: int, limit: int = 200) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id LIMIT ?",
            (conversation_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_recent_messages(conversation_id: int, n: int = 6) -> List[Dict]:
    rows = get_history(conversation_id, limit=n)
    return rows


def list_conversations(limit: int = 50) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at,"
            " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count"
            " FROM conversations c ORDER BY c.id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Training samples & feedback
# --------------------------------------------------------------------------- #
def add_training_sample(prompt: str, response: str, source: str) -> int:
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO training_samples (prompt, response, source) VALUES (?, ?, ?)",
            (prompt, response, source))
        return int(cur.lastrowid)


def get_training_samples(exclude_negative: bool = True) -> List[Dict]:
    query = "SELECT * FROM training_samples"
    if exclude_negative:
        query += " WHERE quality >= 0"
    query += " ORDER BY id"
    with _connect() as conn:
        rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def add_feedback(message_id: int, rating: int, comment: str = "") -> bool:
    """Store feedback and propagate the rating to matching training samples."""
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT content FROM messages WHERE id = ? AND role = 'assistant'",
            (message_id,)).fetchone()
        if row is None:
            return False
        conn.execute("INSERT INTO feedback (message_id, rating, comment) VALUES (?, ?, ?)",
                     (message_id, rating, comment))
        conn.execute(
            "UPDATE training_samples SET quality = ? WHERE response = ?",
            (rating, row["content"]))
    return True


def add_search_log(query: str, results: List[Dict]) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO search_logs (query, results_json, used) VALUES (?, ?, 1)",
            (query, json.dumps(results)[:8000]))


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #
def start_training_run(trigger: str, steps: int) -> int:
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO training_runs (trigger, steps, status) VALUES (?, ?, 'running')",
            (trigger, steps))
        return int(cur.lastrowid)


def finish_training_run(run_id: int, status: str, samples: int, max_sample_id: int,
                        loss_before: float, loss_after: float,
                        error: str = "") -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE training_runs SET status = ?, samples = ?, max_sample_id = ?,"
            " loss_before = ?, loss_after = ?, error = ?, finished_at = datetime('now')"
            " WHERE id = ?",
            (status, samples, max_sample_id, loss_before, loss_after, error, run_id))


def last_completed_run_max_sample_id() -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(max_sample_id), 0) AS m FROM training_runs"
            " WHERE status = 'done'").fetchone()
        return int(row["m"])


def count_new_samples() -> int:
    """Samples collected since the last completed training run."""
    cutoff = last_completed_run_max_sample_id()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM training_samples WHERE id > ? AND quality >= 0",
            (cutoff,)).fetchone()
        return int(row["c"])


def mark_samples_used(max_sample_id: int) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE training_samples SET used_in_runs = used_in_runs + 1 WHERE id <= ?",
            (max_sample_id,))


def list_training_runs(limit: int = 10) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def stats() -> Dict:
    with _connect() as conn:
        def one(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])
        return {
            "conversations": one("SELECT COUNT(*) FROM conversations"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "training_samples": one("SELECT COUNT(*) FROM training_samples"),
            "new_samples_pending": count_new_samples(),
            "searches": one("SELECT COUNT(*) FROM search_logs"),
            "feedback_up": one("SELECT COUNT(*) FROM feedback WHERE rating = 1"),
            "feedback_down": one("SELECT COUNT(*) FROM feedback WHERE rating = -1"),
            "training_runs": one("SELECT COUNT(*) FROM training_runs"),
        }
