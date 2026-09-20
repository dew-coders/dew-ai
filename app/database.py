"""Persistence layer — fast local SQLite + cloud-ready schema.

Every conversation, message, feedback event, search log and training run is
stored locally first (WAL mode, one lock, tiny inserts) so chat saving is
instant. Every row gets a server-generated UUID that is shared with Supabase,
and each write enqueues an entry in the `sync_outbox` table; a background
worker (app/sync.py) pushes queued rows to the cloud and pulls snapshots for
the chat-recovery system.

Legacy chat.db files (integer ids, no users) are migrated automatically so
old chats survive the upgrade.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from app import config

_LOCK = threading.Lock()

SESSION_TTL_DAYS = 30


def new_id() -> str:
    return uuid.uuid4().hex


def to_cloud_uuid(local_id: str) -> str:
    """Local hex id → canonical dashed uuid (as Supabase stores/returns it)."""
    return str(uuid.UUID(local_id))


def to_local_id(any_uuid: str) -> str:
    """Any uuid form (dashed or hex) → local hex id."""
    return uuid.UUID(str(any_uuid)).hex


def _now() -> str:
    # Microsecond precision: messages created in the same second must still
    # sort in conversation order (user msg before the assistant reply).
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _norm_ts(value: Optional[str]) -> str:
    """Normalize an ISO timestamp (from Supabase) to our sortable format."""
    if not value:
        return _now()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return str(value)


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE COLLATE NOCASE,
    lang        TEXT NOT NULL DEFAULT 'en',
    created_at  TEXT NOT NULL,
    last_seen   TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT REFERENCES users(id),
    title       TEXT,
    lang        TEXT DEFAULT 'en',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,          -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    source          TEXT,                   -- neural | search | template | fallback
    confidence      REAL,
    lang            TEXT DEFAULT 'en',
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  TEXT NOT NULL REFERENCES messages(id),
    rating      INTEGER NOT NULL,           -- +1 / -1
    comment     TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_samples (
    id            TEXT PRIMARY KEY,
    prompt        TEXT NOT NULL,
    response      TEXT NOT NULL,
    source        TEXT,
    quality       INTEGER DEFAULT 0,        -- -1 (downvoted) .. +1 (upvoted)
    lang          TEXT DEFAULT 'en',
    used_in_runs  INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    results_json TEXT,
    used         INTEGER DEFAULT 0,
    created_at   TEXT NOT NULL
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
    created_at     TEXT NOT NULL,
    finished_at    TEXT
);
CREATE TABLE IF NOT EXISTS sync_outbox (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    tbl        TEXT NOT NULL,               -- users | conversations | messages | training_samples
    row_id     TEXT NOT NULL,
    attempts   INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv   ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_samples_created ON training_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_outbox_seq      ON sync_outbox(seq);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# --------------------------------------------------------------------------- #
# Legacy migration (pre-users schema used INTEGER ids)
# --------------------------------------------------------------------------- #
def _is_legacy(conn: sqlite3.Connection) -> bool:
    cols = conn.execute("PRAGMA table_info(conversations)").fetchall()
    return bool(cols) and cols[0]["name"] == "id" and "INTEGER" in cols[0]["type"].upper()


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    print("[db] migrating legacy chat.db (integer ids) → uuid schema…", flush=True)
    conn.execute("ALTER TABLE conversations RENAME TO conversations_old")
    conn.execute("ALTER TABLE messages RENAME TO messages_old")
    conn.execute("ALTER TABLE training_samples RENAME TO samples_old")
    conn.execute("ALTER TABLE feedback RENAME TO feedback_old")
    conn.executescript(SCHEMA)

    conv_map: Dict[int, str] = {}
    for row in conn.execute("SELECT * FROM conversations_old ORDER BY id"):
        cid = new_id()
        conv_map[row["id"]] = cid
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, lang, created_at)"
            " VALUES (?, NULL, ?, 'en', ?)", (cid, row["title"], row["created_at"]))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('conversations', ?, ?)",
                     (cid, _now()))

    msg_map: Dict[int, str] = {}
    for row in conn.execute("SELECT * FROM messages_old ORDER BY id"):
        mid = new_id()
        msg_map[row["id"]] = mid
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, source,"
            " confidence, lang, created_at) VALUES (?, ?, ?, ?, ?, ?, 'en', ?)",
            (mid, conv_map.get(row["conversation_id"], new_id()), row["role"],
             row["content"], row["source"], row["confidence"], row["created_at"]))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('messages', ?, ?)",
                     (mid, _now()))

    for row in conn.execute("SELECT * FROM samples_old ORDER BY id"):
        sid = new_id()
        conn.execute(
            "INSERT INTO training_samples (id, prompt, response, source, quality,"
            " lang, used_in_runs, created_at) VALUES (?, ?, ?, ?, ?, 'en', ?, ?)",
            (sid, row["prompt"], row["response"], row["source"],
             row["quality"], row["used_in_runs"], row["created_at"]))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('training_samples', ?, ?)",
                     (sid, _now()))

    for row in conn.execute("SELECT * FROM feedback_old"):
        conn.execute(
            "INSERT INTO feedback (message_id, rating, comment, created_at)"
            " VALUES (?, ?, ?, ?)",
            (msg_map.get(row["message_id"], new_id()), row["rating"],
             row["comment"], row["created_at"]))

    # children first — dropping a parent with live FK children raises IntegrityError
    for tbl in ("feedback_old", "messages_old", "samples_old", "conversations_old"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    print("[db] legacy migration complete — old chats preserved", flush=True)


def init_db() -> None:
    with _LOCK, _connect() as conn:
        if _is_legacy(conn):
            _migrate_legacy(conn)
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Users & sessions (username-only login)
# --------------------------------------------------------------------------- #
def create_user(username: str, lang: str = "en") -> Dict:
    uid = new_id()
    now = _now()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, username, lang, created_at, last_seen) VALUES (?, ?, ?, ?, ?)",
            (uid, username, lang, now, now))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('users', ?, ?)",
                     (uid, now))
    return {"id": uid, "username": username, "lang": lang, "created_at": now}


def get_user_by_username(username: str) -> Optional[Dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                           (username,)).fetchone()
    return dict(row) if row else None


def get_user(user_id: str) -> Optional[Dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def touch_user(user_id: str) -> None:
    with _LOCK, _connect() as conn:
        conn.execute("UPDATE users SET last_seen = ? WHERE id = ?", (_now(), user_id))


def update_user_lang(user_id: str, lang: str) -> None:
    with _LOCK, _connect() as conn:
        conn.execute("UPDATE users SET lang = ? WHERE id = ?", (lang, user_id))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('users', ?, ?)",
                     (user_id, _now()))


def login_or_register(username: str, lang: str = "en") -> Tuple[Dict, str, bool]:
    """Username-only login: returns (user, session_token, created)."""
    user = get_user_by_username(username)
    created = False
    if user is None:
        user = create_user(username, lang)
        created = True
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)
               ).strftime("%Y-%m-%d %H:%M:%S")
    with _LOCK, _connect() as conn:
        conn.execute("INSERT INTO sessions (token, user_id, created_at, expires_at)"
                     " VALUES (?, ?, ?, ?)", (token, user["id"], _now(), expires))
    touch_user(user["id"])
    return user, token, created


def get_session_user(token: str) -> Optional[Dict]:
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token = ? AND s.expires_at > ?",
            (token, _now())).fetchone()
    return dict(row) if row else None


def logout(token: str) -> None:
    with _LOCK, _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# --------------------------------------------------------------------------- #
# Conversations & messages
# --------------------------------------------------------------------------- #
def create_conversation(user_id: Optional[str], title: str = "", lang: str = "en") -> str:
    cid = new_id()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, lang, created_at)"
            " VALUES (?, ?, ?, ?, ?)", (cid, user_id, title[:80], lang, _now()))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('conversations', ?, ?)",
                     (cid, _now()))
    return cid


def get_conversation(conversation_id: str) -> Optional[Dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?",
                           (conversation_id,)).fetchone()
    return dict(row) if row else None


def add_message(conversation_id: str, role: str, content: str,
                source: Optional[str] = None, confidence: Optional[float] = None,
                lang: str = "en") -> str:
    mid = new_id()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, source,"
            " confidence, lang, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, content, source, confidence, lang, _now()))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('messages', ?, ?)",
                     (mid, _now()))
    return mid


def get_history(conversation_id: str, limit: int = 200) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, id LIMIT ?",
            (conversation_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_recent_messages(conversation_id: str, n: int = 6) -> List[Dict]:
    """Last n messages in chronological order (for prompt building)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ?"
            " ORDER BY created_at DESC, id DESC LIMIT ?) sub"
            " ORDER BY created_at ASC, id ASC",
            (conversation_id, n)).fetchall()
    return [dict(r) for r in rows]


def list_conversations(user_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    with _connect() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT c.id, c.user_id, c.title, c.lang, c.created_at,"
                " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count"
                " FROM conversations c WHERE c.user_id = ?"
                " ORDER BY c.created_at DESC LIMIT ?", (user_id, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT c.id, c.user_id, c.title, c.lang, c.created_at,"
                " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count"
                " FROM conversations c ORDER BY c.created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Training samples & feedback
# --------------------------------------------------------------------------- #
def add_training_sample(prompt: str, response: str, source: str,
                        lang: str = "en") -> str:
    sid = new_id()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO training_samples (id, prompt, response, source, lang, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)", (sid, prompt, response, source, lang, _now()))
        conn.execute("INSERT INTO sync_outbox (tbl, row_id, created_at) VALUES ('training_samples', ?, ?)",
                     (sid, _now()))
    return sid


def get_training_samples(exclude_negative: bool = True) -> List[Dict]:
    query = "SELECT * FROM training_samples"
    if exclude_negative:
        query += " WHERE quality >= 0"
    query += " ORDER BY created_at, id"
    with _connect() as conn:
        rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def add_feedback(message_id: str, rating: int, comment: str = "") -> bool:
    """Store feedback and propagate the rating to matching training samples."""
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT content FROM messages WHERE id = ? AND role = 'assistant'",
            (message_id,)).fetchone()
        if row is None:
            return False
        conn.execute("INSERT INTO feedback (message_id, rating, comment, created_at)"
                     " VALUES (?, ?, ?, ?)", (message_id, rating, comment, _now()))
        conn.execute(
            "UPDATE training_samples SET quality = ? WHERE response = ?",
            (rating, row["content"]))
    return True


def add_search_log(query: str, results: List[Dict]) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO search_logs (query, results_json, used, created_at) VALUES (?, ?, 1, ?)",
            (query, json.dumps(results)[:8000], _now()))


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #
def start_training_run(trigger: str, steps: int) -> int:
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO training_runs (trigger, steps, status, created_at)"
            " VALUES (?, ?, 'running', ?)", (trigger, steps, _now()))
        return int(cur.lastrowid)


def finish_training_run(run_id: int, status: str, samples: int, max_sample_id: int,
                        loss_before: float, loss_after: float,
                        error: str = "") -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE training_runs SET status = ?, samples = ?, max_sample_id = ?,"
            " loss_before = ?, loss_after = ?, error = ?, finished_at = ?"
            " WHERE id = ?",
            (status, samples, max_sample_id, loss_before, loss_after, error,
             _now(), run_id))


def last_completed_run_cutoff() -> str:
    """created_at of the newest sample consumed by the last completed run."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT MAX(created_at) AS m FROM training_samples"
            " WHERE used_in_runs > 0").fetchone()
        return row["m"] or ""


def count_new_samples() -> int:
    """Samples collected since the last completed training run."""
    cutoff = last_completed_run_cutoff()
    with _connect() as conn:
        if cutoff:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM training_samples"
                " WHERE created_at > ? AND quality >= 0", (cutoff,)).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM training_samples WHERE quality >= 0").fetchone()
        return int(row["c"])


def mark_samples_used(cutoff_created_at: str) -> None:
    """Mark samples created at or before the cutoff as consumed by a training run."""
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE training_samples SET used_in_runs = used_in_runs + 1"
            " WHERE created_at <= ?", (cutoff_created_at,))


def list_training_runs(limit: int = 10) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Sync outbox (fast local queue consumed by app/sync.py)
# --------------------------------------------------------------------------- #
def outbox_batch(limit: int = 50) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT seq, tbl, row_id, attempts FROM sync_outbox"
            " ORDER BY seq LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def outbox_pending() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM sync_outbox").fetchone()[0])


def outbox_done(seq: int) -> None:
    with _LOCK, _connect() as conn:
        conn.execute("DELETE FROM sync_outbox WHERE seq = ?", (seq,))


def outbox_retry(seq: int) -> int:
    """Bump the attempt counter; returns the new attempt count."""
    with _LOCK, _connect() as conn:
        conn.execute("UPDATE sync_outbox SET attempts = attempts + 1 WHERE seq = ?", (seq,))
        row = conn.execute("SELECT attempts FROM sync_outbox WHERE seq = ?", (seq,)).fetchone()
    return int(row["attempts"]) if row else 0


def outbox_drop_poisoned(max_attempts: int) -> int:
    """Give up on entries that failed too many times (e.g. schema not applied)."""
    with _LOCK, _connect() as conn:
        cur = conn.execute("DELETE FROM sync_outbox WHERE attempts >= ?", (max_attempts,))
        return cur.rowcount


def row_for_cloud(tbl: str, row_id: str) -> Optional[Dict]:
    """Fetch a row as the dict expected by the matching Supabase table.
    Ids are converted to canonical dashed uuids (Postgres normalizes uuid
    columns, so pushes and later cloud reads stay consistent)."""
    def cid(value: Optional[str]) -> Optional[str]:
        return to_cloud_uuid(value) if value else None
    with _connect() as conn:
        if tbl == "users":
            row = conn.execute(
                "SELECT id, username, lang, created_at, last_seen FROM users"
                " WHERE id = ?", (row_id,)).fetchone()
        elif tbl == "conversations":
            row = conn.execute(
                "SELECT id, user_id, title, lang, created_at FROM conversations"
                " WHERE id = ?", (row_id,)).fetchone()
        elif tbl == "messages":
            row = conn.execute(
                "SELECT id, conversation_id, role, content, source, confidence,"
                " lang, created_at FROM messages WHERE id = ?", (row_id,)).fetchone()
        elif tbl == "training_samples":
            row = conn.execute(
                "SELECT id, prompt, response, source, quality, lang, created_at"
                " FROM training_samples WHERE id = ?", (row_id,)).fetchone()
        else:
            return None
    if not row:
        return None
    out = dict(row)
    for key in ("id", "user_id", "conversation_id"):
        if key in out:
            out[key] = cid(out[key])
    return out


# --------------------------------------------------------------------------- #
# Cloud → local (recovery): direct inserts, NOT re-enqueued to the outbox
# (otherwise restored rows would ping-pong back to the cloud forever).
# --------------------------------------------------------------------------- #
def create_conversation_from_cloud(conv: Dict) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, user_id, title, lang, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (to_local_id(conv["id"]),
             to_local_id(conv["user_id"]) if conv.get("user_id") else None,
             conv.get("title") or "", conv.get("lang") or "en",
             _norm_ts(conv.get("created_at"))))


def add_message_from_cloud(conversation_id: str, m: Dict) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO messages (id, conversation_id, role, content,"
            " source, confidence, lang, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (to_local_id(m["id"]), conversation_id, m.get("role") or "assistant",
             m.get("content") or "", m.get("source"), m.get("confidence"),
             m.get("lang") or "en", _norm_ts(m.get("created_at"))))


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def stats() -> Dict:
    with _connect() as conn:
        def one(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])
        return {
            "users": one("SELECT COUNT(*) FROM users"),
            "conversations": one("SELECT COUNT(*) FROM conversations"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "training_samples": one("SELECT COUNT(*) FROM training_samples"),
            "new_samples_pending": count_new_samples(),
            "searches": one("SELECT COUNT(*) FROM search_logs"),
            "feedback_up": one("SELECT COUNT(*) FROM feedback WHERE rating = 1"),
            "feedback_down": one("SELECT COUNT(*) FROM feedback WHERE rating = -1"),
            "training_runs": one("SELECT COUNT(*) FROM training_runs"),
            "cloud_sync_pending": outbox_pending(),
        }
