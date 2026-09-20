"""Cloud sync engine — fast local saves, resilient Supabase pushes, recovery.

- Writes stay local: every DB write enqueues a row in `sync_outbox` (same
  transaction), so chat saving never waits on the network.
- A background worker drains the outbox to Supabase with idempotent upserts
  (row ids are shared UUIDs), retrying transient failures and dropping
  poisoned rows only after many attempts.
- `recover_from_cloud(user)` pulls the user's conversations + messages back
  into local SQLite — this powers the chat-recovery system (e.g. after
  reinstalling, or on a fresh machine).
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from app import config, database as db
from app.supabase_client import SupabaseError, client as supabase

_LOCK = threading.Lock()
_state: Dict[str, object] = {
    "enabled": False,
    "running": False,
    "pushed_total": 0,
    "failed_total": 0,
    "last_error": "",
    "last_push_at": "",
    "last_recovery": "",
}


def start() -> None:
    """Boot the background sync worker (no-op if disabled or already running)."""
    with _LOCK:
        if _state["running"]:
            return
        _state["enabled"] = bool(config.SYNC_ENABLED and supabase.available)
        if not _state["enabled"]:
            if not config.SYNC_ENABLED:
                print("[sync] disabled via SYNC_ENABLED=0", flush=True)
            else:
                print("[sync] SUPABASE_URL/KEY missing — cloud sync disabled", flush=True)
            return
        _state["running"] = True
    threading.Thread(target=_worker, name="cloud-sync", daemon=True).start()


def status() -> Dict:
    with _LOCK:
        snapshot = dict(_state)
    snapshot["pending"] = _safe_pending()
    snapshot["supabase_url"] = config.SUPABASE_URL
    return snapshot


def _safe_pending() -> int:
    try:
        return db.outbox_pending()
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Background push loop
# --------------------------------------------------------------------------- #
def _worker() -> None:
    print(f"[sync] worker started → {config.SUPABASE_URL}", flush=True)
    idle_rounds = 0
    while True:
        batch = []
        try:
            batch = db.outbox_batch(config.SYNC_BATCH)
        except Exception as exc:  # sqlite hiccup — brief pause
            _set_error(f"outbox read failed: {exc}")
            time.sleep(2)
            continue
        if not batch:
            idle_rounds += 1
            time.sleep(min(5.0, 0.2 * idle_rounds))
            continue
        idle_rounds = 0
        for entry in batch:
            _push_one(entry)


def _push_one(entry: Dict) -> None:
    seq, tbl, row_id = entry["seq"], entry["tbl"], entry["row_id"]
    try:
        row = db.row_for_cloud(tbl, row_id)
        if row is None:            # row vanished (e.g. legacy migration artifact)
            db.outbox_done(seq)
            return
        ok, err = supabase.upsert(tbl, [row], conflict="id")
        if ok:
            db.outbox_done(seq)
            with _LOCK:
                _state["pushed_total"] = int(_state["pushed_total"]) + 1
                _state["last_error"] = ""
                _state["last_push_at"] = time.strftime("%H:%M:%S")
        else:
            _record_failure(seq, err)
    except SupabaseError as exc:
        _record_failure(seq, str(exc))
    except Exception as exc:  # network down, etc. — retry later
        _record_failure(seq, f"{type(exc).__name__}: {exc}")


def _record_failure(seq: int, error: str) -> None:
    attempts = db.outbox_retry(seq)
    with _LOCK:
        _state["failed_total"] = int(_state["failed_total"]) + 1
        _state["last_error"] = error[:300]
    if attempts >= config.SYNC_MAX_ATTEMPTS:
        dropped = db.outbox_drop_poisoned(config.SYNC_MAX_ATTEMPTS)
        if dropped:
            print(f"[sync] dropped {dropped} poisoned outbox row(s) after "
                  f"{config.SYNC_MAX_ATTEMPTS} attempts: {error[:120]}", flush=True)


def _set_error(msg: str) -> None:
    with _LOCK:
        _state["last_error"] = msg[:300]


# --------------------------------------------------------------------------- #
# Recovery (chat history restore from the cloud)
# --------------------------------------------------------------------------- #
def recover_from_cloud(user: Dict) -> Dict:
    """Pull the user's conversations and messages from Supabase into SQLite.
    Only inserts rows that don't exist locally, so it is safe to run anytime."""
    result: Dict[str, object] = {"enabled": bool(_state["enabled"]),
                                 "conversations": 0, "messages": 0}
    if not _state["enabled"]:
        result["error"] = "cloud sync disabled"
        return result

    try:
        convs = supabase.select(
            "conversations",
            filters={"user_id": f"eq.{db.to_cloud_uuid(user['id'])}"},
            order="created_at.desc", limit=200)
        for conv in convs:
            cid = db.to_local_id(conv["id"])   # local hex form of the cloud uuid
            existing = db.get_conversation(cid)
            if existing is None:
                db.create_conversation_from_cloud({**conv, "id": cid})
            known = {m["id"] for m in db.get_history(cid, limit=1000)}
            msgs = supabase.select("messages",
                                   filters={"conversation_id": f"eq.{conv['id']}"},
                                   order="created_at.asc", limit=500)
            fresh = [m for m in msgs if db.to_local_id(m["id"]) not in known]
            for m in fresh:
                db.add_message_from_cloud(cid, {**m, "id": db.to_local_id(m["id"])})
            if fresh:
                result["conversations"] = int(result["conversations"]) + 1
                result["messages"] = int(result["messages"]) + len(fresh)
            elif existing is None:
                result["conversations"] = int(result["conversations"]) + 1
        with _LOCK:
            _state["last_recovery"] = time.strftime("%H:%M:%S")
        return result
    except SupabaseError as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
