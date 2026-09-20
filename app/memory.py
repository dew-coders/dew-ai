"""Long-term user memory — Dew AI remembers facts about each user across
conversations (ChatGPT-style "memory" feature).

- `extract_facts()`  : pattern-based fact detection ("my name is X", "I live in
  Y", "I like Z", "remember that …") in English + Sinhala.
- `remember()`       : store a fact in SQLite (auto-synced to Supabase).
- `recall()`         : retrieve the user's stored facts, optionally filtered.
- Pipeline hook      : every user message is scanned; found facts are stored
  and recalled facts are injected into the model prompt.

Facts are per-user, timestamped and never shared between users.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# fact patterns (English + Sinhala, written + spoken)
# --------------------------------------------------------------------------- #
_FACT_PATTERNS = [
    # name
    (re.compile(r"\bmy name is\s+(?P<value>[\w .\-]{2,40})", re.I), "name"),
    (re.compile(r"\bi(?:'m| am)\s+(?P<value>[A-Z][\w\-]{1,20})\b(?!\s*(?:fine|good|ok|okay|happy|sad|tired))"), "name"),
    (re.compile(r"මගේ\s*නම\s*([\w ]{2,40})", re.I), "name"),
    (re.compile(r"මම\s+([\w ]{2,30})\s*(?:නම්|කියලා)", re.I), "name"),
    # location
    (re.compile(r"\bi\s+(?:live|am)\s+in\s+(?P<value>[\w .\-]{2,40})", re.I), "location"),
    (re.compile(r"මම\s*(?:හිටින්නේ|ඉන්නේ|වාසය\s*කරන්නේ)\s*([\w ]{2,40})", re.I), "location"),
    # likes / dislikes
    (re.compile(r"\bi\s+(?:really\s+)?(?:like|love|enjoy)\s+(?P<value>[\w .\-]{2,60})", re.I), "likes"),
    (re.compile(r"මට\s*([\w ]{2,40})\s*කැමති", re.I), "likes"),
    (re.compile(r"\bi\s+(?:hate|dislike|can't stand)\s+(?P<value>[\w .\-]{2,60})", re.I), "dislikes"),
    (re.compile(r"මට\s*([\w ]{2,40})\s*කැමති\s*නෑ", re.I), "dislikes"),
    # job / study
    (re.compile(r"\bi\s+(?:work\s+(?:as|at)|am\s+studying)\s+(?P<value>[\w .\-]{2,60})", re.I), "work"),
    (re.compile(r"මම\s*(?:වැඩ\s*කරන්නේ|ඉගෙන\s*ගන්නේ)\s*([\w ]{2,40})", re.I), "work"),
    # arbitrary "remember that X"
    (re.compile(r"\bremember(?:\s+that)?\s+(?P<value>.{4,200})", re.I), "note"),
    (re.compile(r"මතක\s*තියාගන්න\s*(.{3,200})", re.I), "note"),
]

# facts that should override older ones of the same kind
_SINGLETON_KINDS = {"name", "location", "work"}


_VALUE_CUT = re.compile(
    r"\s+(?:and|but|who|which|that|also|so|i\s+am|i\s+live|i\s+like)\s+|[,.;!?\u0d94\u0d95]",
    re.I)


def _clean_value(raw: str) -> str:
    """Trim a captured fact at the first conjunction/punctuation boundary so
    'my name is Hansa and I live in X' yields just 'Hansa'."""
    value = _VALUE_CUT.split(raw.strip())[0].strip(" .,!?.")
    return value


def extract_facts(text: str) -> List[Tuple[str, str]]:
    """Scan a user message for self-describing facts. Returns [(kind, value)]."""
    facts = []
    for pattern, kind in _FACT_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group("value") if "value" in m.groupdict() else m.group(1)
            value = _clean_value(raw)
            if value and len(value) >= 2:
                facts.append((kind, value))
    return facts[:3]  # cap per message


def normalize_value(kind: str, value: str) -> str:
    value = value.strip()
    if kind == "name":
        return value.title()
    return value


# --------------------------------------------------------------------------- #
# DB-backed storage (via app.database)
# --------------------------------------------------------------------------- #
def remember(user_id: str, kind: str, value: str, source: str = "user") -> bool:
    """Store a fact for a user. Returns True if newly stored."""
    from app import database as db
    return db.add_user_memory(user_id, kind, normalize_value(kind, value), source)


def recall(user_id: str, kind: Optional[str] = None,
           limit: int = 20) -> List[Dict]:
    """Fetch stored facts for a user (optionally one kind)."""
    from app import database as db
    return db.get_user_memory(user_id, kind=kind, limit=limit)


def recall_text(user_id: str) -> str:
    """One-line-per-fact text block for prompt injection (empty if none)."""
    facts = recall(user_id, limit=12)
    if not facts:
        return ""
    labels = {"name": "Name", "location": "Lives in", "likes": "Likes",
              "dislikes": "Dislikes", "work": "Work", "note": "Note"}
    lines = [f"{labels.get(f['kind'], f['kind'])}: {f['value']}" for f in facts]
    return "[What I remember about you]\n" + "\n".join(lines)


def process_message(user_id: str, text: str) -> List[Dict]:
    """Extract + store facts from a user message. Returns stored facts."""
    stored = []
    for kind, value in extract_facts(text):
        if remember(user_id, kind, value, source="message"):
            stored.append({"kind": kind, "value": normalize_value(kind, value)})
    return stored
