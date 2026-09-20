"""Dataset builder — turn collected chat data into official-style training sets.

Official AI teams iterate on *datasets*, not just weights. This module turns
the continuous-learning collection (real chats + feedback + memory notes) into
reusable corpora:

- `build_chat_dataset()` : clean (prompt, response) pairs → JSONL, ready for
  another training pass or external tools. Downvoted samples are excluded;
  upvoted ones carry `"weight": 2`.
- `export_jsonl()`       : write the JSONL to data/exports/ (downloadable via
  GET /api/data/export).
- `stats()`              : how much data has been collected, by source & language.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from app import config, database as db


def build_chat_dataset(limit: int = 5000, quality: str = "all") -> List[Dict]:
    """Curated (prompt, response) pairs. quality: all | curated (👍 only)."""
    samples = db.get_training_samples(exclude_negative=True, limit=limit)
    out: List[Dict] = []
    seen: set = set()
    for s in samples:
        if quality == "curated" and (s.get("quality") or 0) < 1:
            continue
        prompt, response = (s["prompt"] or "").strip(), (s["response"] or "").strip()
        if not prompt or not response:
            continue
        key = (prompt.lower(), response.lower())
        if key in seen:                       # dedupe identical pairs
            continue
        seen.add(key)
        out.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "meta": {
                "source": s.get("source", "chat"),
                "lang": s.get("lang", "en"),
                "quality": s.get("quality", 0),
                "weight": 2 if (s.get("quality") or 0) > 0 else 1,
            },
        })
    return out


def export_jsonl(limit: int = 5000, quality: str = "all") -> Dict:
    """Write the collected dataset to data/exports/ and return file info."""
    rows = build_chat_dataset(limit=limit, quality=quality)
    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"chat_dataset_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    path = config.EXPORTS_DIR / fname
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"file": fname, "path": str(path), "rows": len(rows),
            "bytes": path.stat().st_size, "quality": quality}


def latest_export() -> Optional[Path]:
    if not config.EXPORTS_DIR.exists():
        return None
    files = sorted(config.EXPORTS_DIR.glob("chat_dataset_*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def stats() -> Dict:
    """Overview of the data-collection system (for /api/stats)."""
    s = db.stats()
    by_source: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    try:
        with db._connect() as conn:  # noqa: SLF001 — single tiny read query
            for row in conn.execute(
                    "SELECT source, lang, COUNT(*) c FROM training_samples "
                    "GROUP BY source, lang"):
                by_source[row["source"] or "chat"] = \
                    by_source.get(row["source"] or "chat", 0) + row["c"]
                by_lang[row["lang"] or "en"] = \
                    by_lang.get(row["lang"] or "en", 0) + row["c"]
    except Exception:
        pass
    return {
        "total_samples": s.get("training_samples", 0),
        "by_source": by_source,
        "by_lang": by_lang,
        "feedback_up": s.get("feedback_up", 0),
        "feedback_down": s.get("feedback_down", 0),
        "exports": [p.name for p in sorted(config.EXPORTS_DIR.glob("*.jsonl"))]
        if config.EXPORTS_DIR.exists() else [],
    }
