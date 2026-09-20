"""Knowledge engine — retrieval over the long-term dataset memory.

Dew AI's neural net is tiny (0.6M params), so it can't memorize 24k news
headlines from limited training steps. This module gives it *retrieval*:
the stored news headlines (Sinhala and any other language) are indexed with
a from-scratch TF-IDF engine, and when a user's message looks like a
knowledge question the most relevant headlines are attached to the answer
with source metadata. Matching is character n-gram based, so it works across
scripts without extra dependencies.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app import datasets

_LOCK = threading.Lock()
_index: List[Dict] = []          # [{text, dataset, line_no}]
_index_stats: Dict = {"docs": 0, "datasets": 0, "built_at": ""}

_TOK = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(text)]


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return [t[i:i + n] for i in range(max(1, len(t) - n + 1))] or [t]


# --------------------------------------------------------------------------- #
# Index building (called on boot and after dataset (re-)downloads)
# --------------------------------------------------------------------------- #
def build_index() -> Dict:
    """Index every corpus file in the dataset registry."""
    global _index, _index_stats
    with _LOCK:
        docs: List[Dict] = []
        for path in datasets.corpus_paths():
            try:
                lines = Path(path).read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                text = line.strip()
                if len(text) < 12:      # skip tiny fragments
                    continue
                # never serve raw chat-training format lines as knowledge
                if text.startswith(("User:", "Bot:")):
                    continue
                docs.append({"text": text, "dataset": Path(path).stem,
                             "line_no": i})
        df: Counter = Counter()
        for d in docs:
            df.update(set(_char_ngrams(d["text"])))
        for d in docs:
            grams = _char_ngrams(d["text"])
            tf = Counter(grams)
            d["tfidf"] = {g: (1 + math.log(c)) * math.log((len(docs) + 1) / (df[g] + 1))
                          for g, c in tf.items()}
        _index = docs
        _index_stats = {"docs": len(docs),
                        "datasets": len({d["dataset"] for d in docs}),
                        "built_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
    return dict(_index_stats)


def stats() -> Dict:
    with _LOCK:
        return dict(_index_stats)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def search(query: str, top_k: int = 3) -> List[Dict]:
    """Most relevant knowledge snippets for a query, with relevance scores."""
    if not _index:
        return []
    q_grams = Counter(_char_ngrams(query))
    scored: List[Tuple[float, int, Dict]] = []
    for i, d in enumerate(_index):
        score = 0.0
        for g, q_count in q_grams.items():
            w = d["tfidf"].get(g)
            if w:
                score += w * (1 + math.log(q_count))
        if score > 0:
            scored.append((score / math.sqrt(max(1, len(d["tfidf"]))), i, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"text": d["text"], "dataset": d["dataset"],
             "line_no": d["line_no"],
             "score": round(score, 4)}
            for score, _, d in scored[:top_k]]


def looks_like_knowledge_query(text: str) -> bool:
    """Heuristic: does this message want factual/knowledge answers?"""
    t = text.strip()
    if len(t) < 10:
        return False
    kw = ("what", "who", "when", "where", "why", "how", "news", "tell me about",
          "define", "explain", "latest", "today")
    low = t.lower()
    if any(k in low for k in kw):
        return True
    # non-Latin scripts (Sinhala etc.): rely on question marks / length
    if any("\u0d80" <= c <= "\u0dff" for c in t) and len(t) >= 12:
        return True
    return False


def answer(query: str, max_chars: int = 420) -> Optional[Dict]:
    """Compose a knowledge-grounded answer snippet from the datasets."""
    hits = search(query, top_k=3)
    if not hits or hits[0]["score"] < 0.25:
        return None
    # safety net: strip any chat-format leakage from retrieved lines
    lines = [re.sub(r"^(User|Bot)\s*:\s*", "", h["text"]) for h in hits]
    text = " | ".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return {"text": text, "hits": hits}
