"""Import the SinhalaASR-1000 transcriptions from Hugging Face.

Dew AI's neural net is a text model, so the audio column isn't used — only
the Sinhala `sentence` transcriptions are. They're downloaded through the
public datasets-server API (no auth, no SDK), deduped, and combined with
conversational User/Bot pairs built from the Sinhala reply templates, so the
model learns both the Sinhala script and how to chat in Sinhala.

Usage:  python -m app.import_dataset
"""
from __future__ import annotations

import time
from typing import Dict, List

import requests

from app import config, languages

DATASET = "Ransaka/SinhalaASR-1000"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100

# Sinhala prompts paired with the localized replies (teaches the chat format).
_SI_PROMPTS: Dict[str, str] = {
    "greeting": "ආයුබෝවන්",
    "identity": "ඔයා කවුද?",
    "creator": "ඔයාව හදපු කවුද?",
    "help": "ඔයාට මොනවද කරන්න පුළුවන්?",
    "thanks": "ස්තූතියි!",
    "bye": "ගිහින් එන්නම්",
    "fallback": "මොකක් හරි කියන්න",
}

# Short everyday exchanges that teach the model simple Sinhala dialogue.
_SI_EXTRA: List[tuple] = [
    ("කොහොමද ඉන්නේ?", "හොඳයි, ස්තූතියි! ඔයා කොහොමද ඉන්නේ?"),
    ("ඔයාගේ නම මොකක්ද?", "මගේ නම Dew AI."),
    ("ඔයා හොඳද?", "ඔව්, මම හොඳයි! ඔයා මොකද කරන්නේ?"),
    ("ආයුබෝවන්, මොකද වෙන්නේ?", "මම හොඳට ඉන්නවා. ඔයාට කොහොමද උදව් කරන්න පුළුවන්?"),
    ("ඔයා ඉගෙන ගන්නවාද?", "ඔව්! අපේ හැම කතාවකින්ම මම අලුත් දෙයක් ඉගෙන ගන්නවා."),
    ("සුබ උදෑසනක්", "සුබ උදෑසනක්! අද දවස ඔයාට සතුටක් වේවා."),
    ("ඔයා ලොකුද?", "නෑ, මම කුඩා නියුරල් ජාලයක් — ඒත් හැම දවසකම වැඩෙනවා."),
    ("ඔයාට කාගේ උදව් ඕනද?", "මට මගේ නිර්මාණකරු Hansa Dewmina ගේ උදව් තියෙනවා."),
    ("මම ඔයාට ආදරෙයි", "ස්තූතියි! මමත් ඔයාව සතුටු කරන්න කැමති."),
    ("ඔයා සිංහල කතා කරනවද?", "ඔව්! මම සිංහලෙන් කතා කරනවා. ඔයාට පුළුවන් මගෙන් ඕනම දෙයක් අහන්න."),
]


def fetch_sentences(max_rows: int = 1000) -> List[str]:
    """Download the `sentence` transcriptions (paginated)."""
    sentences: List[str] = []
    offset = 0
    while offset < max_rows:
        resp = requests.get(API, params={
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": min(PAGE, max_rows - offset),
        }, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("rows", [])
        if not rows:
            break
        for row in rows:
            text = (row["row"].get("sentence") or "").strip()
            if text:
                sentences.append(text)
        offset += len(rows)
        time.sleep(0.2)  # be gentle with the public API
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def build_corpus(sentences: List[str]) -> str:
    """Conversational Sinhala pairs + raw ASR sentences as language-model data."""
    parts: List[str] = []
    for kind, prompt in _SI_PROMPTS.items():
        for reply in languages.TEMPLATES["si"].get(kind, []):
            parts.append(f"User: {prompt}\nBot: {reply}\n")
    for prompt, reply in _SI_EXTRA:
        parts.append(f"User: {prompt}\nBot: {reply}\n")
    parts.append("")
    parts.extend(sentences)
    return "\n".join(parts) + "\n"


def main() -> None:
    print(f"fetching transcriptions from {DATASET} …")
    sentences = fetch_sentences()
    print(f"got {len(sentences)} unique Sinhala sentences")
    corpus = build_corpus(sentences)
    config.SINHALA_CORPUS_PATH.write_text(corpus, encoding="utf-8")
    print(f"wrote {config.SINHALA_CORPUS_PATH} "
          f"({len(corpus):,} chars) — included in the next training run")


if __name__ == "__main__":
    main()
