"""Response post-processing: small deterministic transforms that make the
tiny model's output feel more conversational ("humanized") — contractions,
gentle softeners, clean punctuation.
"""
from __future__ import annotations

import re
import zlib

CONTRACTIONS = [
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bi am\b", "I'm"),
    (r"\byou are\b", "you're"),
    (r"\bwe are\b", "we're"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\blet us\b", "let's"),
    (r"\bi will\b", "I'll"),
    (r"\bthere is\b", "there's"),
]

SOFTENERS = [
    "I think ",
    "Honestly, ",
    "Well, ",
    "From what I know, ",
    "Sure thing — ",
    "Good question — ",
]


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def humanize(text: str, allow_softener: bool = True) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return t

    # Tidy spacing around punctuation.
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)

    # Natural contractions.
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    # Occasionally add a conversational softener (deterministic per message).
    lower = t.lower()
    if (allow_softener and len(t) > 80 and _stable_hash(t) % 3 == 0
            and not lower.startswith(("i think", "honestly", "well,", "sure", "good question"))):
        soft = SOFTENERS[_stable_hash(t) % len(SOFTENERS)]
        t = soft + t[0].lower() + t[1:]

    # Clean casing and punctuation.
    t = t[0].upper() + t[1:]
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\.{4,}", "...", t)
    if t[-1] not in ".!?…\"):":
        t += "."
    return t
