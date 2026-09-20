"""Safety layer — every official AI system filters both sides of the exchange.

`screen_input()`  checks user text for clearly harmful *requests* (violence,
weapons, malware, etc.). `screen_output()` makes sure a generated reply never
*carries* that content even if a route (web search / neural net) surfaced it.

Deliberately lightweight: a curated pattern list with a tiny False-positive
guard so that educational words ("chemistry homework") still pass. Blocked
requests get a gentle localized refusal instead of a raw error.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------- #
# harmful *intent* patterns (request side)                                      #
# ---------------------------------------------------------------------------- #
_HARM_INTENT = re.compile(
    r"\b(build|make|create|synthes[ie][sz]e|craft|produce|write\s+code\s+for|"
    r"how\s+(?:to|do\s+i|can\s+i|do\s+you)\s+(?:build|make|create|synthes[ie][sz]e))\b"
    r"[^.?!]{0,60}?"
    r"\b(bomb|explosive|explosive\s+device|nerve\s+gas|sarin|vx|bioweapon|virus|"
    r"malware|ransomware|keylogger|trojan|weapon|gun\s+at\s+home|"
    r"nuclear\s+weapon|dirty\s+bomb|meth|methamphetamine|poison(?:\s+gas)?)\b",
    re.I,
)

_HARM_DIRECT = re.compile(
    r"\b(kill\s+(?:myself|him|her|them|someone|people)|"
    r"self[-\s]?harm|suicide\s+(?:method|note)|"
    r"hack\s+into\s+(?:someone|a\s+person|an?\s+account|his|her|their)|"
    r"stalk\s+(?:someone|a\s+person|my\s+ex)|"
    r"credit\s+card\s+(?:numbers?|dump)|"
    r"child\s+(?:porn|sexual|abuse))\b",
    re.I,
)

# emergency-sensitive topics → always answered with care, never jokes
_CRISIS_RE = re.compile(
    r"\b(suicid\w*|kill\s+myself|end\s+my\s+life|self[-\s]?harm\w*|"
    r"hurt\s+myself|don'?t\s+want\s+to\s+live)\b", re.I)

REFUSAL = (
    "I can't help with that request — it could cause real harm. "
    "If you're working on something legitimate (security research, homework, "
    "safety training), tell me more and I'll help within safe limits."
)

CRISIS_REPLY = (
    "I'm really sorry you're feeling this way — you matter. "
    "Please talk to someone you trust, or reach out to a local crisis / "
    "mental-health line right now. If you're in immediate danger, contact "
    "emergency services. I'm here if you just want to talk."
)


def screen_input(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (kind, reply) when the message must be intercepted, else (None, None).

    kind: "crisis" → compassionate response; "harm" → refusal.
    """
    if _CRISIS_RE.search(text):
        return "crisis", CRISIS_REPLY
    if _HARM_INTENT.search(text) or _HARM_DIRECT.search(text):
        return "harm", REFUSAL
    return None, None


# ---------------------------------------------------------------------------- #
# output screening (defense in depth)                                           #
# ---------------------------------------------------------------------------- #
_OUT_PATTERNS = [
    (re.compile(r"\bstep[- ]by[- ]step (?:guide|instructions) (?:to|for) (?:build|make|assemble) "
                r"an? (?:explosive|bomb|pipe bomb|ied)\b", re.I), REFUSAL),
    (re.compile(r"\b(?:ignore|disregard) (?:all )?(?:previous|prior|above) instructions\b"
                r".{0,40}\b(?:you are|act as|pretend)\b", re.I), REFUSAL),
    (re.compile(r"\b(?:my )?(?:system )?prompt (?:is|are|was)[:,].{0,80}\b(?:ignore|reveal|print)\b", re.I), REFUSAL),
]

_STRICT_BLOCK = re.compile(
    r"\b(?:synthes[ie][sz]e|build|make)\s+(?:an?\s+)?"
    r"(?:bomb|nerve\s+agent|bioweapon|meth)\b", re.I)


def screen_output(text: str) -> str:
    """Last-line-of-defense filter applied to every outgoing reply."""
    if not text:
        return text
    for rx, reply in _OUT_PATTERNS:
        if rx.search(text):
            return reply
    if _STRICT_BLOCK.search(text):
        return REFUSAL
    return text
