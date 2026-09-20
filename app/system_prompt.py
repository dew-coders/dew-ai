"""System-prompt layer — how an official AI system "thinks" about itself.

Official assistants (ChatGPT / Gemini / Claude) don't just complete text;
every request is wrapped in a system prompt defining identity, capabilities,
memory of the user and behavioral rules. This module builds that context for
Dew AI from:

- the core persona (identity, creator, capabilities),
- the user's long-term memory (name, likes, notes — ChatGPT-style memory),
- the active conversation language,
- retrieved dataset knowledge / web results when available.

The block is injected at the front of the model prompt and also served via
/api/system-prompt for inspection and debugging.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app import config

CORE_PERSONA = (
    "You are Dew AI, a helpful, multilingual AI assistant created by Hansa Dewmina. "
    "You are built on a from-scratch NumPy transformer trained continuously on "
    "real conversations, curated datasets and user feedback. "
    "You can: chat naturally, execute tasks (math, date & time, unit conversion), "
    "search the live web, remember facts about the user long-term, answer from your "
    "dataset memory, generate images, and speak 14 languages. "
    "You are honest about uncertainty; when unsure, say so instead of inventing facts."
)

CAPABILITY_LINES = {
    "tool": "Task result from a built-in tool is provided below — present it clearly.",
    "search": "Live web results are provided below — summarize them faithfully and cite hosts.",
    "knowledge": "Relevant passages from your long-term dataset memory are below — ground your answer in them.",
    "image": "The user asked for an image — the generator has handled it; describe it warmly.",
}


def build(user: Optional[Dict], lang: str, memory_block: str = "",
          knowledge_block: str = "", route: str = "") -> str:
    """Assemble the full system prompt for one request."""
    parts: List[str] = [f"[System]\n{CORE_PERSONA}"]

    name = (user or {}).get("username")
    if name:
        parts.append(f"[System] You are talking with {name}.")

    if lang != "en":
        parts.append(f"[System] Reply in the user's language ({lang}).")

    if memory_block:
        parts.append(f"[Memory]\n{memory_block}")

    cap = CAPABILITY_LINES.get(route)
    if cap:
        parts.append(f"[System] {cap}")

    if knowledge_block:
        parts.append(f"[Context]\n{knowledge_block}")

    parts.append("[System] Stay concise, friendly and safe. Never claim to be "
                 "another AI product.")
    return "\n".join(parts)


def inspect(user: Optional[Dict], lang: str, memory_block: str = "",
            knowledge_block: str = "", route: str = "") -> Dict:
    """Structured view for the /api/system-prompt endpoint."""
    return {
        "core_persona": CORE_PERSONA,
        "assembled": build(user, lang, memory_block, knowledge_block, route),
        "user": (user or {}).get("username"),
        "lang": lang,
        "route": route,
        "memory_injected": bool(memory_block),
        "knowledge_injected": bool(knowledge_block),
    }
