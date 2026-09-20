"""Intent router — ChatGPT-style message dispatch for Dew AI.

Classifies each user message into one route; the chat pipeline picks the
handler:

  chitchat   → localized template answers (greetings/thanks/bye/identity/creator)
  tool       → task execution (math, date/time, unit conversion)
  image      → image generation
  search     → live web search (DuckDuckGo)
  knowledge  → dataset-memory retrieval (long-term memory)
  neural     → the from-scratch transformer (creative/conversational)

Rules run in priority order. The route is returned with metadata so the
pipeline can record it and the UI can show which "model" handled a message.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from app import languages, tools

ROUTES = ("chitchat", "tool", "image", "search", "knowledge", "neural")


def _re(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


# --------------------------------------------------------------------------- #
# image generation triggers (multi-language)
# --------------------------------------------------------------------------- #
_IMAGE_RE = _re(
    r"\b(draw|paint|sketch|generate|create|make|show)\b[^.?!]{0,30}\b(image|picture|photo|drawing|art|pic)\b"
    r"|\b(image|picture|photo|pic)\s+of\b"
    # sinhala (written + spoken)
    r"|රූපයක්"
    r"|පින්තූරයක්"
    r"|චායාරූපයක්"
    r"|චිත්\u200d‍රයක්"
    r"|රූපය\s"
    r"|පින්තූරය\s"
    # spanish / french / german / portuguese / italian
    r"|\b(dibuja|dibujar|genera[r]?|crea[r]?)\b[^.?!]{0,20}\b(imagen|dibujo|foto)\b"
    r"|\b(dessine|dessiner|g[eé]n[e]re[r]?|cr[eé]e[r]?)\b[^.?!]{0,20}\b(image|dessin|photo)\b"
    r"|\b(zeichne|male|erstell[e]?)\b[^.?!]{0,20}\b(bild|foto|zeichnung)\b"
    r"|\b(desenh[ao]|cria[r]?|gera[r]?)\b[^.?!]{0,20}\b(imagem|desenho|foto)\b"
    r"|\b(disegna|crea|genera)\b[^.?!]{0,20}\b(imma[gq]ine|disegno|foto)\b"
    # russian / turkish / arabic / hindi
    r"|(нарисуй|нарисовать|создай|создать)"
    r"|(çiz\b|resim\s*(yap|oluştur))"
    r"|(ارسم|رسم\s+لي|أنشئ\s+صورة|اصنع\s+صورة)"
    r"|(चित्र|तस्वीर)\s*(बनाओ|बनाइए|दिखाओ|बना)"
    # cjk
    r"|(画|畫)\s*(一|一個|一个|一幅|一張|一张)?"
    r"|(描いて|書いて|画像を作|画像を生成)"
    r"|(그림을\s*그려|이미지를\s*만들|그림\s*그려줘)"
)

_IMG_STRIP_RE = _re(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:draw|paint|sketch|generate|create|make|show\s*(?:me)?)\s+"
    r"(?:(?:me\s+)?(?:an?\s+|the\s+)?)?"
    r"(?:image|picture|photo|drawing|art|pic)\s*(?:of\s+|about\s+)?"
    r"|^\s*(?:image|picture|photo|pic)\s+of\s+"
    r"|රූපයක්\s*(?:හදන්න|හදපන්|හදමු|අඳින්න|අදින්න|පේන්න|පෙන්නන්න|හදන්නේ)?\s*[:：]?"
    r"|පින්තූරයක්\s*(?:හදන්න|හදපන්|හදමු|පෙන්නන්න)?\s*[:：]?"
    r"|චායාරූපයක්\s*(?:හදන්න|පෙන්නන්න)?\s*[:：]?"
    r"|රූපය\s*(?:හදන්න|පෙන්නන්න)?\s*[:：]?"
    r"|\s*(?:dibuja|dibujar|genera|generar|crea|crear)\s+(?:un[ao]?\s+|la\s+)?(?:imagen|dibujo|foto)\s*(?:de\s+|sobre\s+)?"
    r"|\s*(?:dessine|dessiner|g[eé]n[e]re|g[eé]n[e]rer|cr[eé]e|cr[eé]er)\s+(?:un[ae]?\s+|le\s+|la\s+)?(?:image|dessin|photo)\s*(?:de\s+|d')?"
    r"|\s*(?:zeichne|male|erstell[e]?)\s+(?:ein(e|en)?\s+|das\s+)?(?:bild|foto|zeichnung)\s*(?:von\s+)?"
    r"|\s*(?:нарисуй|нарисовать|создай|создать)\s+"
    r"|\s*(?:çiz|resim\s*(?:yap|oluştur))\s*[:：]?"
    r"|\s*(?:ارسم|أنشئ\s+صورة|اصنع\s+صورة)\s*"
    r"|\s*(?:चित्र|तस्वीर)\s*(?:बनाओ|बनाइए|दिखाओ|बना)?\s*[:：]?"
    r"|\s*(?:画|畫)\s*(?:一|一個|一个|一幅|一張|一张)?\s*"
    r"|\s*(?:描いて|書いて|画像を作って|画像を生成)\s*[:：]?"
    r"|\s*(?:그림|이미지)(?:을|를)?\s*(?:그려|그려줘|만들어|생성)?\s*[:：]?"
)


def extract_image_prompt(text: str) -> Optional[str]:
    """Return the image prompt if the message asks for an image, else None."""
    if not _IMAGE_RE.search(text):
        return None
    prompt = _IMG_STRIP_RE.sub("", text.strip(), count=1)
    prompt = prompt.strip(" :：,،。\"'“”‘’").strip()
    return prompt or None


# --------------------------------------------------------------------------- #
# tool use (task execution) — delegates to app/tools.py detection
# --------------------------------------------------------------------------- #
def detect_tool(text: str) -> Optional[Tuple[str, str]]:
    """Return (tool_name, argument) if the message is a runnable task."""
    return tools.detect(text)


# --------------------------------------------------------------------------- #
# main classifier
# --------------------------------------------------------------------------- #
def classify(text: str) -> Tuple[str, dict]:
    """Return (route, meta) for a user message."""
    meta: dict = {}
    t = text.strip()

    # 1) chit-chat wins (instant, high-confidence localized replies)
    kind = languages.template_kind(t)
    if kind:
        meta["template_kind"] = kind
        return "chitchat", meta

    # 2) runnable tasks (math, time, conversions)
    tool = detect_tool(t)
    if tool:
        meta["tool"], meta["tool_arg"] = tool
        return "tool", meta

    # 3) image generation
    if _IMAGE_RE.search(t):
        meta["image_prompt"] = extract_image_prompt(t) or t.strip()[:120]
        return "image", meta

    # 4) explicit / factual live web search
    query = __import__("app.search", fromlist=["needs_search"]).needs_search(t)
    if query:
        meta["search_query"] = query
        return "search", meta

    # 5) knowledge / long-term dataset memory
    from app import knowledge
    if knowledge.looks_like_knowledge_query(t):
        return "knowledge", meta

    # 6) everything else → the neural net
    return "neural", meta
