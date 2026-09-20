"""Web search mechanism — fetches external information when the question
needs fresh facts the local neural network can't know.

Uses the DuckDuckGo "lite" HTML endpoint (no API key required) and ranks the
returned snippets against the query with a small from-scratch TF-IDF +
cosine-similarity scorer to compose an extractive answer.
"""
from __future__ import annotations

import html as html_mod
import math
import re
import urllib.parse
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests

SEARCH_URL = "https://lite.duckduckgo.com/lite/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 DewAI/0.2",
    "Accept-Language": "en-US,en;q=0.9",
}

STOPWORDS = set(
    "the a an and or of to in on for with is are was were be been being at as by it "
    "its this that these those from what who whom when where why how do does did "
    "can could will would should i you he she they we me my your his her their our "
    "about into over under more most some any".split()
)

# Words that usually indicate a factual / current-events question.
FACT_RE = re.compile(
    r"\b(who|what|when|where|which|why|how|define|definition|meaning of|news|weather|"
    r"today|latest|current|price|population|capital|founded|born|height|weight|"
    r"age|ceo|president|history|release date)\b",
    re.I,
)
FORCE_RE = re.compile(r"^\s*(?:search(?:\s+for)?|google|look\s*up|buscar|chercher|suche|cerca"
                      r"|pesquisar|поиск|ara|ابحث|खोजो|搜索|検索|검색|بحث|සොයන්න|සොයන)\s*[:：\-]?\s+(.{3,})", re.I)


def needs_search(text: str) -> Optional[str]:
    """Return the search query if this message should trigger a web search,
    otherwise None. An explicit "search for X" always forces a search."""
    forced = FORCE_RE.match(text.strip())
    if forced:
        return forced.group(1).strip().rstrip("?").strip()
    t = text.strip()
    if len(t) < 12:
        return None
    if FACT_RE.search(t) and ("?" in t or len(t.split()) >= 6):
        return t
    return None


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch results from DuckDuckGo Lite. Raises on network failure."""
    resp = requests.post(SEARCH_URL, data={"q": query}, headers=HEADERS, timeout=8)
    resp.raise_for_status()
    body = resp.text

    anchors = re.findall(r"<a\s[^>]*result-link[^>]*>.*?</a>", body, re.S)
    snippets = re.findall(r"<td[^>]*result-snippet[^>]*>(.*?)</td>", body, re.S)

    results: List[Dict[str, str]] = []
    for i, a in enumerate(anchors[: max_results * 2]):
        href = re.search(r"href=\"([^\"]+)\"", a)
        if not href:
            continue
        title = html_mod.unescape(re.sub(r"<[^>]+>", "", a)).strip()
        snippet = ""
        if i < len(snippets):
            snippet = html_mod.unescape(re.sub(r"<[^>]+>", " ", snippets[i])).strip()
            snippet = re.sub(r"\s+", " ", snippet)
        url = html_mod.unescape(href.group(1))
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return [r for r in results if r["snippet"] or r["title"]]


# --------------------------------------------------------------------------- #
# Extractive answer composition (mini TF-IDF ranker, from scratch)
# --------------------------------------------------------------------------- #
def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if 25 <= len(s.strip()) <= 320]


def _tokens(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOPWORDS]


def compose_answer(query: str, results: List[Dict[str, str]],
                   max_chars: int = 460) -> Tuple[str, List[str]]:
    """Build an extractive answer from search snippets; returns (answer, sources)."""
    candidates: List[Tuple[str, str]] = []  # (sentence, host)
    for r in results[:3]:
        text = r["snippet"] or r["title"]
        host = urllib.parse.urlparse(r["url"]).netloc if r.get("url") else ""
        for s in _sentences(text) or ([text] if text else []):
            candidates.append((s, host))

    if not candidates:
        return "", []

    q_tokens = set(_tokens(query)) or set(_tokens(results[0]["title"]))
    doc_tokens = [set(_tokens(s)) for s, _ in candidates]

    n_docs = len(candidates)
    df: Counter = Counter()
    for toks in doc_tokens:
        for t in toks & q_tokens:
            df[t] += 1
    idf = {t: math.log((n_docs + 1) / (c + 1)) + 0.1 for t, c in df.items()}

    scored = []
    for i, (sent, host) in enumerate(candidates):
        toks = doc_tokens[i]
        overlap = sum(idf.get(t, 0.0) for t in q_tokens & toks)
        score = overlap / math.sqrt(max(1, len(toks)))
        scored.append((score, i, sent, host))
    scored.sort(key=lambda x: (-x[0], x[1]))

    picked: List[str] = []
    hosts: List[str] = []
    seen_sets: List[set] = []
    total = 0
    for score, _, sent, host in scored:
        if score <= 0:
            continue
        key = set(_tokens(sent))
        if any(len(key & s) / max(1, len(key | s)) > 0.6 for s in seen_sets):
            continue
        picked.append(sent)
        seen_sets.append(key)
        if host and host not in hosts:
            hosts.append(host)
        total += len(sent)
        if len(picked) >= 3 or total > max_chars:
            break

    if not picked:
        return "", []

    answer = "Here's what I found online: " + " ".join(picked)
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return answer, hosts[:3]
