"""Text generation: temperature / top-k / top-p sampling on top of TinyGPT.

Also provides a character-by-character streaming generator used by the
WebSocket endpoint for real-time output.
"""
from __future__ import annotations

from typing import Dict, Iterator, List

import numpy as np

from app import config, model
from app.tokenizer import CharTokenizer

STOP_SUFFIXES = ("\n\n", "\nUser", "\nuser", "\nBot", "\nbot")


def _sample_next(logits: np.ndarray, temperature: float, top_k: int,
                 top_p: float, rng: np.random.Generator,
                 recent_ids: list | None = None,
                 rep_penalty: float | None = None):
    """Returns (token_id, probability) sampled with temperature / top-k / top-p
    and a repetition penalty over recently produced tokens (official-style
    decoding: keeps long generations on-topic without breaking rare chars)."""
    logits = logits.astype(np.float64) / max(1e-6, temperature)
    if recent_ids and rep_penalty and rep_penalty > 1.0:
        uniq, counts = np.unique(np.asarray(list(recent_ids[-64:])), return_counts=True)
        logits[uniq] /= (rep_penalty ** np.minimum(counts, 4.0))
    if top_k and 0 < top_k < logits.size:
        kth = np.sort(logits)[-top_k]
        logits[logits < kth] = -1e9
    probs = _softmax64(logits)
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        s = mask.sum()
        if s > 0:
            probs = mask / s
    idx = int(rng.choice(probs.size, p=probs))
    return idx, float(probs[idx])


def _softmax64(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _looks_loopy(out: str) -> bool:
    """Detect degenerate repetition that tiny char models sometimes fall into."""
    if len(out) >= 3 and out[-1] == out[-2] == out[-3] and out[-1] not in ".!?… \n":
        return True
    if len(out) >= 12 and out[-4:] == out[-8:-4] == out[-12:-8]:
        return True
    return False


def _loop(params, tok: CharTokenizer, prompt: str, max_new: int, temperature: float,
          top_k: int, top_p: float, rng: np.random.Generator):
    """Shared generation loop: yields (char, prob) one at a time."""
    ids = tok.encode(prompt)[-config.CONTEXT_LEN:]
    produced = 0
    recent: list = []
    for _ in range(max_new):
        window = np.array([ids[-config.CONTEXT_LEN:]], dtype=np.int64)
        logits, _ = model.forward(params, window)
        idx, prob = _sample_next(logits[0, -1], temperature, top_k, top_p, rng,
                                 recent_ids=recent, rep_penalty=config.REPETITION_PENALTY)
        ids.append(idx)
        recent.append(idx)
        yield tok.itos[idx], prob
        produced += 1
        if produced >= max_new:
            break


def generate(params, tok: CharTokenizer, prompt: str, max_new: int | None = None,
             temperature: float | None = None, top_k: int | None = None,
             top_p: float | None = None, seed: int | None = None):
    """Blocking generation. Returns (reply_text, mean_confidence)."""
    rng = np.random.default_rng(seed)
    out: List[str] = []
    confs: List[float] = []
    for ch, prob in _loop(params, tok, prompt, max_new or config.MAX_NEW_TOKENS,
                          temperature or config.TEMPERATURE, top_k or config.TOP_K,
                          top_p or config.TOP_P, rng):
        out.append(ch)
        confs.append(prob)
        text = "".join(out)
        if text.endswith(STOP_SUFFIXES) or _looks_loopy(text):
            break
    reply = "".join(out).split("\n", 1)[0].strip()
    conf = float(np.mean(confs)) if confs else 0.0
    return reply, conf


def stream_generate(params, tok: CharTokenizer, prompt: str,
                    chunk_size: int = 3, max_new: int | None = None,
                    temperature: float | None = None, top_k: int | None = None,
                    top_p: float | None = None, seed: int | None = None):
    """Streaming generation for real-time chat.

    Yields dicts: {"type": "delta", "text": ...} and finally
    {"type": "final", "reply": ..., "confidence": ...}.
    """
    rng = np.random.default_rng(seed)
    chars: List[str] = []
    confs: List[float] = []
    buffer = ""
    for ch, prob in _loop(params, tok, prompt, max_new or config.MAX_NEW_TOKENS,
                          temperature or config.TEMPERATURE, top_k or config.TOP_K,
                          top_p or config.TOP_P, rng):
        chars.append(ch)
        confs.append(prob)
        buffer += ch
        text = "".join(chars)
        if text.endswith(STOP_SUFFIXES) or _looks_loopy(text):
            break
        if len(buffer) >= chunk_size:
            yield {"type": "delta", "text": buffer}
            buffer = ""
    if buffer:
        yield {"type": "delta", "text": buffer}
    reply = "".join(chars).split("\n", 1)[0].strip()
    conf = float(np.mean(confs)) if confs else 0.0
    yield {"type": "final", "reply": reply, "confidence": conf}
