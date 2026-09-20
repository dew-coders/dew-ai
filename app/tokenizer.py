"""Character-level tokenizer, built from scratch (no external tokenizer libs).

A character-level vocabulary keeps the model tiny and dependency-free while
still being able to produce conversational text after training.
"""
from __future__ import annotations

import json
from typing import Iterable, List

from app import config

# Printable ASCII plus newline — guarantees the model can represent any text.
DEFAULT_CHARS = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"


class CharTokenizer:
    def __init__(self, chars: Iterable[str]):
        self.chars: List[str] = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        # Unknown characters map to a space so encoding never crashes.
        self.fallback = self.stoi.get(" ", 0)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.fallback) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocab covering DEFAULT_CHARS plus anything in `text`."""
        chars = sorted(set(DEFAULT_CHARS) | set(text))
        return cls(chars)

    @classmethod
    def load(cls) -> "CharTokenizer | None":
        try:
            chars = json.loads(config.VOCAB_PATH.read_text(encoding="utf-8"))
            return cls(chars)
        except Exception:
            return None

    def save(self) -> None:
        config.VOCAB_PATH.write_text(json.dumps(self.chars), encoding="utf-8")
