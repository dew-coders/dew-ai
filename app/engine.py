"""Engine: manages the loaded neural network.

Watches the weights file and hot-reloads whenever the continuous-learning
trainer writes a new checkpoint, so the running chat server improves over
time without a restart.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterator, Optional, Tuple

from app import config, model
from app.generate import generate, stream_generate
from app.tokenizer import CharTokenizer


class Engine:
    def __init__(self) -> None:
        self._params: Optional[Params] = None
        self._tok: Optional[CharTokenizer] = None
        self._mtime: float = -1.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        with self._lock:
            try:
                mtime = config.WEIGHTS_PATH.stat().st_mtime
            except OSError:
                mtime = -1.0
            if mtime != self._mtime:
                params = model.load_weights()
                tok = CharTokenizer.load()
                if params is not None and tok is not None:
                    self._params, self._tok, self._mtime = params, tok, mtime

    @property
    def ready(self) -> bool:
        self.refresh()
        return self._params is not None and self._tok is not None

    # ------------------------------------------------------------------ #
    def complete(self, prompt: str):
        """Blocking generation. Returns (reply, confidence) or ("", 0.0)."""
        self.refresh()
        if not self.ready:
            return "", 0.0
        with model.MODEL_LOCK:
            return generate(self._params, self._tok, prompt)

    def stream(self, prompt: str) -> Optional[Iterator[Dict]]:
        """Streaming generation under the model lock. Returns a generator."""
        self.refresh()
        if not self.ready:
            return None
        params, tok = self._params, self._tok

        def gen():
            with model.MODEL_LOCK:
                yield from stream_generate(params, tok, prompt)

        return gen()

    # ------------------------------------------------------------------ #
    def info(self) -> Dict:
        self.refresh()
        if not self.ready:
            return {"ready": False}
        return {
            "ready": True,
            "vocab_size": self._tok.vocab_size,
            "parameters": model.param_count(self._params),
            "context_len": config.CONTEXT_LEN,
            "layers": config.N_LAYERS,
            "heads": config.N_HEADS,
            "d_model": config.D_MODEL,
            "tokenizer": "char-level (from scratch)",
        }


# Type alias only used above.
Params = Dict[str, "object"]

engine = Engine()
