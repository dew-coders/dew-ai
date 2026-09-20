"""TinyGPT — a small GPT-style causal transformer implemented from scratch in NumPy.

No PyTorch / TensorFlow / JAX: embeddings, multi-head self-attention,
feed-forward blocks, layer norm, the cross-entropy loss, full back-propagation
and the Adam optimizer are all hand-written below.

Architecture (pre-norm):
    x = tok_emb(chars) + pos_emb(positions)
    repeat N_LAYERS times:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    logits = LayerNorm(x) @ W_out
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app import config

# Serialises model access between the chat server (generation) and the
# training thread (continuous learning) so weights are never read mid-update.
MODEL_LOCK = threading.Lock()

LN_EPS = 1e-5
Params = Dict[str, np.ndarray]


# --------------------------------------------------------------------------- #
# Parameter initialisation
# --------------------------------------------------------------------------- #
def init_params(vocab_size: int, seed: int = 0) -> Params:
    rng = np.random.default_rng(seed)
    C, F = config.D_MODEL, config.D_FF

    def n(*shape, s=0.02):
        return rng.normal(0.0, s, shape).astype(np.float32)

    p: Params = {}
    p["wte"] = n(vocab_size, C)
    p["wpe"] = n(config.CONTEXT_LEN, C, s=0.01)
    for l in range(config.N_LAYERS):
        pre = f"l{l}_"
        p[pre + "ln1_g"] = np.ones(C, np.float32)
        p[pre + "ln1_b"] = np.zeros(C, np.float32)
        p[pre + "wqkv"] = n(C, 3 * C)
        p[pre + "bqkv"] = np.zeros(3 * C, np.float32)
        p[pre + "wo"] = n(C, C, s=0.02 / math.sqrt(2 * config.N_LAYERS))
        p[pre + "bo"] = np.zeros(C, np.float32)
        p[pre + "ln2_g"] = np.ones(C, np.float32)
        p[pre + "ln2_b"] = np.zeros(C, np.float32)
        p[pre + "w1"] = n(C, F)
        p[pre + "b1"] = np.zeros(F, np.float32)
        p[pre + "w2"] = n(F, C)
        p[pre + "b2"] = np.zeros(C, np.float32)
    p["lnf_g"] = np.ones(C, np.float32)
    p["lnf_b"] = np.zeros(C, np.float32)
    p["wout"] = n(C, vocab_size)
    p["bout"] = np.zeros(vocab_size, np.float32)
    return p


def param_count(p: Params) -> int:
    return int(sum(a.size for a in p.values()))


# --------------------------------------------------------------------------- #
# Primitive ops (forward + backward)
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _ln_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray):
    std = np.sqrt(x.var(axis=-1, keepdims=True) + LN_EPS)
    xhat = (x - x.mean(axis=-1, keepdims=True)) / std
    return g * xhat + b, (xhat, std)


def _ln_backward(dy: np.ndarray, cache, g: np.ndarray):
    """Returns (dx, dg, db). Compact layer-norm backward pass."""
    xhat, std = cache
    C = xhat.shape[-1]
    dxhat = dy * g
    dg = (dy * xhat).sum(axis=tuple(range(xhat.ndim - 1)))
    db = dy.sum(axis=tuple(range(xhat.ndim - 1)))
    axes = (-1,)
    dx = (1.0 / (C * std)) * (
        C * dxhat
        - dxhat.sum(axis=axes, keepdims=True)
        - xhat * (dxhat * xhat).sum(axis=axes, keepdims=True)
    )
    return dx.astype(np.float32), dg, db


def _gelu(x: np.ndarray) -> np.ndarray:
    u = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(u))


def _gelu_backward(dy: np.ndarray, x: np.ndarray) -> np.ndarray:
    u = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = 0.7978845608028654 * (1.0 + 3 * 0.044715 * x ** 2)
    return (dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)).astype(np.float32)


def _attention_forward(h: np.ndarray, p: Params, pre: str):
    """Causal multi-head self-attention. h: (B, T, C)."""
    B, T, C = h.shape
    H, Dh = config.N_HEADS, C // config.N_HEADS

    qkv = h @ p[pre + "wqkv"] + p[pre + "bqkv"]            # (B, T, 3C)
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)       # (B, H, T, Dh)
    k = k.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(Dh)   # (B, H, T, T)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), 1)
    scores = np.where(causal_mask, np.float32(-1e9), scores)
    probs = _softmax(scores)                               # (B, H, T, T)

    y = probs @ v                                          # (B, H, T, Dh)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = y @ p[pre + "wo"] + p[pre + "bo"]
    cache = (h, q, k, v, probs, causal_mask)
    return out.astype(np.float32), cache


def _attention_backward(dout: np.ndarray, cache, p: Params, pre: str):
    h, q, k, v, probs, mask = cache
    B, T, C = h.shape
    H, Dh = config.N_HEADS, C // config.N_HEADS
    y = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    y2 = y.reshape(B * T, C)
    d2 = dout.reshape(B * T, C)

    dbo = dout.sum(axis=(0, 1))
    dwo = y2.T @ d2
    dy = (dout @ p[pre + "wo"].T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # (B, H, T, Dh)

    dprobs = dy @ v.transpose(0, 1, 3, 2)                  # (B, H, T, T)
    dv = probs.transpose(0, 1, 3, 2) @ dy                  # (B, H, T, Dh)
    dscores = probs * (dprobs - (dprobs * probs).sum(axis=-1, keepdims=True))
    dscores = np.where(mask, np.float32(0.0), dscores.astype(np.float32))

    dq = (dscores @ k) / math.sqrt(Dh)                     # (B, H, T, Dh)
    dk = (dscores.transpose(0, 1, 3, 2) @ q) / math.sqrt(Dh)

    dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
    dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
    dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)
    dqkv = np.concatenate([dq, dk, dv], axis=-1)           # (B, T, 3C)
    qkv2 = dqkv.reshape(B * T, 3 * C)
    h2 = h.reshape(B * T, C)

    dbqkv = dqkv.sum(axis=(0, 1))
    dwqkv = h2.T @ qkv2
    dh = (dqkv @ p[pre + "wqkv"].T).astype(np.float32)
    return dh, {"wqkv": dwqkv.astype(np.float32), "bqkv": dbqkv,
                "wo": dwo.astype(np.float32), "bo": dbo}


def _mlp_forward(u: np.ndarray, p: Params, pre: str):
    h = u @ p[pre + "w1"] + p[pre + "b1"]
    a = _gelu(h)
    out = a @ p[pre + "w2"] + p[pre + "b2"]
    return out.astype(np.float32), (u, h, a)


def _mlp_backward(dout: np.ndarray, cache, p: Params, pre: str):
    u, h, a = cache
    u2 = u.reshape(-1, u.shape[-1])
    d2 = dout.reshape(-1, dout.shape[-1])
    db2 = dout.sum(axis=(0, 1))
    dw2 = a.reshape(-1, a.shape[-1]).T @ d2
    da = dout @ p[pre + "w2"].T
    dh = _gelu_backward(da, h)
    db1 = dh.sum(axis=(0, 1))
    dw1 = u2.T @ dh.reshape(-1, dh.shape[-1])
    du = dh @ p[pre + "w1"].T
    return du.astype(np.float32), {"w1": dw1.astype(np.float32), "b1": db1,
                                   "w2": dw2.astype(np.float32), "b2": db2}


# --------------------------------------------------------------------------- #
# Full forward / backward
# --------------------------------------------------------------------------- #
def forward(p: Params, idx: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    """idx: (B, T) int array of token ids. Returns (logits, caches)."""
    B, T = idx.shape
    x = p["wte"][idx] + p["wpe"][:T]
    caches: List[dict] = []
    for l in range(config.N_LAYERS):
        pre = f"l{l}_"
        h1, ln1c = _ln_forward(x, p[pre + "ln1_g"], p[pre + "ln1_b"])
        attn, ac = _attention_forward(h1, p, pre)
        u = x + attn
        h2, ln2c = _ln_forward(u, p[pre + "ln2_g"], p[pre + "ln2_b"])
        m, mc = _mlp_forward(h2, p, pre)
        x = u + m
        caches.append({"pre": pre, "x_in": x, "ln1c": ln1c, "ac": ac,
                       "ln2c": ln2c, "mc": mc})
    xf, lnfc = _ln_forward(x, p["lnf_g"], p["lnf_b"])
    logits = xf @ p["wout"] + p["bout"]
    caches.append({"pre": None, "lnfc": lnfc, "xf": xf})
    return logits.astype(np.float32), caches


def loss_and_grads(p: Params, idx: np.ndarray, targets: np.ndarray):
    """Cross-entropy on next-char prediction. Returns (loss, grads)."""
    B, T = idx.shape
    logits, caches = forward(p, idx)
    V = logits.shape[-1]
    flat = logits.reshape(B * T, V)
    tgt = targets.reshape(B * T)
    rows = np.arange(B * T)

    sm = _softmax(flat)
    loss = float(-np.log(sm[rows, tgt] + 1e-9).mean())

    dlogits = sm.copy()
    dlogits[rows, tgt] -= 1.0
    dlogits /= B * T

    grads: Params = {}
    final = caches.pop()
    xf = final["xf"]
    dx = (dlogits @ p["wout"].T).reshape(B, T, -1)
    grads["wout"] = (xf.reshape(B * T, -1).T @ dlogits).astype(np.float32)
    grads["bout"] = dlogits.sum(axis=0)
    dx, dg, db = _ln_backward(dx, final["lnfc"], p["lnf_g"])
    grads["lnf_g"], grads["lnf_b"] = dg, db

    # Backprop through embedding table comes at the very end.
    dwte = np.zeros_like(p["wte"])
    dwpe = np.zeros_like(p["wpe"])

    for cache in reversed(caches):
        pre = cache["pre"]

        # --- residual split: dx flows into both the MLP branch and straight down
        du_mlp, g_mlp = _mlp_backward(dx, cache["mc"], p, pre)
        for name, g in g_mlp.items():
            grads[pre + name] = g
        du, dg2, db2 = _ln_backward(du_mlp, cache["ln2c"], p[pre + "ln2_g"])
        grads[pre + "ln2_g"], grads[pre + "ln2_b"] = dg2, db2

        # --- attention branch: gradient into attn output = dL/du_total
        du_total = dx + du
        dh1, g_attn = _attention_backward(du_total, cache["ac"], p, pre)
        for name, g in g_attn.items():
            grads[pre + name] = g
        dx_branch, dg1, db1 = _ln_backward(dh1, cache["ln1c"], p[pre + "ln1_g"])
        grads[pre + "ln1_g"], grads[pre + "ln1_b"] = dg1, db1

        dx = du_total + dx_branch          # through both residuals

    # Backprop into the embedding tables (once, using the final dx).
    np.add.at(dwte, idx, dx)
    dwpe[: idx.shape[1]] += dx.sum(axis=0)

    grads["wte"] = dwte
    grads["wpe"] = dwpe
    return loss, grads


# --------------------------------------------------------------------------- #
# Adam optimizer
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params: Params, lr: float):
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Params, grads: Params, lr: float | None = None) -> float:
        lr = self.lr if lr is None else lr
        gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads.values())) + 1e-12
        scale = min(1.0, config.GRAD_CLIP / gnorm)
        self.t += 1
        b1, b2, eps = config.ADAM_BETAS[0], config.ADAM_BETAS[1], 1e-8
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k, g in grads.items():
            g = (g * scale).astype(np.float32)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            params[k] -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps)
        return gnorm


def lr_at(step: int) -> float:
    """Linear warmup then constant learning rate."""
    warm = config.LEARNING_RATE * min(1.0, (step + 1) / max(1, config.LR_WARMUP_STEPS))
    return warm


# --------------------------------------------------------------------------- #
# Persistence (atomic save so a running server can hot-reload weights)
# --------------------------------------------------------------------------- #
def save_weights(p: Params, path: Path | None = None) -> None:
    path = Path(path or config.WEIGHTS_PATH)
    tmp = path.with_suffix(".tmp.npz")
    meta = np.array([p["wte"].shape[0], config.D_MODEL, config.N_LAYERS,
                     config.N_HEADS, config.CONTEXT_LEN])
    with MODEL_LOCK:
        np.savez(tmp, __meta__=meta, **{k: v for k, v in p.items()})
        os.replace(tmp, path)          # atomic on POSIX + Windows


def load_weights(path: Path | None = None) -> Params | None:
    path = Path(path or config.WEIGHTS_PATH)
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files if k != "__meta__"}
