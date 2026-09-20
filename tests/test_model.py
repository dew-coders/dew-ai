"""Sanity tests for the from-scratch model.

Run:  python3 tests/test_model.py

1. Numerical gradient check — verifies the hand-written backpropagation
   against central finite differences.
2. Loss-decrease check — verifies the optimizer actually learns.
3. Tokenizer round-trip.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import config, model
from app.tokenizer import CharTokenizer

# Use a tiny config so finite differences are fast.
config.D_MODEL = 16
config.N_LAYERS = 2
config.N_HEADS = 2
config.D_FF = 32
config.CONTEXT_LEN = 8
config.BATCH_SIZE = 2

VOCAB = 30
rng = np.random.default_rng(7)
params = model.init_params(VOCAB, seed=7)
idx = rng.integers(0, VOCAB, size=(config.BATCH_SIZE, config.CONTEXT_LEN))
targets = rng.integers(0, VOCAB, size=(config.BATCH_SIZE, config.CONTEXT_LEN))


def check_gradients():
    loss, grads = model.loss_and_grads(params, idx, targets)
    eps = 1e-2
    checked = 0
    worst = 0.0
    for name in sorted(params):
        p = params[name]
        flat = p.reshape(-1)
        gflat = grads[name].reshape(-1)
        for j in rng.choice(flat.size, size=min(3, flat.size), replace=False):
            orig = flat[j]
            flat[j] = orig + eps
            lp, _ = model.loss_and_grads(params, idx, targets)
            flat[j] = orig - eps
            lm, _ = model.loss_and_grads(params, idx, targets)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[j]
            # Combined tolerance: float32 finite differences have a noise floor
            # of roughly machine-eps(loss)/eps ≈ 1e-5 on the numerical value,
            # so tiny gradient components are compared with an absolute bound.
            abs_err = abs(num - ana)
            rel = abs_err / max(1e-6, abs(num) + abs(ana))
            assert rel < 0.05 or abs_err < 1e-4, (
                f"gradient mismatch at {name}[{j}]: numerical={num:.6f} "
                f"analytic={ana:.6f} rel_err={rel:.4f} abs_err={abs_err:.2e}")
            worst = max(worst, rel)
            checked += 1
    print(f"✓ gradient check passed ({checked} samples, worst rel err {worst:.2e})")


def check_loss_decreases():
    p = model.init_params(VOCAB, seed=1)
    adam = model.Adam(p, lr=1e-2)
    first = None
    for step in range(60):
        loss, grads = model.loss_and_grads(p, idx, targets)
        adam.step(p, grads)
        if first is None:
            first = loss
    assert loss < first * 0.7, f"loss did not decrease enough: {first:.4f} → {loss:.4f}"
    print(f"✓ optimizer learns: loss {first:.4f} → {loss:.4f} in 60 steps")


def check_tokenizer():
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    assert tok.encode("café ☕")  # unknown chars fall back gracefully
    print("✓ tokenizer round-trip + unknown-char fallback")


if __name__ == "__main__":
    check_tokenizer()
    check_gradients()
    check_loss_decreases()
    print("all model tests passed ✅")
