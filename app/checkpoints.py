"""Model version registry — official-AI-style checkpointing for Dew AI.

Every successful training run snapshots the weights + tokenizer into
`data/checkpoints/` with metadata (trigger, loss curve, sample counts). This
gives the system:

- versioned models (`list_checkpoints()`),
- one-click rollback to any previous version (`rollback()` — weights and
  vocab are restored and the live engine hot-reloads them),
- automatic pruning so the disk never fills up.

It mirrors what production LLM systems do, scaled to this from-scratch codebase.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from app import config


def _ckpt_dir(ckpt_id: str) -> Path:
    return config.CHECKPOINTS_DIR / ckpt_id


def save_checkpoint(params_size: int, run_id: int, trigger: str, steps: int,
                    loss_before: float, loss_after: float,
                    n_samples: int) -> Optional[Dict]:
    """Snapshot the freshly saved weights + vocab as a named version.

    Call immediately after model.save_weights() inside a training run.
    """
    try:
        if not config.WEIGHTS_PATH.exists():
            return None
        ts = int(time.time())
        ckpt_id = f"v{ts}_{run_id:04d}_{trigger}"
        dest = _ckpt_dir(ckpt_id)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.WEIGHTS_PATH, dest / "model_weights.npz")
        if config.VOCAB_PATH.exists():
            shutil.copy2(config.VOCAB_PATH, dest / "vocab.json")
        meta = {
            "id": ckpt_id,
            "run_id": run_id,
            "trigger": trigger,
            "steps": steps,
            "loss_before": round(float(loss_before), 4),
            "loss_after": round(float(loss_after), 4),
            "n_samples": n_samples,
            "parameters": params_size,
            "vocab_size": _vocab_size(),
            "created_at": ts,
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _prune()
        return meta
    except OSError as exc:  # checkpointing must never break training
        print(f"[checkpoints] save failed: {exc}", flush=True)
        return None


def _vocab_size() -> int:
    try:
        chars = json.loads(config.VOCAB_PATH.read_text(encoding="utf-8"))
        return len(chars)
    except (OSError, ValueError):
        return 0


def list_checkpoints() -> List[Dict]:
    """All saved model versions, newest first."""
    out: List[Dict] = []
    if not config.CHECKPOINTS_DIR.exists():
        return out
    for meta_path in sorted(config.CHECKPOINTS_DIR.glob("*/meta.json"),
                            reverse=True):
        try:
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def latest() -> Optional[Dict]:
    cps = list_checkpoints()
    return cps[0] if cps else None


def get_checkpoint(ckpt_id: str) -> Optional[Dict]:
    for c in list_checkpoints():
        if c["id"] == ckpt_id:
            return c
    return None


def checkpoint_weights_path(ckpt_id: str) -> Optional[Path]:
    """Path to a checkpoint's weights file (None if the version is unknown
    or its file is missing) — used by the download endpoint."""
    d = _ckpt_dir(ckpt_id)
    if not (d / "meta.json").exists():
        return None
    weights = d / "model_weights.npz"
    return weights if weights.exists() else None


def rollback(ckpt_id: Optional[str] = None, steps_back: int = 1) -> Optional[Dict]:
    """Restore a previous model version as the live weights.

    Pass an explicit `ckpt_id`, or `steps_back=1` for the previous version.
    The restored weights+vocab replace the live files; the engine hot-reloads
    them on its next mtime check.
    """
    cps = list_checkpoints()
    if not cps:
        return None
    if ckpt_id is None:
        # live weights are (usually) the newest checkpoint → go one further back
        idx = min(max(1, steps_back), len(cps) - 1) if len(cps) > 1 else 0
        target = cps[idx]
    else:
        target = get_checkpoint(ckpt_id)
        if target is None:
            return None
    src = _ckpt_dir(target["id"])
    weights, vocab = src / "model_weights.npz", src / "vocab.json"
    if not weights.exists():
        return None
    shutil.copy2(weights, config.WEIGHTS_PATH)
    if vocab.exists():
        shutil.copy2(vocab, config.VOCAB_PATH)
    # refresh the live engine against the restored files
    from app import engine
    engine.engine.refresh()
    return target


def _prune(keep: Optional[int] = None) -> None:
    """Keep only the newest `keep` checkpoints (config.CHECKPOINT_KEEP)."""
    keep = config.CHECKPOINT_KEEP if keep is None else keep
    cps = list_checkpoints()
    for old in cps[keep:]:
        shutil.rmtree(_ckpt_dir(old["id"]), ignore_errors=True)


def stats() -> Dict:
    cps = list_checkpoints()
    return {"count": len(cps), "keep": config.CHECKPOINT_KEEP,
            "latest": cps[0]["id"] if cps else None,
            "versions": cps[:8]}
