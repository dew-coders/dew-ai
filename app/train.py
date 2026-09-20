"""Training pipeline for TinyGPT.

- Builds the training corpus from the seed data plus every conversation
  collected in the database (continuous learning).
- Trains the from-scratch NumPy model with Adam.
- Runs in a background thread when enough new conversations accumulate,
  sharing the model lock with the chat server so live traffic stays responsive.
- Saves checkpoints atomically; the running server hot-reloads them.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app import config, database as db, datasets, model
from app.tokenizer import CharTokenizer

_TRAIN_GATE = threading.Lock()   # only one training run at a time

state: Dict[str, Optional[object]] = {
    "running": False,
    "progress": "idle",          # idle | building-dataset | training | saving | done | error
    "last_message": "",
}


def is_running() -> bool:
    return bool(state["running"])


def status() -> Dict:
    return dict(state)


# --------------------------------------------------------------------------- #
# Dataset construction (seed corpus + collected conversations)
# --------------------------------------------------------------------------- #
def build_training_text() -> Tuple[str, int, int]:
    """Returns (corpus_text, sample_count, cutoff_created_at)."""
    parts: List[str] = []
    if config.SEED_CORPUS_PATH.exists():
        parts.append(config.SEED_CORPUS_PATH.read_text(encoding="utf-8"))
    # Long-term dataset memory: every registered dataset corpus is included on
    # every (re)train — deleting the weights never loses the datasets.
    for path in datasets.corpus_paths():
        try:
            parts.append(Path(path).read_text(encoding="utf-8"))
        except OSError:
            continue

    samples = db.get_training_samples(exclude_negative=True)
    cutoff = db.last_completed_run_cutoff()
    cutoff_new = ""

    for s in samples:
        cutoff_new = max(cutoff_new, s["created_at"])
        block = f"User: {s['prompt']}\nBot: {s['response']}\n\n"
        reps = 1
        if s["quality"] and s["quality"] > 0:
            reps += 1                      # upvoted replies get extra weight
        if s["created_at"] > cutoff:
            reps += 2                      # freshly collected data gets extra weight
        parts.extend([block] * reps)

    text = "".join(parts)
    config.TRAIN_TEXT_PATH.write_text(text, encoding="utf-8")
    return text, len(samples), cutoff_new


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def run_training(steps: int, trigger: str = "manual",
                 finetune: bool = False, finetune_text: str = "") -> int:
    """Blocking training run. Returns the run id. Use start_background() for async.

    finetune=True: gentle low-LR pass on curated data only (finetune_text),
    keeping existing weights (ChatGPT-style fine-tuning instead of re-training)."""
    if not _TRAIN_GATE.acquire(blocking=False):
        raise RuntimeError("A training run is already in progress")

    run_id = -1
    held = False
    try:
        state.update(running=True, progress="building-dataset", last_message="")
        run_id = db.start_training_run(trigger, steps)

        if finetune:
            text = finetune_text
            n_samples = 0
            cutoff_created = ""
        else:
            text, n_samples, cutoff_created = build_training_text()
        data_stream = text[-600_000:] if len(text) > 600_000 else text

        # Vocabulary: reuse the saved one so existing weights stay compatible.
        tok = CharTokenizer.load() or CharTokenizer.from_text(data_stream)
        if not config.VOCAB_PATH.exists():
            tok.save()

        params = model.load_weights()
        if params is None:
            params = model.init_params(tok.vocab_size)
            state["progress"] = "training (from scratch)"
        elif finetune:
            state["progress"] = "fine-tuning (low LR)"

        data = np.array(tok.encode(data_stream), dtype=np.int64)
        if data.size < config.CONTEXT_LEN + 2:
            raise RuntimeError("Not enough data to train on")

        rng = np.random.default_rng(1234)
        base_lr = config.FINETUNE_LR if finetune else config.LEARNING_RATE
        adam = model.Adam(params, lr=base_lr)
        losses: List[float] = []
        CHUNK = 40          # steps per model-lock hold, keeps chat responsive
        t0 = time.time()

        for step in range(steps):
            if step % CHUNK == 0:
                if held:
                    model.MODEL_LOCK.release()
                    held = False
                model.MODEL_LOCK.acquire()
                held = True

            starts = rng.integers(0, data.size - config.CONTEXT_LEN - 1,
                                  size=config.BATCH_SIZE)
            x = np.stack([data[s:s + config.CONTEXT_LEN] for s in starts])
            y = np.stack([data[s + 1:s + 1 + config.CONTEXT_LEN] for s in starts])

            loss, grads = model.loss_and_grads(params, x, y)
            step_lr = model.lr_at(step) if not finetune else base_lr
            adam.step(params, grads, lr=step_lr)
            losses.append(loss)

            if step % 25 == 0 or step == steps - 1:
                msg = (f"step {step + 1}/{steps}  loss {loss:.4f}  "
                       f"lr {model.lr_at(step):.5f}  {time.time() - t0:.1f}s")
                state["last_message"] = msg
                print(f"[train:{trigger}] {msg}", flush=True)

        if held:
            model.MODEL_LOCK.release()
            held = False

        state["progress"] = "saving"
        model.save_weights(params)

        loss_before = float(np.mean(losses[: min(5, len(losses))]))
        loss_after = float(np.mean(losses[-min(20, len(losses)):]))
        db.finish_training_run(run_id, "done", n_samples, 0,
                               loss_before, loss_after)
        db.mark_samples_used(cutoff_created)
        # Model version registry: snapshot this run as a rollback-able version.
        try:
            from app import checkpoints
            checkpoints.save_checkpoint(model.param_count(params), run_id, trigger,
                                        steps, loss_before, loss_after, n_samples)
        except Exception as exc:  # noqa: BLE001 — never fail a run on housekeeping
            print(f"[train:{trigger}] checkpoint skipped: {exc}", flush=True)
        state.update(progress="done",
                     last_message=f"done in {time.time() - t0:.1f}s — "
                                  f"loss {loss_before:.3f} → {loss_after:.3f}")
        print(f"[train:{trigger}] finished: loss {loss_before:.4f} → {loss_after:.4f}",
              flush=True)
        return run_id
    except Exception as exc:  # noqa: BLE001 — record any failure in the DB
        if held:
            try:
                model.MODEL_LOCK.release()
            except RuntimeError:
                pass
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if run_id >= 0:
            db.finish_training_run(run_id, "error", 0, 0, 0.0, 0.0, err)
        state.update(progress="error", last_message=err)
        raise
    finally:
        state["running"] = False
        _TRAIN_GATE.release()


def start_background(steps: int, trigger: str = "auto",
                     finetune: bool = False, finetune_text: str = "") -> bool:
    """Kick off a training run in a daemon thread. Returns False if busy."""
    if is_running():
        return False

    def _runner():
        try:
            run_training(steps, trigger, finetune=finetune,
                         finetune_text=finetune_text)
        except Exception as exc:  # already recorded in DB
            print(f"[train:{trigger}] background run failed: {exc}", flush=True)

    threading.Thread(target=_runner, name=f"trainer-{trigger}", daemon=True).start()
    return True


def start_finetune() -> bool:
    """Fine-tune on curated (upvoted + remembered-note) samples only."""
    if is_running():
        return False
    curated = [s for s in db.get_training_samples(exclude_negative=True)
               if s.get("quality", 0) >= config.FINETUNE_MIN_QUALITY]
    if not curated:
        return False
    parts = [f"User: {s['prompt']}\nBot: {s['response']}\n\n" for s in curated]
    # repeat the small curated set so the pass has enough steps of signal
    text = "".join(parts) * max(1, config.FINETUNE_STEPS // 10)
    return start_background(config.FINETUNE_STEPS, trigger="finetune",
                            finetune=True, finetune_text=text)


# --------------------------------------------------------------------------- #
# CLI:  python -m app.train --steps 600
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Train the NeuroChat model")
    ap.add_argument("--steps", type=int, default=config.INITIAL_TRAIN_STEPS)
    ap.add_argument("--trigger", type=str, default="manual")
    args = ap.parse_args()
    db.init_db()
    run_training(args.steps, args.trigger)


if __name__ == "__main__":
    _cli()
