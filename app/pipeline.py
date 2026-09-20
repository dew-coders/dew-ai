"""Data collection & continuous-learning pipeline.

Every exchange (prompt + reply + its source + user feedback) flows into the
database. When enough new samples accumulate, this pipeline automatically
launches a background fine-tuning run — that's how the model improves from
real conversations.
"""
from __future__ import annotations

from typing import Dict, Optional

from app import config, database as db, train


def record_exchange(prompt: str, response: str, source: str) -> None:
    """Store one user/assistant exchange as a future training sample."""
    if prompt and response:
        db.add_training_sample(prompt=prompt.strip(), response=response.strip(),
                               source=source)


def maybe_start_retrain(force: bool = False) -> Optional[int]:
    """Launch a background fine-tune when there is enough new data (or forced)."""
    if train.is_running():
        return None
    pending = db.count_new_samples()
    if not force and pending < config.RETRAIN_THRESHOLD:
        return None
    ok = train.start_background(steps=config.AUTO_TRAIN_STEPS, trigger="auto")
    return pending if ok else None


def retrain_feedback_quality() -> Dict:
    """Summary of how user feedback influences the dataset."""
    s = db.stats()
    return {
        "upvoted": s["feedback_up"],
        "downvoted": s["feedback_down"],
        "downvoted_samples_excluded_from_training": True,
        "upvoted_samples_weighted_x2": True,
    }
