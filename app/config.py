"""Central configuration: paths, model hyper-parameters, training settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Tiny .env loader (no dependency): KEY=VALUE lines, existing env wins."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"

DB_PATH = DATA_DIR / "chat.db"
WEIGHTS_PATH = DATA_DIR / "model_weights.npz"
VOCAB_PATH = DATA_DIR / "vocab.json"
SEED_CORPUS_PATH = DATA_DIR / "seed_corpus.txt"
SINHALA_CORPUS_PATH = DATA_DIR / "sinhala_corpus.txt"
TRAIN_TEXT_PATH = DATA_DIR / "train.txt"

# Model version registry (official-AI-style checkpointing)
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_KEEP = int(os.environ.get("CHECKPOINT_KEEP", "8"))  # versions kept on disk

# Exported training data (JSONL) lands here
EXPORTS_DIR = DATA_DIR / "exports"

# Make sure the data directory exists (weights/db/datasets live here).
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Model architecture --------------------------
# Scaled-up GPT-style configuration (~2.7M params): wider residual stream,
# more layers and a longer context so the model can absorb the dataset memory.
CONTEXT_LEN = 256      # characters of context the model can see
D_MODEL = 256          # embedding / residual width
N_LAYERS = 4           # transformer blocks
N_HEADS = 8            # attention heads per block
D_FF = 768             # feed-forward hidden width

# ----------------------------- Training ------------------------------------
LEARNING_RATE = 3e-3
BATCH_SIZE = 16
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)
LR_WARMUP_STEPS = 20

INITIAL_TRAIN_STEPS = 600   # bootstrap training run on the seed corpus
AUTO_TRAIN_STEPS = 300      # steps for automatic continuous-learning runs
RETRAIN_THRESHOLD = 15      # new collected samples before auto-retrain fires

# Fine-tuning: gentler LR + fewer steps, only on high-quality curated samples
FINETUNE_STEPS = 200
FINETUNE_LR = 5e-4          # ~6x lower than base LR to avoid forgetting
FINETUNE_MIN_QUALITY = 1    # only upvoted / curated samples

# ----------------------------- Generation ----------------------------------
TEMPERATURE = 0.85
TOP_K = 40
TOP_P = 0.92
MAX_NEW_TOKENS = 220
REPETITION_PENALTY = 1.15          # >1 dampens tokens seen in the last 64 chars
LOW_CONFIDENCE_THRESHOLD = 0.30   # below this the bot prefers search/templates

# ----------------------------- Supabase (cloud database) --------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

# Chat saves go to fast local SQLite first; a background worker syncs them to
# Supabase. Set SYNC_ENABLED=0 to run fully offline.
SYNC_ENABLED = os.environ.get("SYNC_ENABLED", "1").lower() not in ("0", "false", "no")
SYNC_INTERVAL = float(os.environ.get("SYNC_INTERVAL", "1.0"))   # seconds between pushes
SYNC_BATCH = int(os.environ.get("SYNC_BATCH", "50"))            # rows per push
SYNC_MAX_ATTEMPTS = int(os.environ.get("SYNC_MAX_ATTEMPTS", "40"))

# ----------------------------- Languages ------------------------------------
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "en")

# ----------------------------- Server ---------------------------------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
