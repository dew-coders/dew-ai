"""Central configuration: paths, model hyper-parameters, training settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"

DB_PATH = DATA_DIR / "chat.db"
WEIGHTS_PATH = DATA_DIR / "model_weights.npz"
VOCAB_PATH = DATA_DIR / "vocab.json"
SEED_CORPUS_PATH = DATA_DIR / "seed_corpus.txt"
TRAIN_TEXT_PATH = DATA_DIR / "train.txt"

# Make sure the data directory exists (weights/db/datasets live here).
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Model architecture --------------------------
CONTEXT_LEN = 192      # characters of context the model can see
D_MODEL = 128          # embedding / residual width
N_LAYERS = 3           # transformer blocks
N_HEADS = 4            # attention heads per block
D_FF = 512             # feed-forward hidden width

# ----------------------------- Training ------------------------------------
LEARNING_RATE = 3e-3
BATCH_SIZE = 16
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)
LR_WARMUP_STEPS = 20

INITIAL_TRAIN_STEPS = 600   # bootstrap training run on the seed corpus
AUTO_TRAIN_STEPS = 300      # steps for automatic continuous-learning runs
RETRAIN_THRESHOLD = 15      # new collected samples before auto-retrain fires

# ----------------------------- Generation ----------------------------------
TEMPERATURE = 0.85
TOP_K = 40
TOP_P = 0.92
MAX_NEW_TOKENS = 220
LOW_CONFIDENCE_THRESHOLD = 0.30   # below this the bot prefers search/templates

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
