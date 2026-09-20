# NeuroChat 🧠

A full-stack AI chat application with a **neural network built completely from scratch in NumPy** — no PyTorch, no TensorFlow, no API keys. The model is a tiny GPT-style causal transformer with hand-written forward/backward passes, trained via a **data-collection pipeline that continuously learns from every conversation** stored in SQLite. A search mechanism fetches external information when a question needs fresh facts, and a real-time chat UI streams answers token-by-token over a WebSocket.

## Architecture

```
┌──────────────┐   WebSocket (real-time)   ┌───────────────────────────┐
│  /frontend   │◄─────────────────────────►│  FastAPI server (app/)    │
│  chat UI     │        REST /api/*        │                           │
└──────────────┘                           │  chat pipeline            │
                                           │   ├─ templates            │
                                           │   ├─ 🌐 web search        │──► DuckDuckGo
                                           │   ├─ 🧠 neural net        │
                                           │   └─ ✨ humanizer         │
                                           │        │                  │
                                           │  SQLite (data/chat.db)    │
                                           │        │                  │
                                           │  continuous-learning      │
                                           │  trainer (background)     │
                                           └───────────────────────────┘
                                                    │ hot-reload
                                              model_weights.npz
```

### The from-scratch model (`app/model.py`)
- Tiny causal transformer: token + positional embeddings → N pre-norm blocks (multi-head self-attention + GELU MLP) → final layer norm → output head.
- Full **backpropagation** (layer-norm, softmax-attention, GELU gradients) and an **Adam optimizer** implemented by hand.
- Character-level tokenizer (`app/tokenizer.py`) — no external tokenizer libraries.
- ~0.6M parameters; trains in about a minute on CPU.

### Humanized responses
Generation uses temperature / top-k / top-p sampling with repetition guards, and a post-processing layer (`app/humanize.py`) adds contractions, conversational softeners and clean punctuation.

### Search mechanism (`app/search.py`)
Factual or current-events questions ("who is…", "what's the latest…", or an explicit `search: …`) trigger a DuckDuckGo Lite lookup. Results are ranked with a small from-scratch TF-IDF cosine scorer and composed into an extractive answer with source links. If the neural model's confidence is low, search is used as a backup.

### Data collection & continuous learning (`app/database.py`, `app/pipeline.py`, `app/train.py`)
1. Every exchange (prompt, reply, source) and every 👍/👎 is stored in SQLite.
2. Downvoted answers are excluded from training; upvoted ones get extra weight.
3. When `RETRAIN_THRESHOLD` (15) new samples accumulate — or you press **Train now** — a background thread fine-tunes the model.
4. Weights are saved atomically; the running server **hot-reloads** them, so the bot visibly improves without restarts.
5. Every training run (loss before → after) is logged in the database.

## Quickstart

```bash
pip install -r requirements.txt

# Option A: train the model explicitly first (~1 min on CPU)
python -m app.train --steps 600

# Option B: just start the server — it trains automatically on first boot
python run.py
```

Open **http://localhost:8000** and chat. While the first training runs, the bot answers with templates; once weights exist, replies come from the neural net and stream in live.

## API

| Method | Path | Description |
|---|---|---|
| GET  | `/` | Chat UI (served from `/frontend`) |
| WS   | `/ws` | Real-time streaming chat + stats |
| POST | `/api/chat` | One-shot chat (non-streaming) |
| GET  | `/api/stats` | DB / model / training statistics |
| GET  | `/api/conversations` | List conversations |
| GET  | `/api/history/{id}` | Message history |
| POST | `/api/feedback` | 👍/👎 a message (feeds the learning loop) |
| POST | `/api/train` | Manually trigger fine-tuning |
| GET  | `/api/health` | Health + model readiness |

## Project layout

```
app/
  config.py      # paths + hyper-parameters
  tokenizer.py   # char-level tokenizer (from scratch)
  model.py       # NumPy transformer, backprop, Adam
  generate.py    # sampling + streaming generation
  engine.py      # model manager with hot-reload
  search.py      # DuckDuckGo search + TF-IDF answer composition
  humanize.py    # response post-processing
  database.py    # SQLite layer
  pipeline.py    # data collection / retrain triggers
  train.py       # training loop (CLI + background)
  main.py        # FastAPI server (REST + WebSocket)
frontend/
  index.html / styles.css / app.js
data/
  seed_corpus.txt  # committed seed training data
  chat.db, model_weights.npz, …  # generated (gitignored)
```

## Notes & tuning

All hyper-parameters live in `app/config.py` — model size (`D_MODEL`, `N_LAYERS`, …), training (`LEARNING_RATE`, `BATCH_SIZE`), generation (`TEMPERATURE`, `TOP_K`, `TOP_P`) and the continuous-learning thresholds. The model is intentionally tiny: it's a real, trainable neural network you can inspect end-to-end, not a wrapper around a hosted LLM.
