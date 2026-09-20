# Dew AI 🧠

A full-stack AI chat application with a **neural network built completely from scratch in NumPy** — no PyTorch, no TensorFlow, no API keys. Created by **Hansa Dewmina**. The model is a tiny GPT-style causal transformer with hand-written forward/backward passes, trained via a **data-collection pipeline that continuously learns from every conversation** stored in SQLite and synced to Supabase. A search mechanism fetches external information when a question needs fresh facts, and a real-time chat UI streams answers token-by-token over a WebSocket.

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

### Long-term dataset memory (`app/datasets.py`, `app/knowledge.py`)
Dew AI never forgets its training data. Every dataset is **registered** in `app/datasets.py` (`REGISTRY`) and downloaded once into `data/datasets/` with a **manifest** (row counts + SHA-256 hashes). On every boot the registry is verified and anything missing/corrupted is **automatically re-downloaded**; `train.py` always rebuilds its corpus from the registry, so a fresh re-train (even after deleting the weights) automatically includes every dataset:

| Dataset | Source | Rows |
|---|---|---|
| `sinhala-asr-1000` | [Ransaka/SinhalaASR-1000](https://huggingface.co/datasets/Ransaka/SinhalaASR-1000) transcriptions | 1,000 |
| `sinhala-news-24k` | [NLPC-UOM/Sinhala-News-Source-classification](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification) headlines | 1,000 (of 24k available) |

Because the neural net is intentionally tiny, **retrieval** makes the datasets useful in chat: `app/knowledge.py` indexes every stored headline with a from-scratch TF-IDF engine and knowledge questions are answered from dataset memory (answer source: `📚 dataset memory`, with per-dataset source chips). Registry status is mirrored to the Supabase `registered_datasets` table, and the REST API exposes `GET /api/datasets` + `POST /api/datasets/{name}/refresh`.

### Search mechanism (`app/search.py`)
Factual or current-events questions ("who is…", "what's the latest…", or an explicit `search: …`) trigger a DuckDuckGo Lite lookup. Results are ranked with a small from-scratch TF-IDF cosine scorer and composed into an extractive answer with source links. If the neural model's confidence is low, search is used as a backup.

### Languages (`app/languages.py`)
Dew AI speaks **14 languages**: English, සිංහල (Sinhala), Español, Français, Deutsch, Italiano, Português, Русский, Türkçe, العربية, हिन्दी, 中文, 日本語 and 한국어. Detection combines Unicode-script analysis with stop-word scoring; greetings, identity, creator, thanks and goodbye are matched in every language, and canned replies are localized (Sinhala detection powers `lang` metadata on every stored message). The neural net itself reads characters, so its vocabulary already covers the Sinhala script — the [SinhalaASR-1000](https://huggingface.co/datasets/Ransaka/SinhalaASR-1000) transcriptions are baked into the training corpus (audio isn't used by the text model; run `python -m app.import_dataset` to refresh it).

### Data collection & continuous learning (`app/database.py`, `app/pipeline.py`, `app/train.py`)
1. Every exchange (prompt, reply, source) and every 👍/👎 is stored in SQLite.
2. Downvoted answers are excluded from training; upvoted ones get extra weight.
3. When `RETRAIN_THRESHOLD` (15) new samples accumulate — or you press **Train now** — a background thread fine-tunes the model.
4. Weights are saved atomically; the running server **hot-reloads** them, so the bot visibly improves without restarts.
5. Every training run (loss before → after) is logged in the database.

### Model versions & rollback (`app/checkpoints.py`)
Every training run snapshots its weights + tokenizer into `data/checkpoints/v<time>_<run>_<trigger>/` with metadata (trigger, loss curve, sample count). The newest `CHECKPOINT_KEEP` versions are kept; older ones are pruned automatically. Roll back anytime with `POST /api/model/rollback {"id": "v…"}` — the live engine hot-reloads the restored weights.

### System prompt & safety (`app/system_prompt.py`, `app/safety.py`)
Like official AI systems, every request is wrapped in a **system prompt** assembled from the core persona (identity, capabilities), the user's long-term memory and the active language. A **safety layer** screens both sides: harmful requests get a refusal, crisis language gets a compassionate response with help resources, and every outgoing reply passes an output filter. Inspect the assembled prompt anytime via `GET /api/system-prompt`.

### Dataset builder & export (`app/datacol.py`)
The data-collection system turns real chats into reusable training sets: deduplicated (prompt, response) pairs with source/language/quality metadata, upvoted samples weighted ×2, downvoted excluded. Export as JSONL (official-style dataset files) via `POST /api/data/export` and download via `GET /api/data/export/latest`.

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
| POST | `/api/finetune` | Curated low-LR fine-tune on upvoted samples |
| POST | `/api/chat/{id}/stop` | Stop an in-flight generation |
| POST | `/api/chat/{id}/regenerate` | Re-answer the last message |
| GET  | `/api/model/versions` | List saved model checkpoints |
| POST | `/api/model/rollback` | Restore a previous model version |
| GET  | `/api/system-prompt` | Inspect the assembled system prompt |
| GET  | `/api/data/stats` | Data-collection overview |
| POST | `/api/data/export` | Export collected chats as JSONL dataset |
| GET  | `/api/data/export/latest` | Download the latest dataset export |
| GET  | `/api/memory` · POST · DELETE | Long-term user memory (ChatGPT-style) |
| GET  | `/api/health` | Health + model readiness |

## Project layout

```
app/
  config.py      # paths + hyper-parameters
  tokenizer.py   # char-level tokenizer (from scratch)
  model.py       # NumPy transformer, backprop, Adam
  generate.py    # sampling + streaming generation (rep-penalty)
  engine.py      # model manager with hot-reload
  router.py      # intent router (chitchat/tool/image/search/knowledge/neural)
  tools.py       # task execution: math, date/time, unit conversion
  imagegen.py    # image generation (free API + offline SVG fallback)
  memory.py      # long-term user memory (extract/store/recall)
  search.py      # DuckDuckGo search + TF-IDF answer composition
  safety.py      # input screening + output filter + crisis response
  system_prompt.py # official-style system-prompt assembly
  checkpoints.py # model version registry + rollback
  datacol.py     # dataset builder + JSONL export
  datasets.py    # dataset registry (auto-healing long-term memory)
  knowledge.py   # TF-IDF retrieval over stored datasets
  humanize.py    # response post-processing
  languages.py   # 14-language detection + localized replies
  database.py    # SQLite layer + outbox queue
  pipeline.py    # data collection / retrain triggers
  train.py       # training + fine-tuning loop (CLI + background)
  sync.py        # Supabase cloud sync + chat recovery
  main.py        # FastAPI server (REST + WebSocket)
frontend/
  index.html / styles.css / app.js
docs/
  ARCHITECTURE.md  # request lifecycle + design notes
data/
  seed_corpus.txt  # committed seed training data
  chat.db, model_weights.npz, checkpoints/, …  # generated (gitignored)
```

## Notes & tuning

All hyper-parameters live in `app/config.py` — model size (`D_MODEL`, `N_LAYERS`, …), training (`LEARNING_RATE`, `BATCH_SIZE`), generation (`TEMPERATURE`, `TOP_K`, `TOP_P`) and the continuous-learning thresholds. The model is intentionally tiny: it's a real, trainable neural network you can inspect end-to-end, not a wrapper around a hosted LLM.
