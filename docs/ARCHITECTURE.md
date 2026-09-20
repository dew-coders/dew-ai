# Dew AI — System Architecture

Technical reference for how Dew AI is built. It follows the same *shape* as
official AI systems (ChatGPT / Gemini): a client, an API gateway, an intent
router, tool use, retrieval over a long-term memory, a decoder with
guardrails, a data engine, and a training/fine-tuning loop with versioned
checkpoints.

```
┌────────────────────────────────────────────────────────────────────────┐
│                             CLIENT (/frontend)                         │
│  login · streaming chat · mic (STT) / speaker (TTS) · image rendering  │
│  copy / regenerate / stop · 👍👎 feedback · suggestion chips · i18n UI  │
└───────────────▲────────────────────────────────────────────────────────┘
                │ WebSocket /ws (token auth)  +  REST /api/*
┌───────────────┴────────────────────────────────────────────────────────┐
│                         API GATEWAY (app/main.py)                      │
│  username-only login · sessions · rate-safe WS loop · ownership checks │
└───────────────▲────────────────────────────────────────────────────────┘
                │ chat_stream()
┌───────────────┴────────────────────────────────────────────────────────┐
│                        PIPELINE (per message)                          │
│                                                                        │
│  1. save user msg (SQLite, instant)  ──► outbox ──► Supabase (async)   │
│  2. long-term memory extraction      (app/memory.py)                   │
│  3. SAFETY screening                 (app/safety.py)                   │
│  4. SYSTEM PROMPT assembly           (app/system_prompt.py)            │
│  5. INTENT ROUTING                   (app/router.py)                   │
│        chitchat ─► localized templates (app/languages.py, 14 langs)    │
│        tool     ─► calculator / datetime / units (app/tools.py)        │
│        image    ─► image generation (app/imagegen.py)                  │
│        search   ─► DuckDuckGo + TF-IDF extractive answer (app/search)  │
│        knowledge┘─► TF-IDF over dataset memory (app/knowledge.py)      │
│        neural   ─► char transformer decoder (app/engine.py → model.py) │
│  6. OUTPUT SAFETY filter + humanizer                                   │
│  7. persist reply, record training sample, auto-retrain trigger        │
└────────────────────────────────────────────────────────────────────────┘

DATA ENGINE                          TRAINING ENGINE
app/database.py (SQLite+WAL)         app/train.py (+ fine-tune mode)
app/sync.py (Supabase outbox)        app/pipeline.py (collect → trigger)
app/datacol.py (JSONL export)        app/checkpoints.py (versions/rollback)
app/datasets.py (registry+heal)      app/model.py (NumPy transformer+Adam)
```

## Request lifecycle (neural route)

1. **Ingest** — the user message is saved to SQLite immediately (fast chat
   save) and queued in `sync_outbox` for the Supabase worker.
2. **Memory** — `memory.process_message()` extracts self-describing facts
   ("my name is X", Sinhala variants) into `user_memory`; recalled facts are
   injected into the system prompt block.
3. **Safety** — `safety.screen_input()` intercepts harmful intent (refusal)
   and crisis language (compassionate reply) before any generation.
4. **System prompt** — `system_prompt.build()` assembles persona + user
   memory + language directive + route-specific capability line.
5. **Route** — `router.classify()` picks a handler. Deterministic routes
   (tools, image, search, chitchat) answer instantly; ambiguous text falls
   through to the neural net.
6. **Decode** — the engine streams tokens (char-level) under the model lock
   with temperature / top-k / top-p **and a repetition penalty** over the
   last 64 produced tokens. The client can stop mid-stream (`stop` event →
   `threading.Event` checked per delta).
7. **Post** — humanizer polish, `safety.screen_output()` filter, save reply,
   record the exchange as a training sample, and maybe fire the background
   retrain when enough new samples accumulated.

## Continuous learning

- Every exchange lands in `training_samples` with `source` and `lang`.
- 👍/👎 feedback propagates: downvoted excluded, upvoted ×2 weight, and
  upvoted samples form the curated set for low-LR **fine-tuning**.
- `RETRAIN_THRESHOLD` new samples → automatic background run; weights save
  atomically and the running engine **hot-reloads** on mtime change.
- Each completed run is snapshotted into `data/checkpoints/` (weights +
  vocab + meta). Rollback restores an older version and hot-reloads it.

## Long-term data guarantees

- **Dataset registry** (`datasets.py`): every registered dataset is
  downloaded once into `data/datasets/` with a SHA-256 manifest; on boot
  anything missing/corrupted is re-downloaded (self-healing) and the
  knowledge index is rebuilt.
- **Cloud mirror** (Supabase): users, conversations, messages, memory,
  training runs and dataset registry rows are pushed through an idempotent
  outbox (upsert on UUID); `POST /api/recover` pulls chats back on any device.
- **Exports** (`datacol.py`): the collected corpus can be exported as JSONL
  (`POST /api/data/export`) — the same file format official fine-tuning
  pipelines consume.

## Design constraints (on purpose)

- **No ML frameworks** — the transformer, backprop, Adam, tokenizer, TF-IDF
  and the sampling stack are hand-written NumPy/stdlib. The point is a real,
  inspectable AI system, not a wrapper around a hosted LLM.
- **Fast chat saves** — SQLite WAL + an async outbox; the user message is
  durable before generation starts.
- **Deterministic-first routing** — tools and search answer factual asks
  exactly; the tiny neural net is reserved for creative/conversational text
  where a 2.7M-param model is honest about what it knows.
