"""NeuroChat server — FastAPI app wiring everything together.

- POST /api/chat            : one-shot chat (non-streaming fallback)
- WS   /ws                  : real-time streaming chat (token-by-token)
- GET  /api/stats           : DB + model + training statistics
- GET  /api/conversations   : conversation list
- GET  /api/history/{id}    : message history
- POST /api/feedback        : 👍/👎 per message (feeds the learning loop)
- POST /api/train           : manually trigger a fine-tuning run
- GET  /                    : the chat UI (served from /frontend)
"""
from __future__ import annotations

import re
import zlib
from contextlib import asynccontextmanager
from typing import Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from starlette.concurrency import iterate_in_threadpool

from app import config, database as db, engine, humanize, pipeline, search, train


# --------------------------------------------------------------------------- #
# App lifespan: init DB, auto-train the model from scratch on first boot
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    engine.engine.refresh()
    if not engine.engine.ready:
        print("[boot] no weights found — training the model from scratch in the "
              "background (the chat will use templates until it's ready)…", flush=True)
        train.start_background(config.INITIAL_TRAIN_STEPS, trigger="bootstrap")
    yield


app = FastAPI(title="NeuroChat", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Template / fallback responses (used while training or for simple chit-chat)
# --------------------------------------------------------------------------- #
GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo|sup|hola|good\s+(morning|afternoon|evening))\b", re.I)
IDENTITY_RE = re.compile(r"\b(who are you|what are you|your name|about yourself)\b", re.I)
HELP_RE = re.compile(r"\b(help|what can you do|capabilit|how do you work)\b", re.I)
THANKS_RE = re.compile(r"^\s*(thanks|thank you|thx|ty)\b", re.I)
BYE_RE = re.compile(r"^\s*(bye|goodbye|see you|cya|good night)\b", re.I)

TEMPLATES = {
    "identity": [
        "I'm NeuroChat — a tiny neural network built completely from scratch in NumPy, and I get a little smarter every time we talk.",
        "I'm a from-scratch transformer (no PyTorch, promise) trained on our own conversations. Think of me as a very small, very eager brain.",
    ],
    "greeting": [
        "Hey there! Great to see you. What's on your mind?",
        "Hello! Ask me anything — and if I don't know, say 'search: …' and I'll look it up online.",
        "Hi! I'm all ears. How's your day going?",
    ],
    "help": [
        "I can chat, remember our conversations, search the web for fresh facts (just say 'search for …'), and I retrain myself on everything we discuss. Try me!",
        "Ask me questions, or prefix with 'search:' to fetch live info from the web. Everything you teach me goes into my training data.",
    ],
    "thanks": [
        "Anytime! That's what I'm here for.",
        "You're welcome! Come back with more questions.",
    ],
    "bye": [
        "See you soon! I'll keep learning while you're away.",
        "Bye! Every chat we have makes me better.",
    ],
    "fallback": [
        "I'm still training my neural network on our conversations, so I'm not sure about that one yet. Try again in a minute, or say 'search: …' and I'll look it up online.",
        "Hmm, my tiny brain doesn't have that figured out yet — it's literally learning as we speak. Ask me again shortly!",
    ],
}


def _pick(options: List[str], seed: str) -> str:
    return options[zlib.crc32(seed.encode()) % len(options)]


def template_reply(text: str, kind: str = "fallback") -> str:
    return _pick(TEMPLATES[kind], text)


def detect_template(text: str) -> Optional[str]:
    if IDENTITY_RE.search(text):
        return "identity"
    if GREETING_RE.match(text) and len(text) < 60:
        return "greeting"
    if HELP_RE.search(text):
        return "help"
    if THANKS_RE.match(text) and len(text) < 40:
        return "thanks"
    if BYE_RE.match(text) and len(text) < 40:
        return "bye"
    return None


# --------------------------------------------------------------------------- #
# Prompt building & chat pipeline
# --------------------------------------------------------------------------- #
def build_prompt(history: List[Dict], user_text: str) -> str:
    lines: List[str] = []
    for m in history[-6:]:
        content = m["content"].replace("\n", " ").strip()[:200]
        who = "User" if m["role"] == "user" else "Bot"
        lines.append(f"{who}: {content}")
    lines.append(f"User: {user_text.strip()[:300]}")
    lines.append("Bot:")
    prompt = "\n".join(lines)
    if len(prompt) > 800:  # keep inside a sane window for the tiny model
        prompt = "\n".join(prompt.split("\n")[-6:])
    return prompt


def chat_stream(conversation_id: int, text: str) -> Iterator[Dict]:
    """Shared chat pipeline. Yields events consumed by WS and REST alike:
    status / notice / delta / done."""
    user_msg_id = db.add_message(conversation_id, "user", text)

    yield {"type": "status", "stage": "thinking"}

    mode, reply, conf, sources, searched = "fallback", "", 0.0, [], False

    # 1) Simple chit-chat gets an instant humanized template answer.
    tpl_kind = detect_template(text)
    if tpl_kind:
        reply = template_reply(text, tpl_kind)
        mode, conf = "template", 0.85
        yield {"type": "delta", "text": reply}

    # 2) Factual / current-events questions go to the web-search mechanism.
    if not reply:
        query = search.needs_search(text)
        if query:
            yield {"type": "status", "stage": "searching"}
            try:
                results = search.web_search(query)
                if results:
                    answer, hosts = search.compose_answer(query, results)
                    if answer:
                        db.add_search_log(query, results)
                        mode, reply, conf, searched = "search", answer, 0.95, True
                        sources = [
                            {"title": r["title"], "url": r["url"]} for r in results[:3]
                        ]
                        yield {"type": "delta", "text": reply}
            except Exception as exc:  # network down, rate limited, etc.
                yield {"type": "notice",
                       "text": f"Web search unavailable ({exc.__class__.__name__}) — answering from my own head."}

    # 3) Everything else goes through the neural network (streamed).
    if not reply:
        history = db.get_recent_messages(conversation_id, 6)
        prompt = build_prompt(history, text)
        gen = engine.engine.stream(prompt)
        if gen is None:
            reply = template_reply(text)
            yield {"type": "delta", "text": reply}
        else:
            yield {"type": "status", "stage": "generating"}
            parts: List[str] = []
            final: Optional[Dict] = None
            for event in gen:
                if event["type"] == "delta":
                    parts.append(event["text"])
                    yield event
                else:
                    final = event
            raw = (final or {}).get("reply") or "".join(parts)
            conf = float((final or {}).get("confidence", 0.0))
            if raw:
                mode, reply = "neural", raw
            else:
                reply = template_reply(text)

    # 4) Humanize, persist, and feed the continuous-learning loop.
    final_text = humanize.humanize(reply, allow_softener=(mode == "neural"))
    msg_id = db.add_message(conversation_id, "assistant", final_text, mode, conf)
    if mode in ("neural", "search", "template"):
        pipeline.record_exchange(text, final_text, mode)
    pending = pipeline.maybe_start_retrain()
    if pending:
        yield {"type": "notice",
               "text": f"🧠 {pending} new exchanges collected — fine-tuning started in the background."}

    yield {"type": "done", "conversation_id": conversation_id,
           "user_message_id": user_msg_id, "message_id": msg_id,
           "reply": final_text, "source": mode,
           "confidence": round(conf, 3), "searched": searched,
           "sources": sources}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatIn(BaseModel):
    text: str
    conversation_id: Optional[int] = None


class FeedbackIn(BaseModel):
    message_id: int
    rating: int
    comment: str = ""

    @field_validator("rating")
    @classmethod
    def rating_valid(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        return v


class TrainIn(BaseModel):
    steps: Optional[int] = None


def build_stats() -> Dict:
    return {
        "database": db.stats(),
        "model": engine.engine.info(),
        "training": train.status(),
        "recent_runs": db.list_training_runs(5),
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "feedback_policy": pipeline.retrain_feedback_quality(),
    }


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": engine.engine.ready}


@app.get("/api/stats")
def stats():
    return build_stats()


@app.get("/api/model")
def model_info():
    return engine.engine.info()


@app.get("/api/conversations")
def conversations():
    return db.list_conversations()


@app.get("/api/history/{conversation_id}")
def history(conversation_id: int):
    if conversation_id <= 0:
        raise HTTPException(400, "invalid conversation id")
    return db.get_history(conversation_id)


@app.post("/api/chat")
def chat(body: ChatIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    conv_id = body.conversation_id or db.create_conversation(title=text[:60])
    done: Dict = {}
    for event in chat_stream(conv_id, text):
        if event["type"] == "done":
            done = event
    return {"conversation_id": conv_id, **done}


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    ok = db.add_feedback(body.message_id, body.rating, body.comment)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"recorded": True}


@app.post("/api/train")
def start_training(body: TrainIn):
    if train.is_running():
        return JSONResponse({"started": False, "detail": "training already in progress"},
                            status_code=409)
    steps = max(50, min(int(body.steps or config.AUTO_TRAIN_STEPS), 5000))
    ok = train.start_background(steps, trigger="manual")
    return {"started": ok, "steps": steps}


# --------------------------------------------------------------------------- #
# Real-time WebSocket chat
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "hello", "model": engine.engine.info(),
                        "training": train.status()})
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                break
            mtype = msg.get("type")
            if mtype == "ping":
                await ws.send_json({"type": "pong"})
                continue
            if mtype == "stats":
                await ws.send_json({"type": "stats", "data": build_stats()})
                continue
            if mtype != "chat":
                continue
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            conv_id = msg.get("conversation_id")
            if not conv_id:
                conv_id = db.create_conversation(title=text[:60])
            async for event in iterate_in_threadpool(chat_stream(conv_id, text)):
                await ws.send_json(event)
    except Exception as exc:  # keep the socket protocol friendly on errors
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Frontend (served from /frontend) — mounted last so /api routes win
# --------------------------------------------------------------------------- #
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True),
          name="frontend")
