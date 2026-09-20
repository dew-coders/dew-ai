"""NeuroChat server — FastAPI app wiring everything together.

Auth (username-only login):
- POST /api/login          : {username} → session token (creates user if new)
- POST /api/logout         : invalidate the session
- GET  /api/me             : logged-in user details + usage stats
- POST /api/recover        : pull this user's chats back from Supabase

Chat & data:
- WS   /ws                 : real-time streaming chat (token via query string)
- POST /api/chat            : one-shot chat (non-streaming fallback)
- GET  /api/conversations   : the user's conversation list
- GET  /api/history/{id}    : message history (owner only)
- POST /api/feedback        : 👍/👎 per message (feeds the learning loop)
- POST /api/train           : manually trigger a fine-tuning run
- GET  /api/stats           : DB + model + training + cloud-sync statistics
- GET  /                    : the chat UI (served from /frontend)

Every chat message is saved to local SQLite instantly (fast chat save) and
queued to Supabase in the background (cloud database + chat recovery), with
per-message language detection driving a multi-language reply system.
"""
from __future__ import annotations

import re
import zlib
from contextlib import asynccontextmanager
from typing import Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from starlette.concurrency import iterate_in_threadpool

from app import config, database as db, engine, humanize, languages, pipeline, search, sync, train


# --------------------------------------------------------------------------- #
# App lifespan: init DB, start cloud sync, auto-train on first boot
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    engine.engine.refresh()
    sync.start()
    if not engine.engine.ready:
        print("[boot] no weights found — training the model from scratch in the "
              "background (the chat will use templates until it's ready)…", flush=True)
        train.start_background(config.INITIAL_TRAIN_STEPS, trigger="bootstrap")
    yield


app = FastAPI(title="NeuroChat", version="0.2.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Auth helpers (username-only login; token via Authorization header / ?token=)
# --------------------------------------------------------------------------- #
def _token_from_request(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.query_params.get("token") or "").strip()


def current_user(request: Request) -> Optional[Dict]:
    return db.get_session_user(_token_from_request(request))


def require_user(request: Request) -> Dict:
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "login required (POST /api/login with a username)")
    return user


def _user_from_ws_token(token: str) -> Optional[Dict]:
    return db.get_session_user((token or "").strip())


# --------------------------------------------------------------------------- #
# Prompt building & chat pipeline (multi-language aware)
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


def chat_stream(conversation_id: str, text: str) -> Iterator[Dict]:
    """Shared chat pipeline. Yields events consumed by WS and REST alike:
    status / notice / delta / done."""
    lang = languages.detect_language(text)
    user_msg_id = db.add_message(conversation_id, "user", text, lang=lang)

    yield {"type": "status", "stage": "thinking"}

    mode, reply, conf, sources, searched = "fallback", "", 0.0, [], False

    # 1) Simple chit-chat gets an instant localized template answer.
    tpl_kind = languages.template_kind(text)
    if tpl_kind:
        reply = languages.template_reply(text, tpl_kind, lang)
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

    # 3) Everything else goes through the neural network (streamed). The tiny
    #    char-level model is English-only, so non-English gets a localized
    #    template answer instead of gibberish.
    if not reply:
        history = db.get_recent_messages(conversation_id, 6)
        prompt = build_prompt(history, text)
        gen = engine.engine.stream(prompt)
        if gen is None:
            reply = languages.template_reply(text, "fallback", lang)
            mode = "template"
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
            if raw and lang == "en":
                mode, reply = "neural", raw
            else:
                mode = "template"
                reply = languages.template_reply(text, "fallback", lang)

    # 4) Humanize, persist (instant local save + cloud queue), and feed the
    #    continuous-learning loop.
    final_text = humanize.humanize(reply, allow_softener=(mode == "neural"), lang=lang)
    msg_id = db.add_message(conversation_id, "assistant", final_text, mode, conf, lang=lang)
    if mode in ("neural", "search", "template"):
        pipeline.record_exchange(text, final_text, mode, lang=lang)
    pending = pipeline.maybe_start_retrain()
    if pending:
        yield {"type": "notice",
               "text": f"🧠 {pending} new exchanges collected — fine-tuning started in the background."}

    yield {"type": "done", "conversation_id": conversation_id,
           "user_message_id": user_msg_id, "message_id": msg_id,
           "reply": final_text, "source": mode,
           "confidence": round(conf, 3), "searched": searched,
           "sources": sources, "lang": lang}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class LoginIn(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"[\w .\-]{2,24}", v):
            raise ValueError("username must be 2-24 characters (letters, digits, space, . _ -)")
        return v


class ChatIn(BaseModel):
    text: str
    conversation_id: Optional[str] = None


class SettingsIn(BaseModel):
    lang: str

    @field_validator("lang")
    @classmethod
    def lang_supported(cls, v: str) -> str:
        if v not in languages.LANGS:
            raise ValueError(f"unsupported language (use one of {', '.join(languages.LANGS)})")
        return v


class FeedbackIn(BaseModel):
    message_id: str
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
        "cloud_sync": sync.status(),
    }


def _owned_conversation(user: Dict, conversation_id: Optional[str],
                        title: str) -> str:
    """Return a conversation id owned by the user, creating one if needed."""
    if conversation_id:
        conv = db.get_conversation(conversation_id)
        if conv and conv.get("user_id") == user["id"]:
            return conversation_id
        if conv:  # exists but belongs to someone else → start a fresh one
            raise HTTPException(403, "this conversation belongs to another user")
    return db.create_conversation(user["id"], title=title)


# --------------------------------------------------------------------------- #
# Auth API
# --------------------------------------------------------------------------- #
@app.post("/api/login")
def login(body: LoginIn):
    user, token, created = db.login_or_register(body.username)
    return {"token": token, "created": created,
            "user": _user_details(user["id"])}


def _user_details(user_id: str) -> Dict:
    user = db.get_user(user_id) or {}
    convs = db.list_conversations(user_id=user_id, limit=500)
    msg_count = sum(c["msg_count"] for c in convs)
    return {"id": user.get("id"), "username": user.get("username"),
            "lang": user.get("lang", "en"), "created_at": user.get("created_at"),
            "conversations": len(convs), "messages": msg_count}


@app.post("/api/logout")
def logout(request: Request):
    db.logout(_token_from_request(request))
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    user = require_user(request)
    return _user_details(user["id"])


@app.post("/api/settings")
def settings(body: SettingsIn, request: Request):
    """Persist the user's preferred interface/chat language."""
    user = require_user(request)
    db.update_user_lang(user["id"], body.lang)
    return {"ok": True, "lang": body.lang}


# --------------------------------------------------------------------------- #
# Chat API (user-scoped)
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
def conversations(request: Request):
    user = require_user(request)
    return db.list_conversations(user_id=user["id"])


@app.get("/api/history/{conversation_id}")
def history(conversation_id: str, request: Request):
    user = require_user(request)
    conv = db.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    if conv.get("user_id") != user["id"]:
        raise HTTPException(403, "this conversation belongs to another user")
    return db.get_history(conversation_id)


@app.post("/api/recover")
def recover(request: Request):
    """Chat recovery: pull this user's conversations + messages from Supabase."""
    user = require_user(request)
    result = sync.recover_from_cloud(user)
    result["conversations_list"] = db.list_conversations(user_id=user["id"])
    return result


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    user = require_user(request)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    conv_id = _owned_conversation(user, body.conversation_id, title=text[:60])
    done: Dict = {}
    for event in chat_stream(conv_id, text):
        if event["type"] == "done":
            done = event
    return {"conversation_id": conv_id, **done}


@app.post("/api/feedback")
def feedback(body: FeedbackIn, request: Request):
    user = require_user(request)
    ok = db.add_feedback(body.message_id, body.rating, body.comment)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"recorded": True}


@app.post("/api/train")
def start_training(body: TrainIn, request: Request):
    require_user(request)
    if train.is_running():
        return JSONResponse({"started": False, "detail": "training already in progress"},
                            status_code=409)
    steps = max(50, min(int(body.steps or config.AUTO_TRAIN_STEPS), 5000))
    ok = train.start_background(steps, trigger="manual")
    return {"started": ok, "steps": steps}


# --------------------------------------------------------------------------- #
# Real-time WebSocket chat (token via ?token=… or an auth message)
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    user = _user_from_ws_token(ws.query_params.get("token") or "")
    if not user:
        await ws.send_json({"type": "auth_required"})
        # give the client a moment to send {"type": "auth", "token": …}
        try:
            first = await ws.receive_json()
        except Exception:
            await ws.close(code=4401)
            return
        if isinstance(first, dict) and first.get("type") == "auth":
            user = _user_from_ws_token(str(first.get("token") or ""))
        if not user:
            await ws.send_json({"type": "error", "message": "invalid session token"})
            await ws.close(code=4401)
            return

    await ws.send_json({"type": "hello", "model": engine.engine.info(),
                        "training": train.status(),
                        "user": {"id": user["id"], "username": user["username"]}})
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
            if mtype == "recover":
                result = sync.recover_from_cloud(user)
                await ws.send_json({"type": "notice",
                                    "text": f"☁ recovered {result.get('conversations', 0)} chats / "
                                            f"{result.get('messages', 0)} messages from the cloud"})
                continue
            if mtype != "chat":
                continue
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            try:
                conv_id = _owned_conversation(user, msg.get("conversation_id"),
                                              title=text[:60])
            except HTTPException as exc:
                await ws.send_json({"type": "error", "message": exc.detail})
                continue
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
