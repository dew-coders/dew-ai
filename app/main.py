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
import threading
import zlib
from contextlib import asynccontextmanager
from typing import Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from starlette.concurrency import iterate_in_threadpool

from app import config, database as db, datacol, datasets, engine, humanize, imagegen, knowledge, languages, memory, pipeline, router, safety, search, sync, system_prompt, tools, train


# --------------------------------------------------------------------------- #
# App lifespan: init DB, start cloud sync, auto-train on first boot
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    engine.engine.refresh()
    sync.start()
    # Long-term dataset memory: verify + auto-heal, then (re)build the
    # knowledge index from whatever the registry holds.
    for info in datasets.ensure_all():
        print(f"[datasets] {info['name']}: {info['status']}", flush=True)
    knowledge.build_index()
    if not engine.engine.ready:
        print("[boot] no weights found — training the model from scratch in the "
              "background (the chat will use templates until it's ready)…", flush=True)
        train.start_background(config.INITIAL_TRAIN_STEPS, trigger="bootstrap")
    yield


app = FastAPI(title="Dew AI", version="0.2.0", lifespan=lifespan)


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
def build_prompt(history: List[Dict], user_text: str,
                 system_block: str = "") -> str:
    """Chat-format prompt: system context + recency window + the new message."""
    lines: List[str] = []
    if system_block:
        lines.append(system_block)
        lines.append("")
    for m in history[-6:]:
        content = m["content"].replace("\n", " ").strip()[:200]
        who = "User" if m["role"] == "user" else "Bot"
        lines.append(f"{who}: {content}")
    lines.append(f"User: {user_text.strip()[:300]}")
    lines.append("Bot:")
    prompt = "\n".join(lines)
    if len(prompt) > 1200:  # keep inside the model's context window
        prompt = "\n".join(prompt.split("\n")[-10:])
    return prompt


# stop-generation support: one Event per active conversation
_ACTIVE_STOPS: Dict[str, threading.Event] = {}

_SUGGESTIONS = {
    "en": ["Tell me more about that", "Search the web for it", "Draw an image of it"],
    "si": ["තව කියන්න", "අන්තර්ජාලයේ සොයන්න", "රූපයක් හදන්න"],
    "es": ["Cuéntame más", "Búscalo en la web", "Dibuja una imagen"],
}


def _suggest_followups(route: str, lang: str) -> List[str]:
    base = _SUGGESTIONS.get(lang, _SUGGESTIONS["en"])
    if route in ("tool", "image"):
        return base[:1] + ["What else can you do?" if lang == "en" else base[0]]
    return base


def chat_stream(conversation_id: str, text: str, user: Dict,
                skip_user_save: bool = False,
                lang_override: Optional[str] = None) -> Iterator[Dict]:
    """Shared chat pipeline (router-driven). Yields events consumed by WS and
    REST alike: status / notice / delta / done."""
    lang = lang_override or languages.detect_language(text)
    user_msg_id = None
    if not skip_user_save:
        user_msg_id = db.add_message(conversation_id, "user", text, lang=lang)
        # auto-title (ChatGPT-style): first user message names the conversation
        conv = db.get_conversation(conversation_id)
        if conv and not (conv.get("title") or "").strip():
            db.update_conversation_title(conversation_id, text.strip()[:60])

    # long-term memory: extract facts from what the user just said
    stored_facts = memory.process_message(user["id"], text)
    if stored_facts:
        yield {"type": "notice", "text": "🧠 Remembered: "
               + ", ".join(f"{f['kind']}: {f['value']}" for f in stored_facts)}

    yield {"type": "status", "stage": "thinking"}

    route, meta = router.classify(text)
    mode, reply, conf, sources, searched = "fallback", "", 0.0, [], False
    image_info: Optional[Dict] = None

    # ------------------------------------------------ 0) safety screening
    flag, safe_reply = safety.screen_input(text)
    if flag:
        route, mode, reply, conf = "safety", "safety", safe_reply, 0.99
        yield {"type": "delta", "text": reply}

    # ------------------------------------------------ 1) chit-chat templates
    if not reply and route == "chitchat":
        reply = languages.template_reply(text, meta["template_kind"], lang)
        mode, conf = "template", 0.85
        yield {"type": "delta", "text": reply}

    # ------------------------------------------------ 2) tools (task execution)
    elif not reply and route == "tool":
        tool_name, tool_arg = meta["tool"], meta["tool_arg"]
        yield {"type": "status", "stage": "tool"}
        reply = tools.run(tool_name, tool_arg)
        mode, conf = "tool", 0.99
        sources = [{"kind": "tool", "title": tool_name, "url": ""}]
        yield {"type": "delta", "text": reply}

    # ------------------------------------------------ 3) image generation
    elif not reply and route == "image":
        yield {"type": "status", "stage": "painting"}
        image_info = imagegen.generate(meta["image_prompt"])
        caption = ("🎨 " + (meta["image_prompt"] or "your image"))
        reply, mode, conf = caption, "image", 0.9
        sources = [{"kind": "image", "title": image_info["provider"],
                    "url": image_info["url"], "prompt": meta["image_prompt"]}]
        yield {"type": "delta", "text": reply}

    # ------------------------------------------------ 4) live web search
    elif not reply and route == "search":
        yield {"type": "status", "stage": "searching"}
        try:
            results = search.web_search(meta["search_query"])
            if results:
                answer, hosts = search.compose_answer(meta["search_query"], results)
                if answer:
                    db.add_search_log(meta["search_query"], results)
                    mode, reply, conf, searched = "search", answer, 0.95, True
                    sources = [{"kind": "web", "title": r["title"], "url": r["url"]}
                               for r in results[:3]]
                    yield {"type": "delta", "text": reply}
        except Exception as exc:
            yield {"type": "notice",
                   "text": f"Web search unavailable ({exc.__class__.__name__}) — trying my own head."}

    # ------------------------------------------------ 5) knowledge (dataset memory)
    if not reply and (route in ("search", "knowledge") or lang != "en"):
        kb = knowledge.answer(text)
        if kb:
            mode, reply, conf = "knowledge", kb["text"], 0.8
            sources = (sources or []) + [
                {"kind": "dataset", "title": h["dataset"], "url": "",
                 "score": h.get("score"), "line_no": h.get("line_no"),
                 "snippet": (h.get("text") or "")[:140]}
                for h in kb["hits"]]
            yield {"type": "delta", "text": reply}

    # ------------------------------------------------ 6) neural net (streamed)
    if not reply:
        history = db.get_recent_messages(conversation_id, 6)
        mem_block = memory.recall_text(user["id"])
        sys_block = system_prompt.build(user, lang, memory_block=mem_block,
                                        route=route)
        prompt = build_prompt(history, text, sys_block)
        gen = engine.engine.stream(prompt)
        if gen is None:
            reply = languages.template_reply(text, "fallback", lang)
            mode = "template"
            yield {"type": "delta", "text": reply}
        else:
            yield {"type": "status", "stage": "generating"}
            stop = _ACTIVE_STOPS.setdefault(conversation_id, threading.Event())
            parts: List[str] = []
            final: Optional[Dict] = None
            try:
                for event in gen:
                    if stop.is_set():
                        break
                    if event["type"] == "delta":
                        parts.append(event["text"])
                        yield event
                    else:
                        final = event
            finally:
                _ACTIVE_STOPS.pop(conversation_id, None)
            raw = (final or {}).get("reply") or "".join(parts)
            conf = float((final or {}).get("confidence", 0.0))
            if raw and lang == "en":
                mode, reply = "neural", raw
            else:
                mode = "template"
                reply = languages.template_reply(text, "fallback", lang)

    # ------------------------------------------------ persist + learning loop
    final_text = humanize.humanize(reply, allow_softener=(mode == "neural"), lang=lang)
    final_text = safety.screen_output(final_text)
    msg_id = db.add_message(conversation_id, "assistant", final_text, mode, conf, lang=lang)
    if mode in ("neural", "search", "template", "knowledge"):
        pipeline.record_exchange(text, final_text, mode, lang=lang)
    pending = pipeline.maybe_start_retrain()
    if pending:
        yield {"type": "notice",
               "text": f"🧠 {pending} new exchanges collected — fine-tuning started in the background."}
    if mode == "neural" and conf < config.LOW_CONFIDENCE_THRESHOLD:
        yield {"type": "notice",
               "text": "🎓 I'm still learning this topic — vote 👍/👎 to teach me."}

    yield {"type": "done", "conversation_id": conversation_id,
           "user_message_id": user_msg_id, "message_id": msg_id,
           "reply": final_text, "source": mode, "route": route,
           "confidence": round(conf, 3), "searched": searched,
           "sources": sources, "lang": lang, "image": image_info,
           "suggestions": _suggest_followups(route, lang)}


def regenerate_stream(conversation_id: str, user: Dict) -> Iterator[Dict]:
    """Remove the last assistant reply and re-answer the last user message."""
    hist = db.get_history(conversation_id)
    last_user = next((m for m in reversed(hist) if m["role"] == "user"), None)
    last_bot = next((m for m in reversed(hist) if m["role"] == "assistant"), None)
    if not last_user:
        yield {"type": "error", "message": "nothing to regenerate yet"}
        return
    if last_bot:
        db.delete_message(last_bot["id"])
    yield from chat_stream(conversation_id, last_user["content"], user,
                           skip_user_save=True,
                           lang_override=last_user.get("lang"))


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


class MemoryIn(BaseModel):
    kind: str
    value: str

    @field_validator("kind")
    @classmethod
    def kind_valid(cls, v: str) -> str:
        allowed = {"name", "location", "likes", "dislikes", "work", "note"}
        if v not in allowed:
            raise ValueError(f"kind must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator("value")
    @classmethod
    def value_valid(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 200:
            raise ValueError("value must be 1-200 characters")
        return v


def build_stats() -> Dict:
    return {
        "database": db.stats(),
        "model": engine.engine.info(),
        "training": train.status(),
        "recent_runs": db.list_training_runs(5),
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "feedback_policy": pipeline.retrain_feedback_quality(),
        "cloud_sync": sync.status(),
        "datasets": datasets.status(),
        "knowledge_index": knowledge.stats(),
        "data_collection": datacol.stats(),
        "checkpoints": __import__("app.checkpoints", fromlist=["stats"]).stats(),
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


@app.get("/api/datasets")
def dataset_registry():
    """Long-term dataset memory: registry status + knowledge index stats."""
    return {"datasets": datasets.status(), "knowledge_index": knowledge.stats()}


@app.post("/api/datasets/{name}/refresh")
def refresh_dataset(name: str):
    """Re-download one registry dataset and rebuild the knowledge index."""
    entry = next((e for e in datasets.REGISTRY if e["name"] == name), None)
    if entry is None:
        raise HTTPException(404, f"unknown dataset '{name}'")
    info = datasets.ensure_one(entry)
    knowledge.build_index()
    return {"result": info, "knowledge_index": knowledge.stats()}


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
    for event in chat_stream(conv_id, text, user):
        if event["type"] == "done":
            done = event
    return {"conversation_id": conv_id, **done}


# --------------------------------------------------------------------------- #
# Memory API (long-term user memory)
# --------------------------------------------------------------------------- #
@app.get("/api/memory")
def get_memory(request: Request):
    user = require_user(request)
    return {"memories": memory.recall(user["id"], limit=100)}


@app.post("/api/memory")
def add_memory(body: MemoryIn, request: Request):
    user = require_user(request)
    ok = memory.remember(user["id"], body.kind, body.value, source="api")
    if not ok:
        return JSONResponse({"stored": False,
                             "detail": "already known (or invalid kind)"},
                            status_code=409)
    return {"stored": True}


@app.delete("/api/memory/{memory_id}")
def delete_memory(memory_id: str, request: Request):
    user = require_user(request)
    ok = db.forget_user_memory(user["id"], memory_id)
    if not ok:
        raise HTTPException(404, "memory not found")
    return {"forgotten": True}


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


@app.post("/api/finetune")
def start_finetune(request: Request):
    """Curated fine-tune: gentle low-LR pass on upvoted samples only."""
    require_user(request)
    if train.is_running():
        return JSONResponse({"started": False, "detail": "training already in progress"},
                            status_code=409)
    ok = train.start_finetune()
    if not ok:
        return JSONResponse({"started": False,
                             "detail": "no curated (upvoted 👍) samples yet — "
                                       "vote on answers to build a fine-tune set"},
                            status_code=409)
    return {"started": True, "steps": config.FINETUNE_STEPS,
            "lr": config.FINETUNE_LR}


# --------------------------------------------------------------------------- #
# Model versions (checkpoint registry)
# --------------------------------------------------------------------------- #
@app.get("/api/model/versions")
def model_versions(request: Request):
    require_user(request)
    from app import checkpoints
    return {"versions": checkpoints.list_checkpoints(), "stats": checkpoints.stats()}


@app.post("/api/model/rollback")
def model_rollback(request: Request, body: Dict = None):
    require_user(request)
    from app import checkpoints
    ckpt_id = (body or {}).get("id") if isinstance(body, dict) else None
    target = checkpoints.rollback(ckpt_id)
    if target is None:
        raise HTTPException(404, "no checkpoint to roll back to")
    return {"rolled_back_to": target["id"], "loss_after": target["loss_after"]}


@app.get("/api/model/versions/{ckpt_id}/download")
def download_checkpoint(ckpt_id: str, request: Request):
    """Download a saved model version's weights file (.npz)."""
    require_user(request)
    from app import checkpoints
    from fastapi.responses import FileResponse
    if checkpoints.get_checkpoint(ckpt_id) is None:
        raise HTTPException(404, "checkpoint not found")
    path = checkpoints.checkpoint_weights_path(ckpt_id)
    if path is None:
        raise HTTPException(404, "weights file missing for this checkpoint")
    return FileResponse(path, filename=f"dew-ai-{ckpt_id}.npz",
                        media_type="application/octet-stream")


# --------------------------------------------------------------------------- #
# System prompt (transparency — official AIs publish their system prompts)
# --------------------------------------------------------------------------- #
@app.get("/api/system-prompt")
def get_system_prompt(request: Request):
    user = require_user(request)
    lang = user.get("lang", config.DEFAULT_LANG)
    return system_prompt.inspect(user, lang,
                                 memory_block=memory.recall_text(user["id"]))


# --------------------------------------------------------------------------- #
# Data collection & dataset export
# --------------------------------------------------------------------------- #
@app.get("/api/data/stats")
def data_stats(request: Request):
    require_user(request)
    return datacol.stats()


@app.post("/api/data/export")
def data_export(request: Request):
    """Export the collected chat dataset as JSONL (official-style dataset file)."""
    require_user(request)
    return datacol.export_jsonl()


@app.get("/api/data/export/latest")
def data_export_latest(request: Request):
    """Download the most recent exported dataset file."""
    require_user(request)
    from fastapi.responses import FileResponse
    path = datacol.latest_export()
    if path is None:
        raise HTTPException(404, "no export yet — POST /api/data/export first")
    return FileResponse(path, filename=path.name, media_type="application/x-ndjson")


@app.post("/api/chat/{conversation_id}/stop")
def stop_chat(conversation_id: str, request: Request):
    """Stop an in-flight generation for this conversation (like ChatGPT's ⏹)."""
    user = require_user(request)
    conv = db.get_conversation(conversation_id)
    if conv is None or conv.get("user_id") != user["id"]:
        raise HTTPException(403, "this conversation belongs to another user")
    ev = _ACTIVE_STOPS.get(conversation_id)
    if ev:
        ev.set()
        return {"stopped": True}
    return {"stopped": False}


@app.post("/api/chat/{conversation_id}/regenerate")
def regenerate(conversation_id: str, request: Request):
    """Re-answer the last user message (ChatGPT-style ↻)."""
    user = require_user(request)
    conv = db.get_conversation(conversation_id)
    if conv is None or conv.get("user_id") != user["id"]:
        raise HTTPException(403, "this conversation belongs to another user")
    done: Dict = {}
    for event in regenerate_stream(conversation_id, user):
        if event["type"] == "done":
            done = event
    return {"conversation_id": conversation_id, **done}


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
            if mtype == "stop":
                cid = msg.get("conversation_id")
                ev = _ACTIVE_STOPS.get(cid or "")
                if ev:
                    ev.set()
                    await ws.send_json({"type": "notice", "text": "⏹ stopped"})
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
            async for event in iterate_in_threadpool(chat_stream(conv_id, text, user)):
                await ws.send_json(event)
    except Exception as exc:  # keep the socket protocol friendly on errors
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Generated images (cached on disk by app/imagegen.py)
# --------------------------------------------------------------------------- #
from fastapi.responses import FileResponse


@app.get("/images/{fname}")
def get_image(fname: str):
    path = imagegen.serve_path(fname)
    if path is None:
        raise HTTPException(404, "image not found")
    media = "image/svg+xml" if fname.endswith(".svg") else "image/png"
    return FileResponse(path, media_type=media)


# --------------------------------------------------------------------------- #
# Frontend (served from /frontend) — mounted last so /api routes win
# --------------------------------------------------------------------------- #
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True),
          name="frontend")
