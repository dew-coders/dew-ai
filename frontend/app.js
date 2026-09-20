/* NeuroChat frontend — real-time WebSocket chat client */
"use strict";

const $ = (s) => document.querySelector(s);
const messagesEl = $("#messages");
const typingEl = $("#typing");
const typingLabel = $("#typingLabel");
const inputEl = $("#input");
const sendBtn = $("#sendBtn");
const connDot = $("#connDot");
const modelBadge = $("#modelBadge");
const noticeBar = $("#noticeBar");

let ws = null;
let conversationId = Number(localStorage.getItem("nc_conv_id")) || null;
let currentBot = null;          // streaming target {bubble, parts}
let reconnectTimer = null;
let pollTimer = null;

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderRich(text) {
  // very light markdown: **bold**, `code`, links
  let html = escapeHtml(text);
  html = html.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return html;
}

const SOURCE_LABELS = {
  neural: "🧠 neural net",
  search: "🌐 web search",
  template: "💬 quick reply",
  fallback: "🎓 still learning",
};

function scrollBottom() {
  const chat = document.querySelector(".chat");
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text, meta = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "user" ? "user" : "bot"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = role === "user" ? escapeHtml(text) : renderRich(text);
  wrap.appendChild(bubble);

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  wrap.appendChild(metaEl);

  if (role !== "user") {
    if (meta.source) {
      const label = SOURCE_LABELS[meta.source] || meta.source;
      const conf = meta.confidence != null ? ` · ${(meta.confidence * 100).toFixed(0)}% conf` : "";
      metaEl.innerHTML = `<span>${label}${conf}</span>`;
    }
    if (meta.sources && meta.sources.length) {
      for (const s of meta.sources.slice(0, 3)) {
        const a = document.createElement("a");
        a.className = "src-chip";
        a.href = s.url || "#";
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = s.title || s.url || "source";
        a.title = s.title || "";
        metaEl.appendChild(a);
      }
    }
    if (meta.messageId) {
      const up = document.createElement("button");
      const down = document.createElement("button");
      up.className = "fb"; up.textContent = "👍"; up.title = "good answer";
      down.className = "fb"; down.textContent = "👎"; down.title = "bad answer";
      const vote = (rating, btn, other) => {
        fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message_id: meta.messageId, rating }),
        });
        btn.classList.add("active");
        other.classList.remove("active");
        up.disabled = down.disabled = true;
        showNotice(rating === 1 ? "👍 Thanks! This answer gets extra training weight."
                                : "👎 Got it — answers like this get excluded from training.");
      };
      up.onclick = () => vote(1, up, down);
      down.onclick = () => vote(-1, down, up);
      metaEl.appendChild(up);
      metaEl.appendChild(down);
    }
  }

  messagesEl.appendChild(wrap);
  scrollBottom();
  return { wrap, bubble, metaEl };
}

function setStage(stage) {
  const labels = {
    thinking: "thinking…",
    searching: "searching the web…",
    generating: "generating with my neural net…",
  };
  typingLabel.textContent = labels[stage] || stage;
  typingEl.classList.remove("hidden");
}

function hideTyping() { typingEl.classList.add("hidden"); }

function showNotice(text, sticky = false) {
  const el = document.createElement("div");
  el.className = "notice";
  el.textContent = text;
  noticeBar.appendChild(el);
  if (!sticky) setTimeout(() => el.remove(), 8000);
  return el;
}

function setBadge(info) {
  if (info && info.ready) {
    modelBadge.textContent = `${(info.parameters / 1000).toFixed(0)}k params · ${info.layers} layers · vocab ${info.vocab_size}`;
  } else {
    modelBadge.textContent = "model training from scratch…";
  }
}

/* ------------------------------------------------------------------ */
/* websocket                                                           */
/* ------------------------------------------------------------------ */
function connect() {
  clearTimeout(reconnectTimer);
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    connDot.className = "dot on";
    sendBtn.disabled = false;
  };

  ws.onclose = () => {
    connDot.className = "dot off";
    sendBtn.disabled = true;
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handleMessage(msg);
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case "hello":
      setBadge(msg.model);
      if (msg.training && msg.training.running) showNotice("🧠 Boot-time training in progress — the neural net is being built right now.", true);
      break;

    case "status":
      setStage(msg.stage);
      break;

    case "delta":
      hideTyping();
      if (!currentBot) currentBot = addMessage("bot", "");
      currentBot.parts = (currentBot.parts || "") + msg.text;
      currentBot.bubble.textContent = currentBot.parts;
      scrollBottom();
      break;

    case "notice":
      showNotice(msg.text);
      break;

    case "done": {
      hideTyping();
      conversationId = msg.conversation_id;
      localStorage.setItem("nc_conv_id", conversationId);
      if (currentBot) {
        currentBot.bubble.innerHTML = renderRich(msg.reply);
        // rebuild meta with final info + feedback buttons
        const meta = {
          source: msg.source, confidence: msg.confidence,
          sources: msg.sources, messageId: msg.message_id,
        };
        const wrap = currentBot.wrap;
        wrap.querySelector(".meta").remove();
        currentBot.metaEl = null;
        const fresh = addMessage("bot", msg.reply, meta);
        wrap.replaceWith(fresh.wrap);
      } else {
        addMessage("bot", msg.reply, msg);
      }
      currentBot = null;
      sendBtn.disabled = false;
      refreshStatsIfOpen();
      break;
    }

    case "stats":
      renderStats(msg.data);
      break;

    case "error":
      showNotice(`⚠️ ${msg.message}`);
      hideTyping();
      sendBtn.disabled = false;
      break;

    case "pong":
      break;
  }
}

/* ------------------------------------------------------------------ */
/* sending                                                             */
/* ------------------------------------------------------------------ */
function send() {
  const text = inputEl.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMessage("user", text);
  inputEl.value = "";
  autosize();
  sendBtn.disabled = true;
  setStage("thinking");
  ws.send(JSON.stringify({ type: "chat", text, conversation_id: conversationId }));
}

sendBtn.onclick = send;
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}
inputEl.addEventListener("input", autosize);

/* ------------------------------------------------------------------ */
/* stats + training controls                                           */
/* ------------------------------------------------------------------ */
const statsPanel = $("#statsPanel");
const statsBody = $("#statsBody");

$("#statsBtn").onclick = () => {
  statsPanel.classList.toggle("hidden");
  refreshStatsIfOpen();
};
$("#statsClose").onclick = () => statsPanel.classList.add("hidden");

function refreshStatsIfOpen() {
  if (!statsPanel.classList.contains("hidden") && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stats" }));
  }
}

function renderStats(d) {
  const m = d.model || {};
  const db = d.database || {};
  const t = d.training || {};
  const rows = [
    ["Model ready", m.ready ? "yes ✅" : "training…"],
    ["Parameters", m.parameters ? m.parameters.toLocaleString() : "—"],
    ["Architecture", m.ready ? `${m.layers} layers · ${m.heads} heads · d_model ${m.d_model}` : "—"],
    ["Context window", m.ready ? `${m.context_len} chars` : "—"],
    ["Tokenizer", m.ready ? m.tokenizer : "—"],
    ["—", ""],
    ["Conversations", db.conversations ?? "—"],
    ["Messages stored", db.messages ?? "—"],
    ["Training samples", db.training_samples ?? "—"],
    ["Pending new samples", `${db.new_samples_pending ?? 0} / ${d.retrain_threshold} until auto-retrain`],
    ["Web searches logged", db.searches ?? "—"],
    ["Feedback 👍 / 👎", `${db.feedback_up ?? 0} / ${db.feedback_down ?? 0}`],
    ["—", ""],
    ["Trainer state", t.progress || "idle"],
    ["Last trainer msg", t.last_message || "—"],
  ];
  let html = "<h3>Model</h3><table>";
  for (const [k, v] of rows) {
    if (k === "—") { html += "</table>"; continue; }
    html += `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`;
  }
  html += "<h3>Recent training runs</h3><table>";
  for (const r of (d.recent_runs || [])) {
    html += `<tr><td>#${r.id} ${r.trigger} (${r.status})</td><td>${r.loss_before != null ? r.loss_before.toFixed(3) + " → " + Number(r.loss_after).toFixed(3) : "—"}</td></tr>`;
  }
  html += "</table>";
  html += `<p style="margin-top:10px">${escapeHtml(JSON.stringify(d.feedback_policy || {}))}</p>`;
  statsBody.innerHTML = html;

  // training banner while a run is active
  const existing = document.querySelector(".notice.sticky");
  if (t.running) {
    if (!existing) {
      const el = showNotice(`🧠 ${t.progress || "training"} — ${t.last_message || "…"}`, true);
      el.classList.add("sticky");
    } else {
      existing.textContent = `🧠 ${t.progress || "training"} — ${t.last_message || "…"}`;
    }
    if (!pollTimer) pollTimer = setInterval(() => refreshStatsIfOpen(), 2500);
  } else {
    if (existing) existing.remove();
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }
}

$("#trainBtn").onclick = async () => {
  const res = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps: 300 }),
  });
  const data = await res.json();
  if (res.status === 409 || !data.started) {
    showNotice("Training is already running — check the live progress banner.");
  } else {
    showNotice(`🧠 Manual fine-tuning started (${data.steps} steps).`);
    statsPanel.classList.remove("hidden");
    refreshStatsIfOpen();
  }
};

/* ------------------------------------------------------------------ */
/* boot: restore latest conversation history                           */
/* ------------------------------------------------------------------ */
async function restoreHistory() {
  if (!conversationId) return;
  try {
    const res = await fetch(`/api/history/${conversationId}`);
    if (!res.ok) return;
    const rows = await res.json();
    for (const m of rows.slice(-50)) {
      addMessage(m.role === "user" ? "user" : "bot", m.content, {
        source: m.source, confidence: m.confidence, messageId: m.id,
      });
    }
  } catch { /* first run, ignore */ }
}

connect();
restoreHistory();
