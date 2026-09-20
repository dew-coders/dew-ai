/* NeuroChat frontend — login, real-time WebSocket chat, multi-language UI,
   chats sidebar with cloud recovery. */
"use strict";

const $ = (s) => document.querySelector(s);
const messagesEl = $("#messages");
const typingEl = $("#typing");
const typingLabel = $("#typingLabel");
const inputEl = $("#input");
const sendBtn = $("#sendBtn");
const stopBtn = $("#stopBtn");
const connDot = $("#connDot");
const modelBadge = $("#modelBadge");
const noticeBar = $("#noticeBar");
const langSelect = $("#langSelect");
const userChip = $("#userChip");
const userNameEl = $("#userName");
const logoutBtn = $("#logoutBtn");
const loginOverlay = $("#loginOverlay");
const loginBtn = $("#loginBtn");
const usernameInput = $("#usernameInput");
const loginError = $("#loginError");

let ws = null;
let token = localStorage.getItem("nc_token") || "";
let me = null;                   // {id, username, lang, …}
let conversationId = null;       // uuid string now
let currentBot = null;           // streaming target {wrap, bubble, parts}
let reconnectTimer = null;
let pollTimer = null;
let generating = false;          // true between send and done (⏹ stop support)

/* ------------------------------------------------------------------ */
/* i18n — UI strings per language                                      */
/* ------------------------------------------------------------------ */
const UI_STRINGS = {
  en: {
    tagline: "a neural network built from scratch — it learns from every chat",
    loginTitle: "Dew AI", loginSub: "Enter a username to start chatting — no password needed.",
    loginBtn: "Log in", loginHint: "Your chats are saved to the cloud, so you can recover them anytime on any device.",
    chatsBtn: "💬 Chats", statsBtn: "Stats", trainBtn: "Train now", logoutBtn: "Log out",
    chatsTitle: "Your chats", newChatBtn: "＋ New chat", recoverBtn: "☁ Recover",
    placeholder: "Say something…  (or try: search: latest AI news)",
    thinking: "thinking…", searching: "searching the web…", generating: "generating with my neural net…",
    send: "Send", noChats: "No chats yet — say hello!",
    stop: "⏹", regen: "↻", copy: "⧉", copied: "Copied!",
    recoverDone: "☁ Recovered {c} chat(s) / {m} messages from the cloud.",
    recoverOff: "☁ Cloud sync is not enabled — set SUPABASE_URL/KEY in .env.",
    loginFailed: "Login failed — try another username.",
  },
  es: {
    tagline: "una red neuronal hecha desde cero — aprende con cada chat",
    loginTitle: "Dew AI", loginSub: "Escribe un usuario para empezar — sin contraseña.",
    loginBtn: "Entrar", loginHint: "Tus chats se guardan en la nube: recupéralos cuando quieras, en cualquier dispositivo.",
    chatsBtn: "💬 Chats", statsBtn: "Stats", trainBtn: "Entrenar", logoutBtn: "Salir",
    chatsTitle: "Tus chats", newChatBtn: "＋ Nuevo chat", recoverBtn: "☁ Recuperar",
    placeholder: "Di algo…  (o prueba: buscar: noticias de IA)",
    thinking: "pensando…", searching: "buscando en la web…", generating: "generando con mi red neuronal…",
    send: "Enviar", noChats: "Aún no hay chats — ¡saluda!",
    recoverDone: "☁ Recuperados {c} chat(s) / {m} mensajes de la nube.",
    recoverOff: "☁ La sincronización no está activa — configura SUPABASE_URL/KEY en .env.",
    loginFailed: "No se pudo entrar — prueba otro usuario.",
  },
  fr: {
    tagline: "un réseau de neurones fait from scratch — il apprend à chaque chat",
    loginTitle: "Dew AI", loginSub: "Entre un pseudo pour discuter — sans mot de passe.",
    loginBtn: "Se connecter", loginHint: "Tes chats sont sauvegardés dans le cloud : récupère-les à tout moment, sur n'importe quel appareil.",
    chatsBtn: "💬 Chats", statsBtn: "Stats", trainBtn: "Entraîner", logoutBtn: "Déconnexion",
    chatsTitle: "Tes chats", newChatBtn: "＋ Nouveau chat", recoverBtn: "☁ Récupérer",
    placeholder: "Dis quelque chose…  (ou essaie : chercher : actus IA)",
    thinking: "réflexion…", searching: "recherche web…", generating: "génération avec mon réseau…",
    send: "Envoyer", noChats: "Pas encore de chat — dis bonjour !",
    recoverDone: "☁ {c} chat(s) / {m} messages récupérés du cloud.",
    recoverOff: "☁ Synchronisation inactive — configure SUPABASE_URL/KEY dans .env.",
    loginFailed: "Connexion échouée — essaie un autre pseudo.",
  },
  de: {
    tagline: "ein von Hand gebautes neuronales Netz — es lernt mit jedem Chat",
    loginTitle: "Dew AI", loginSub: "Gib einen Benutzernamen ein — kein Passwort nötig.",
    loginBtn: "Anmelden", loginHint: "Deine Chats werden in der Cloud gespeichert — jederzeit auf jedem Gerät wiederherstellbar.",
    chatsBtn: "💬 Chats", statsBtn: "Stats", trainBtn: "Trainieren", logoutBtn: "Abmelden",
    chatsTitle: "Deine Chats", newChatBtn: "＋ Neuer Chat", recoverBtn: "☁ Wiederherstellen",
    placeholder: "Sag etwas…  (oder: suche: AI-News)",
    thinking: "denkt nach…", searching: "suche im Web…", generating: "generiere mit meinem Netz…",
    send: "Senden", noChats: "Noch keine Chats — sag hallo!",
    recoverDone: "☁ {c} Chat(s) / {m} Nachrichten aus der Cloud wiederhergestellt.",
    recoverOff: "☁ Cloud-Sync aus — SUPABASE_URL/KEY in .env setzen.",
    loginFailed: "Anmeldung fehlgeschlagen — anderer Name?",
  },
  it: {
    tagline: "una rete neurale scritta da zero — impara da ogni chat",
    loginTitle: "Dew AI", loginSub: "Inserisci un nome utente per iniziare — senza password.",
    loginBtn: "Accedi", loginHint: "Le tue chat sono salvate nel cloud: recuperale quando vuoi, da qualsiasi dispositivo.",
    chatsBtn: "💬 Chat", statsBtn: "Stats", trainBtn: "Allena", logoutBtn: "Esci",
    chatsTitle: "Le tue chat", newChatBtn: "＋ Nuova chat", recoverBtn: "☁ Recupera",
    placeholder: "Di qualcosa…  (o prova: cerca: notizie AI)",
    thinking: "sto pensando…", searching: "cerco sul web…", generating: "genero con la mia rete…",
    send: "Invia", noChats: "Nessuna chat — salutaci!",
    recoverDone: "☁ Recuperate {c} chat / {m} messaggi dal cloud.",
    recoverOff: "☁ Sincronizzazione non attiva — imposta SUPABASE_URL/KEY in .env.",
    loginFailed: "Accesso non riuscito — prova un altro nome.",
  },
  pt: {
    tagline: "uma rede neural feita do zero — aprende a cada conversa",
    loginTitle: "Dew AI", loginSub: "Digite um usuário para começar — sem senha.",
    loginBtn: "Entrar", loginHint: "Seus chats ficam salvos na nuvem: recupere quando quiser, em qualquer aparelho.",
    chatsBtn: "💬 Chats", statsBtn: "Stats", trainBtn: "Treinar", logoutBtn: "Sair",
    chatsTitle: "Seus chats", newChatBtn: "＋ Novo chat", recoverBtn: "☁ Recuperar",
    placeholder: "Diga algo…  (ou tente: buscar: notícias de IA)",
    thinking: "pensando…", searching: "buscando na web…", generating: "gerando com minha rede…",
    send: "Enviar", noChats: "Nenhuma conversa ainda — diga oi!",
    recoverDone: "☁ {c} conversa(s) / {m} mensagens recuperadas da nuvem.",
    recoverOff: "☁ Sincronização desativada — configure SUPABASE_URL/KEY no .env.",
    loginFailed: "Falha no login — tente outro usuário.",
  },
  ru: {
    tagline: "нейросеть, написанная с нуля — учится на каждом чате",
    loginTitle: "Dew AI", loginSub: "Введите имя пользователя — без пароля.",
    loginBtn: "Войти", loginHint: "Чаты сохраняются в облако — восстановите их в любой момент на любом устройстве.",
    chatsBtn: "💬 Чаты", statsBtn: "Статы", trainBtn: "Обучить", logoutBtn: "Выйти",
    chatsTitle: "Ваши чаты", newChatBtn: "＋ Новый чат", recoverBtn: "☁ Восстановить",
    placeholder: "Напишите что-нибудь…  (или: поиск: новости ИИ)",
    thinking: "думаю…", searching: "ищу в интернете…", generating: "генерирую нейросетью…",
    send: "Отправить", noChats: "Чатов пока нет — поздоровайтесь!",
    recoverDone: "☁ Восстановлено чатов: {c} / сообщений: {m}.",
    recoverOff: "☁ Облако выключено — задайте SUPABASE_URL/KEY в .env.",
    loginFailed: "Не удалось войти — попробуйте другое имя.",
  },
  tr: {
    tagline: "sıfırdan yazılmış bir sinir ağı — her sohbette öğrenir",
    loginTitle: "Dew AI", loginSub: "Sohbete başlamak için bir kullanıcı adı gir — şifre gerekmez.",
    loginBtn: "Giriş yap", loginHint: "Sohbetlerin buluta kaydedilir — istediğin zaman, her cihazdan geri yükleyebilirsin.",
    chatsBtn: "💬 Sohbetler", statsBtn: "İstatistik", trainBtn: "Eğit", logoutBtn: "Çıkış",
    chatsTitle: "Sohbetlerin", newChatBtn: "＋ Yeni sohbet", recoverBtn: "☁ Geri yükle",
    placeholder: "Bir şey yaz…  (veya: ara: yapay zekâ haberleri)",
    thinking: "düşünüyor…", searching: "web'de arıyor…", generating: "sinir ağım üretiyor…",
    send: "Gönder", noChats: "Henüz sohbet yok — selam ver!",
    recoverDone: "☁ Buluttan {c} sohbet / {m} mesaj geri yüklendi.",
    recoverOff: "☁ Bulut senkronu kapalı — .env içine SUPABASE_URL/KEY ekle.",
    loginFailed: "Giriş başarısız — başka bir adım dene.",
  },
  ar: {
    tagline: "شبكة عصبية مبنية من الصفر — تتعلم من كل محادثة",
    loginTitle: "Dew AI", loginSub: "أدخل اسم مستخدم للبدء — بدون كلمة مرور.",
    loginBtn: "دخول", loginHint: "محادثاتك محفوظة في السحابة — استعدها في أي وقت وعلى أي جهاز.",
    chatsBtn: "💬 محادثات", statsBtn: "إحصاءات", trainBtn: "تدريب", logoutBtn: "خروج",
    chatsTitle: "محادثاتك", newChatBtn: "＋ محادثة جديدة", recoverBtn: "☁ استعادة",
    placeholder: "قل شيئاً…  (أو جرّب: ابحث: أخبار الذكاء الاصطناعي)",
    thinking: "أفكر…", searching: "أبحث في الويب…", generating: "أكتب بشبكتي العصبية…",
    send: "إرسال", noChats: "لا محادثات بعد — قل مرحباً!",
    recoverDone: "☁ تمت استعادة {c} محادثة / {m} رسالة من السحابة.",
    recoverOff: "☁ المزامنة غير مفعلة — عيّن SUPABASE_URL/KEY في .env.",
    loginFailed: "فشل الدخول — جرّب اسماً آخر.",
  },
  hi: {
    tagline: "शुरुआत से बना न्यूरल नेटवर्क — हर चैट से सीखता है",
    loginTitle: "Dew AI", loginSub: "शुरू करने के लिए यूज़रनेम लिखें — पासवर्ड की ज़रूरत नहीं।",
    loginBtn: "लॉग इन", loginHint: "आपकी चैट्स क्लाउड में सेव होती हैं — कभी भी, किसी भी डिवाइस पर वापस ला सकते हैं।",
    chatsBtn: "💬 चैट्स", statsBtn: "आँकड़े", trainBtn: "ट्रेन", logoutBtn: "लॉग आउट",
    chatsTitle: "आपकी चैट्स", newChatBtn: "＋ नई चैट", recoverBtn: "☁ रिकवर",
    placeholder: "कुछ कहिए…  (या: खोजो: AI न्यूज़)",
    thinking: "सोच रहा हूँ…", searching: "वेब पर खोज रहा हूँ…", generating: "न्यूरल नेट से बना रहा हूँ…",
    send: "भेजें", noChats: "अभी कोई चैट नहीं — नमस्ते कहिए!",
    recoverDone: "☁ क्लाउड से {c} चैट / {m} संदेश रिकवर किए।",
    recoverOff: "☁ क्लाउड सिंक बंद है — .env में SUPABASE_URL/KEY डालें।",
    loginFailed: "लॉगिन नाकाम — दूसरा नाम आज़माएँ।",
  },
  zh: {
    tagline: "一个从零开始构建的神经网络 — 它会从每次聊天中学习",
    loginTitle: "Dew AI", loginSub: "输入用户名即可开始聊天 — 无需密码。",
    loginBtn: "登录", loginHint: "你的聊天记录保存在云端，随时可以在任何设备上恢复。",
    chatsBtn: "💬 聊天", statsBtn: "统计", trainBtn: "训练", logoutBtn: "退出",
    chatsTitle: "你的聊天", newChatBtn: "＋ 新聊天", recoverBtn: "☁ 恢复",
    placeholder: "说点什么…  (或试试: 搜索： AI 新闻)",
    thinking: "思考中…", searching: "正在搜索网页…", generating: "神经网络生成中…",
    send: "发送", noChats: "还没有聊天 — 打个招呼吧！",
    recoverDone: "☁ 已从云端恢复 {c} 个聊天 / {m} 条消息。",
    recoverOff: "☁ 云同步未启用 — 请在 .env 中设置 SUPABASE_URL/KEY。",
    loginFailed: "登录失败 — 换个用户名试试。",
  },
  ja: {
    tagline: "ゼロから作ったニューラルネット — おしゃべりするたびに学習する",
    loginTitle: "Dew AI", loginSub: "ユーザー名を入力して開始 — パスワードは不要。",
    loginBtn: "ログイン", loginHint: "チャットはクラウドに保存されるので、いつでもどの端末でも復元できるよ。",
    chatsBtn: "💬 チャット", statsBtn: "統計", trainBtn: "訓練", logoutBtn: "ログアウト",
    chatsTitle: "チャット一覧", newChatBtn: "＋ 新しいチャット", recoverBtn: "☁ 復元",
    placeholder: "何か話しかけて…  (例: 検索: AIニュース)",
    thinking: "考え中…", searching: "ウェブで検索中…", generating: "ニューラルネットで生成中…",
    send: "送信", noChats: "まだチャットがない — 挨拶してみて！",
    recoverDone: "☁ クラウドから {c} 件のチャット / {m} 件のメッセージを復元したよ。",
    recoverOff: "☁ クラウド同期が無効 — .env に SUPABASE_URL/KEY を設定してね。",
    loginFailed: "ログイン失敗 — 別の名前で試して。",
  },
  si: {
    tagline: "සම්පූර්ණයෙන් හදපු නියුරල් ජාලයක් — හැම කතාවකින්ම ඉගෙන ගන්නවා",
    loginTitle: "Dew AI", loginSub: "පටන් ගන්න පරිශීලක නාමයක් ලියන්න — මුරපදයක් අවශ්‍ය නෑ.",
    loginBtn: "ඇතුල් වන්න", loginHint: "ඔයාගේ කතාබහ වලාවේ save වෙනවා — ඕනෑම වෙලාවක නැවත ලබාගන්න පුළුවන්.",
    chatsBtn: "💬 කතාබහ", statsBtn: "සංඛ්‍යාලේඛන", trainBtn: "පුහුණු", logoutBtn: "ඉවත් වන්න",
    chatsTitle: "ඔයාගේ කතාබහ", newChatBtn: "＋ අලුත් කතාබහ", recoverBtn: "☁ නැවත ලබාගන්න",
    placeholder: "මොකක් හරි කියන්න…  (නැත්නම්: සොයන්න: AI පුවත්)",
    thinking: "හිතනවා…", searching: "අන්තර්ජාලයේ සොයනවා…", generating: "නියුරල් ජාලයෙන් ලියනවා…",
    send: "යවන්න", noChats: "තාම කතාබහ නෑ — ආයුබෝවන් කියන්න!",
    recoverDone: "☁ වලාවෙන් කතාබහ {c}ක් / පණිවිඩ {m}ක් නැවත ලබාගත්තා.",
    recoverOff: "☁ වලාකුළු සම්මුහ කිරීම අක්‍රීයයි — .env වල SUPABASE_URL/KEY දාන්න.",
    loginFailed: "ඇතුල් වීම අසාර්ථක — වෙනත් නමක් උත්සාහ කරන්න.",
  },
  ko: {
    tagline: "처음부터 직접 만든 신경망 — every 채팅에서 배워요",
    loginTitle: "Dew AI", loginSub: "사용자 이름만 입력하면 시작할 수 있어요 — 비밀번호 필요 없음.",
    loginBtn: "로그인", loginHint: "채팅은 클라우드에 저장되어 언제든 어떤 기기에서든 복구할 수 있어요.",
    chatsBtn: "💬 채팅", statsBtn: "통계", trainBtn: "학습", logoutBtn: "로그아웃",
    chatsTitle: "내 채팅", newChatBtn: "＋ 새 채팅", recoverBtn: "☁ 복구",
    placeholder: "말 걸어 보세요…  (또는: 검색: AI 뉴스)",
    thinking: "생각 중…", searching: "웹 검색 중…", generating: "신경망이 생성 중…",
    send: "보내기", noChats: "아직 채팅이 없어요 — 인사해 보세요!",
    recoverDone: "☁ 클라우드에서 채팅 {c}개 / 메시지 {m}개를 복구했어요.",
    recoverOff: "☁ 클라우드 동기화 꺼짐 — .env에 SUPABASE_URL/KEY을 설정하세요.",
    loginFailed: "로그인 실패 — 다른 이름으로 시도해 보세요.",
  },
};

const LANGS = ["en", "si", "es", "fr", "de", "it", "pt", "ru", "tr", "ar", "hi", "zh", "ja", "ko"];
const LANG_NAMES = {
  en: "English", si: "සිංහල", es: "Español", fr: "Français", de: "Deutsch", it: "Italiano",
  pt: "Português", ru: "Русский", tr: "Türkçe", ar: "العربية", hi: "हिन्दी",
  zh: "中文", ja: "日本語", ko: "한국어",
};

let uiLang = localStorage.getItem("nc_lang") || "en";
if (!UI_STRINGS[uiLang]) uiLang = "en";

function t(key, vars = {}) {
  let s = (UI_STRINGS[uiLang] && UI_STRINGS[uiLang][key]) || UI_STRINGS.en[key] || key;
  for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, v);
  return s;
}

function applyI18n() {
  document.documentElement.lang = uiLang;
  document.body.dir = uiLang === "ar" ? "rtl" : "ltr";
  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  inputEl.placeholder = t("placeholder");
  sendBtn.textContent = t("send");
  for (const opt of langSelect.options) {
    opt.textContent = LANG_NAMES[opt.value] || opt.value;
  }
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function inlineMd(s) {
  return s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function renderRich(text) {
  /* ChatGPT-style light markdown: **bold**, `code`, [links], - lists,
     1. numbered lines, headings (#/##), line breaks. */
  const lines = escapeHtml(text).split("\n");
  let html = "", inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const line of lines) {
    const li = line.match(/^\s*[-•*]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    const h = line.match(/^#{1,4}\s+(.*)/);
    if (li) { if (!inList) { html += '<ul class="md-list">'; inList = true; }
              html += `<li>${inlineMd(li[1])}</li>`; continue; }
    closeList();
    if (h) { html += `<div class="md-h">${inlineMd(h[1])}</div>`; continue; }
    if (ol) { html += `<div class="md-li">${inlineMd(ol[1])}</div>`; continue; }
    if (line.trim() === "") { html += "<br>"; continue; }
    html += `<div>${inlineMd(line)}</div>`;
  }
  closeList();
  return html;
}

const SOURCE_LABELS = {
  neural: "🧠 neural net", search: "🌐 web search",
  template: "💬 quick reply", fallback: "🎓 still learning",
  knowledge: "📚 dataset memory",
};

function authHeaders(extra = {}) {
  return { "Authorization": `Bearer ${token}`, ...extra };
}

/* ---- rich source cards (dataset memory / web / tool / image) ---- */
function sourceChip(s) {
  const el = document.createElement(s.url ? "a" : "span");
  if (s.url) { el.href = s.url; el.target = "_blank"; el.rel = "noopener"; }
  el.className = `src-chip src-${s.kind || "web"}`;
  const body = (icon, title, sub) =>
    `<span class="src-ico">${icon}</span><span class="src-body">` +
    `<span class="src-title">${escapeHtml(title || "")}</span>` +
    (sub ? `<span class="src-sub">${escapeHtml(sub)}</span>` : "") + `</span>`;
  if (s.kind === "dataset") {
    el.innerHTML = body("📚", s.title || "dataset",
      s.score != null ? `match ${(s.score * 100).toFixed(0)}%` : "");
    el.title = s.snippet ? `${s.title} · line ${s.line_no ?? "?"}\n${s.snippet}` : (s.title || "");
  } else if (s.kind === "image") {
    el.innerHTML = `<img class="src-thumb" src="${s.url}" alt="">` +
      `<span class="src-body"><span class="src-title">🖼 ${escapeHtml(s.title || "image")}</span>` +
      (s.prompt ? `<span class="src-sub">${escapeHtml(String(s.prompt).slice(0, 48))}</span>` : "") +
      `</span>`;
    el.title = s.prompt || "generated image";
  } else if (s.kind === "tool") {
    el.innerHTML = body("🛠", `Tool · ${s.title || ""}`, "");
  } else {
    let host = "";
    if (s.url) { try { host = new URL(s.url).hostname; } catch { host = ""; } }
    el.innerHTML = body("🌐", s.title || host || "source", host);
  }
  return el;
}

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

  // generated image
  if (role !== "user" && meta.image && meta.image.url) {
    const img = document.createElement("img");
    img.className = "gen-image";
    img.src = meta.image.url;
    img.alt = text || "generated image";
    img.loading = "lazy";
    img.onclick = () => window.open(meta.image.url, "_blank");
    wrap.appendChild(img);
  }

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  wrap.appendChild(metaEl);

  if (role !== "user") {
    if (meta.source) {
      const label = SOURCE_LABELS[meta.source] || meta.source;
      const conf = meta.confidence != null ? ` · ${(meta.confidence * 100).toFixed(0)}% conf` : "";
      const langTag = meta.lang && meta.lang !== "en" ? ` · ${meta.lang.toUpperCase()}` : "";
      metaEl.innerHTML = `<span>${label}${conf}${langTag}</span>`;
    }
    if (meta.sources && meta.sources.length) {
      const srcWrap = document.createElement("div");
      srcWrap.className = "sources";
      for (const s of meta.sources.slice(0, 4)) {
        srcWrap.appendChild(sourceChip(s));
      }
      wrap.insertBefore(srcWrap, metaEl);
    }
    if (meta.messageId) {
      const up = document.createElement("button");
      const down = document.createElement("button");
      const speak = document.createElement("button");
      const copy = document.createElement("button");
      const regen = document.createElement("button");
      up.className = "fb"; up.textContent = "👍"; up.title = "good answer";
      down.className = "fb"; down.textContent = "👎"; down.title = "bad answer";
      speak.className = "fb"; speak.textContent = "🔊"; speak.title = "read aloud";
      copy.className = "fb"; copy.textContent = t("copy"); copy.title = "copy answer";
      regen.className = "fb regen"; regen.textContent = t("regen");
      regen.title = "generate a new answer";
      regen.onclick = () => regenerateLast();
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(bubble.innerText); } catch {}
        copy.textContent = "✓";
        setTimeout(() => { copy.textContent = t("copy"); }, 1200);
      };
      speak.onclick = () => speakText(bubble.textContent);
      const vote = (rating, btn, other) => {
        fetch("/api/feedback", {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
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
      metaEl.appendChild(speak);
      metaEl.appendChild(copy);
      metaEl.appendChild(regen);
    }
  }

  messagesEl.appendChild(wrap);
  scrollBottom();
  return { wrap, bubble, metaEl };
}

/* keep ↻ only on the newest bot message (ChatGPT behaviour) */
function markLastRegen(wrap) {
  for (const el of messagesEl.querySelectorAll(".regen")) el.remove();
  if (wrap) {
    const btn = wrap.querySelector(".regen");
    if (btn) btn.classList.remove("hidden");
  }
}

async function regenerateLast() {
  if (!conversationId || generating) return;
  setStage("thinking");
  sendBtn.disabled = true;
  try {
    const res = await fetch(`/api/chat/${conversationId}/regenerate`, {
      method: "POST", headers: authHeaders(),
    });
    const data = await res.json();
    if (res.ok && data.reply !== undefined) {
      const lastBot = [...messagesEl.querySelectorAll(".msg.bot")].pop();
      if (lastBot) lastBot.remove();
      addMessage("bot", data.reply, {
        source: data.source, confidence: data.confidence, lang: data.lang,
        sources: data.sources, messageId: data.message_id, image: data.image,
      });
    } else if (data.detail) {
      showNotice(`⚠️ ${data.detail}`);
    }
  } catch { showNotice("⚠️ regenerate failed"); }
  hideTyping();
  sendBtn.disabled = false;
}

function setStage(stage) {
  const labels = {
    thinking: "thinking…", searching: "searching the web…",
    generating: "generating with my neural net…",
    painting: "painting your image…", tool: "running the task…",
    listening: "listening…",
  };
  typingLabel.textContent = t(stage) || labels[stage] || stage;
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
/* auth                                                                */
/* ------------------------------------------------------------------ */
function showLogin(show) {
  loginOverlay.classList.toggle("hidden", !show);
  if (show) {
    usernameInput.value = localStorage.getItem("nc_last_username") || "";
    setTimeout(() => usernameInput.focus(), 50);
  }
}

async function doLogin() {
  const username = usernameInput.value.trim();
  if (username.length < 2) {
    loginError.textContent = "Username needs at least 2 characters.";
    loginError.classList.remove("hidden");
    return;
  }
  loginBtn.disabled = true;
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "login failed");
    token = data.token;
    me = data.user;
    localStorage.setItem("nc_token", token);
    localStorage.setItem("nc_last_username", username);
    loginError.classList.add("hidden");
    showLogin(false);
    applyUser();
    connect();
    refreshChats();
  } catch (err) {
    loginError.textContent = t("loginFailed");
    loginError.classList.remove("hidden");
  } finally {
    loginBtn.disabled = false;
  }
}

loginBtn.onclick = doLogin;
usernameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doLogin();
});

logoutBtn.onclick = async () => {
  try { await fetch("/api/logout", { method: "POST", headers: authHeaders() }); } catch {}
  localStorage.removeItem("nc_token");
  token = "";
  me = null;
  location.reload();
};

function applyUser() {
  if (!me) return;
  userChip.classList.remove("hidden");
  logoutBtn.classList.remove("hidden");
  userNameEl.textContent = me.username;
  if (me.lang && UI_STRINGS[me.lang]) {
    uiLang = me.lang;
    localStorage.setItem("nc_lang", uiLang);
    langSelect.value = uiLang;
    applyI18n();
  }
}

async function restoreSession() {
  if (!token) { showLogin(true); return; }
  try {
    const res = await fetch("/api/me", { headers: authHeaders() });
    if (!res.ok) throw new Error("expired");
    me = await res.json();
    applyUser();
    showLogin(false);
  } catch {
    token = "";
    localStorage.removeItem("nc_token");
    showLogin(true);
  }
}

/* ------------------------------------------------------------------ */
/* language switcher                                                   */
/* ------------------------------------------------------------------ */
for (const l of LANGS) {
  const opt = document.createElement("option");
  opt.value = l;
  langSelect.appendChild(opt);
}
langSelect.value = uiLang;
langSelect.onchange = async () => {
  uiLang = langSelect.value;
  localStorage.setItem("nc_lang", uiLang);
  applyI18n();
  if (token) {
    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ lang: uiLang }),
      });
    } catch {}
  }
};

/* ------------------------------------------------------------------ */
/* voice: speech-to-text (mic) + text-to-speech (speaker)              */
/* ------------------------------------------------------------------ */
const micBtn = $("#micBtn");
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SR) {
  recognition = new SR();
  recognition.lang = uiLang === "en" ? "en-US" : `${uiLang}-${uiLang.toUpperCase()}`;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onresult = (e) => {
    const said = e.results[0][0].transcript;
    inputEl.value = said;
    autosize();
    send();
  };
  recognition.onend = () => {
    micBtn.classList.remove("listening");
    micBtn.textContent = "🎙";
  };
  recognition.onerror = () => micBtn.classList.remove("listening");
} else {
  micBtn.disabled = true;
  micBtn.title = "speech recognition not supported in this browser";
}

micBtn.onclick = () => {
  if (!recognition) return;
  if (micBtn.classList.contains("listening")) {
    recognition.stop();
    return;
  }
  recognition.lang = uiLang === "en" ? "en-US" : `${uiLang}-${uiLang.toUpperCase()}`;
  micBtn.classList.add("listening");
  micBtn.textContent = "⏺";
  try { recognition.start(); } catch { /* already started */ }
};

function speakText(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.slice(0, 300));
  u.lang = uiLang === "en" ? "en-US" : `${uiLang}-${uiLang.toUpperCase()}`;
  window.speechSynthesis.speak(u);
}

/* ------------------------------------------------------------------ */
/* websocket                                                           */
/* ------------------------------------------------------------------ */
function connect() {
  clearTimeout(reconnectTimer);
  if (!token) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(token)}`);

  ws.onopen = () => {
    connDot.className = "dot on";
    sendBtn.disabled = false;
  };

  ws.onclose = (e) => {
    connDot.className = "dot off";
    sendBtn.disabled = true;
    if (e.code === 4401) {   // session expired → login again
      token = "";
      localStorage.removeItem("nc_token");
      showLogin(true);
      return;
    }
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
      if (msg.user) {
        me = { ...(me || {}), ...msg.user };
        applyUser();
      }
      if (msg.training && msg.training.running) showNotice("🧠 Boot-time training in progress — the neural net is being built right now.", true);
      break;

    case "auth_required":
      token = "";
      localStorage.removeItem("nc_token");
      showLogin(true);
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
      generating = false;
      stopBtn.classList.add("hidden");
      conversationId = msg.conversation_id;
      localStorage.setItem("nc_conv_id", conversationId);
      if (currentBot) {
        currentBot.bubble.innerHTML = renderRich(msg.reply);
        const meta = {
          source: msg.source, confidence: msg.confidence, lang: msg.lang,
          sources: msg.sources, messageId: msg.message_id, image: msg.image,
        };
        const wrap = currentBot.wrap;
        wrap.querySelector(".meta").remove();
        const fresh = addMessage("bot", msg.reply, meta);
        wrap.replaceWith(fresh.wrap);
        markLastRegen(fresh.wrap);
      } else {
        const fresh = addMessage("bot", msg.reply, msg);
        markLastRegen(fresh.wrap);
      }
      currentBot = null;
      sendBtn.disabled = false;
      showSuggestions(msg.suggestions);
      refreshStatsIfOpen();
      refreshChats();
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
  generating = true;
  stopBtn.classList.remove("hidden");
  setStage("thinking");
  ws.send(JSON.stringify({ type: "chat", text, conversation_id: conversationId }));
}

function stopGeneration() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop", conversation_id: conversationId }));
  }
  generating = false;
  stopBtn.classList.add("hidden");
  hideTyping();
  sendBtn.disabled = false;
}

stopBtn.onclick = stopGeneration;

function showSuggestions(list) {
  const old = document.querySelector(".suggestions");
  if (old) old.remove();
  if (!list || !list.length) return;
  const bar = document.createElement("div");
  bar.className = "suggestions";
  for (const s of list) {
    const chip = document.createElement("button");
    chip.className = "suggestion-chip";
    chip.textContent = s;
    chip.onclick = () => { bar.remove(); inputEl.value = s; autosize(); send(); };
    bar.appendChild(chip);
  }
  messagesEl.appendChild(bar);
  scrollBottom();
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
/* chats sidebar (list / switch / new / cloud recovery)                */
/* ------------------------------------------------------------------ */
const chatsPanel = $("#chatsPanel");
const chatsList = $("#chatsList");

$("#chatsBtn").onclick = () => {
  chatsPanel.classList.toggle("hidden");
  refreshChats();
};
$("#chatsClose").onclick = () => chatsPanel.classList.add("hidden");
$("#newChatBtn").onclick = () => {
  conversationId = null;
  localStorage.removeItem("nc_conv_id");
  messagesEl.innerHTML = "";
  chatsPanel.classList.add("hidden");
  inputEl.focus();
};

$("#recoverBtn").onclick = async () => {
  const btn = $("#recoverBtn");
  btn.disabled = true;
  try {
    const res = await fetch("/api/recover", { method: "POST", headers: authHeaders() });
    const data = await res.json();
    if (data.error) {
      showNotice(t("recoverOff"));
    } else {
      showNotice(t("recoverDone", { c: data.conversations || 0, m: data.messages || 0 }));
    }
    refreshChats();
  } catch {
    showNotice(t("recoverOff"));
  } finally {
    btn.disabled = false;
  }
};

async function refreshChats() {
  if (!token || chatsPanel.classList.contains("hidden")) return;
  try {
    const res = await fetch("/api/conversations", { headers: authHeaders() });
    if (!res.ok) return;
    const convs = await res.json();
    chatsList.innerHTML = "";
    if (!convs.length) {
      chatsList.innerHTML = `<p class="chats-empty">${escapeHtml(t("noChats"))}</p>`;
      return;
    }
    for (const c of convs) {
      const item = document.createElement("button");
      item.className = "chat-item" + (c.id === conversationId ? " active" : "");
      const title = document.createElement("span");
      title.className = "chat-title";
      title.textContent = c.title || "…";
      const sub = document.createElement("span");
      sub.className = "chat-sub";
      sub.textContent = `${c.msg_count} · ${new Date(c.created_at + "Z").toLocaleString()}`;
      item.appendChild(title);
      item.appendChild(sub);
      item.onclick = () => openConversation(c.id);
      chatsList.appendChild(item);
    }
  } catch { /* offline */ }
}

async function openConversation(id) {
  conversationId = id;
  localStorage.setItem("nc_conv_id", id);
  messagesEl.innerHTML = "";
  chatsPanel.classList.add("hidden");
  try {
    const res = await fetch(`/api/history/${id}`, { headers: authHeaders() });
    if (!res.ok) return;
    const rows = await res.json();
    for (const m of rows.slice(-50)) {
      const added = addMessage(m.role === "user" ? "user" : "bot", m.content, {
        source: m.source, confidence: m.confidence, lang: m.lang, messageId: m.id,
      });
      if (m.role !== "user") markLastRegen(added.wrap);
    }
  } catch { /* ignore */ }
}

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
  const c = d.cloud_sync || {};
  const rows = [
    ["Model ready", m.ready ? "yes ✅" : "training…"],
    ["Parameters", m.parameters ? m.parameters.toLocaleString() : "—"],
    ["Architecture", m.ready ? `${m.layers} layers · ${m.heads} heads · d_model ${m.d_model}` : "—"],
    ["Context window", m.ready ? `${m.context_len} chars` : "—"],
    ["Tokenizer", m.ready ? m.tokenizer : "—"],
    ["—", ""],
    ["Users", db.users ?? "—"],
    ["Conversations", db.conversations ?? "—"],
    ["Messages stored", db.messages ?? "—"],
    ["Training samples", db.training_samples ?? "—"],
    ["Pending new samples", `${db.new_samples_pending ?? 0} / ${d.retrain_threshold} until auto-retrain`],
    ["Web searches logged", db.searches ?? "—"],
    ["Feedback 👍 / 👎", `${db.feedback_up ?? 0} / ${db.feedback_down ?? 0}`],
    ["—", ""],
    ["Cloud sync", c.enabled ? "on ☁" : "off"],
    ["Sync pending rows", c.pending ?? 0],
    ["Sync pushed / failed", `${c.pushed_total ?? 0} / ${c.failed_total ?? 0}`],
    ["—", ""],
    ["Trainer state", t.progress || "idle"],
    ["Last trainer msg", t.last_message || "—"],
  ];
  let html = "<h3>Model</h3><table>";
  for (const [k, v] of rows) {
    if (k === "—") { html += "</table>"; continue; }
    html += `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`;
  }
  html += "</table>";
  if (c.last_error) html += `<p class="sync-err">⚠ ${escapeHtml(c.last_error)}</p>`;
  html += "<h3>Recent training runs</h3><table>";
  for (const r of (d.recent_runs || [])) {
    html += `<tr><td>#${r.id} ${r.trigger} (${r.status})</td><td>${r.loss_before != null ? r.loss_before.toFixed(3) + " → " + Number(r.loss_after).toFixed(3) : "—"}</td></tr>`;
  }
  html += "</table>";
  html += `<p style="margin-top:10px">${escapeHtml(JSON.stringify(d.feedback_policy || {}))}</p>`;

  // ------------------------------------------------ model versions + rollback
  const cps = (d.checkpoints && d.checkpoints.versions) || [];
  html += "<h3>Model versions</h3>";
  if (!cps.length) {
    html += '<p class="muted">No model versions yet — one is saved automatically after each training run finishes.</p>';
  } else {
    html += '<table class="versions">';
    for (const v of cps) {
      const when = v.created_at ? new Date(v.created_at * 1000).toLocaleString() : "";
      const loss = (v.loss_before != null && v.loss_after != null)
        ? `${v.loss_before.toFixed(2)} → ${v.loss_after.toFixed(2)}` : "—";
      const isLatest = d.checkpoints.latest === v.id;
      html += `<tr>` +
        `<td><span class="ver-id">${escapeHtml(v.id)}</span><br>` +
        `<span class="ver-sub">${escapeHtml(v.trigger)} · ${v.steps} steps · loss ${loss} · ${when}</span></td>` +
        `<td class="ver-actions">` +
        (isLatest ? '<span class="ver-live">live</span>'
                  : `<button class="rollback-btn" data-ckpt="${escapeHtml(v.id)}">↩ Roll back</button>`) +
        ` <button class="rollback-btn dl-btn" data-ckpt="${escapeHtml(v.id)}" title="download weights (.npz)">⬇</button>` +
        `</td></tr>`;
    }
    html += "</table>";
  }
  statsBody.innerHTML = html;

  // wire rollback buttons
  for (const btn of statsBody.querySelectorAll(".rollback-btn:not(.dl-btn)")) {
    btn.onclick = () => rollbackModel(btn.dataset.ckpt, btn);
  }
  for (const btn of statsBody.querySelectorAll(".dl-btn")) {
    btn.onclick = () => downloadCheckpoint(btn.dataset.ckpt);
  }

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

function downloadCheckpoint(ckptId) {
  const a = document.createElement("a");
  a.href = `/api/model/versions/${encodeURIComponent(ckptId)}/download?token=${encodeURIComponent(token)}`;
  a.download = `dew-ai-${ckptId}.npz`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function rollbackModel(ckptId, btn) {
  if (!confirm(`Restore model version ${ckptId}? The live bot will hot-reload those weights.`)) return;
  btn.disabled = true;
  try {
    const res = await fetch("/api/model/rollback", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ id: ckptId }),
    });
    const data = await res.json();
    if (res.ok) {
      showNotice(`↩ Model rolled back to ${data.rolled_back_to} (loss ${data.loss_after}) — hot-reloaded.`);
    } else {
      showNotice(`⚠️ ${data.detail || "rollback failed"}`);
      btn.disabled = false;
    }
  } catch {
    showNotice("⚠️ rollback failed");
    btn.disabled = false;
  }
  refreshStatsIfOpen();
}

$("#trainBtn").onclick = async () => {
  const res = await fetch("/api/train", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
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
/* boot                                                                */
/* ------------------------------------------------------------------ */
applyI18n();
restoreSession().then(() => {
  if (token) {
    connect();
    const saved = localStorage.getItem("nc_conv_id");
    if (saved) openConversation(saved);
  }
});
