"""Multi-language system: language detection, multilingual template matching
and localized canned replies.

The tiny char-level neural model only speaks English, so everything around it
(template answers, status text, stored metadata) adapts to the language the
user writes in. Detection combines Unicode-script analysis (Cyrillic → ru,
Arabic → ar, Devanagari → hi, kana → ja, Hangul → ko, Han → zh, …) with
stop-word scoring for Latin-script languages.
"""
from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional

LANGS: List[str] = ["en", "es", "fr", "de", "it", "pt", "ru", "tr", "ar", "hi", "zh", "ja", "ko"]

LANG_NAMES: Dict[str, str] = {
    "en": "English", "es": "Español", "fr": "Français", "de": "Deutsch",
    "it": "Italiano", "pt": "Português", "ru": "Русский", "tr": "Türkçe",
    "ar": "العربية", "hi": "हिन्दी", "zh": "中文", "ja": "日本語", "ko": "한국어",
}

# Languages where forcing capitalization / a trailing "." is wrong.
NON_CASED_LANGS = {"ar", "hi", "zh", "ja", "ko"}

# --------------------------------------------------------------------------- #
# Script-based detection
# --------------------------------------------------------------------------- #
_SCRIPTS: List[tuple] = [
    (re.compile(r"[\u3040-\u30ff]"), "ja"),    # hiragana + katakana
    (re.compile(r"[\uac00-\ud7af]"), "ko"),    # hangul
    (re.compile(r"[\u4e00-\u9fff]"), "zh"),    # CJK ideographs
    (re.compile(r"[\u0400-\u04ff]"), "ru"),    # cyrillic
    (re.compile(r"[\u0600-\u06ff]"), "ar"),    # arabic
    (re.compile(r"[\u0900-\u097f]"), "hi"),    # devanagari
    (re.compile(r"[\u0e00-\u0e7f]"), "th"),    # thai
    (re.compile(r"[\u0590-\u05ff]"), "he"),    # hebrew
    (re.compile(r"[\u0370-\u03ff]"), "el"),    # greek
]

# Stop-word scoring for Latin-script languages.
_STOPWORDS: Dict[str, tuple] = {
    "en": ("the", "is", "are", "you", "and", "what", "how", "why", "this", "that", "have", "can", "your"),
    "es": ("el", "la", "los", "las", "que", "de", "y", "es", "un", "una", "por", "para", "cómo", "qué", "eres", "hola", "como", "estas", "estás", "tal", "gracias", "muy", "bien"),
    "fr": ("le", "la", "les", "des", "est", "et", "je", "tu", "il", "une", "un", "pour", "comment", "quoi", "vous"),
    "de": ("der", "die", "das", "und", "ist", "ich", "du", "nicht", "ein", "eine", "wie", "was", "du", "bist"),
    "it": ("il", "lo", "la", "che", "di", "e", "un", "una", "sono", "come", "cosa", "per", "sei", "non"),
    "pt": ("os", "as", "que", "um", "uma", "você", "como", "não", "para", "com", "está", "seu", "qual"),
    "ru": ("и", "в", "не", "что", "как", "это", "ты", "я", "на", "с", "а", "по", "ты", "кто"),
    "tr": ("bir", "ve", "bu", "ne", "nasıl", "için", "ben", "sen", "değil", "ama", "kim", "mı", "merhaba", "nasılsın", "iyi", "çok", "evet", "hayır", "teşekkür"),
    "ar": ("في", "من", "على", "ما", "هذا", "كيف", "لماذا", "أنا", "أنت", "هو", "هل", "شيء"),
    "hi": ("है", "हैं", "क्या", "कैसे", "क्यों", "मैं", "आप", "और", "यह", "नहीं", "कौन", "तुम"),
}


def detect_language(text: str) -> str:
    """Best-effort language code for a user message (defaults to 'en')."""
    if not text:
        return "en"
    counts: Dict[str, int] = {}
    for pattern, lang in _SCRIPTS:
        hits = len(pattern.findall(text))
        if hits:
            counts[lang] = counts.get(lang, 0) + hits
    if counts:
        return max(counts, key=lambda k: counts[k])

    words = re.findall(r"[\w\u00c0-\u024f]+", text.lower())
    if not words:
        return "en"
    best_lang, best_score = "en", 0
    for lang, stops in _STOPWORDS.items():
        score = sum(1 for w in words if w in stops)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


# --------------------------------------------------------------------------- #
# Multilingual template matching (extends the original English regexes)
# --------------------------------------------------------------------------- #
_GREET = (r"hi+|hello+|hey+|yo|sup|good\s*(morning|afternoon|evening|day)"
          r"|hola|buen(os)?\s*(d[ií]as|tardes|noches)"
          r"|bonjour|salut|coucou|bonsoir"
          r"|hallo|guten\s*(tag|morgen|abend)|servus"
          r"|ciao|salve|buongiorno|buonasera"
          r"|ol[aá]|oii?|bom\s*dia|boa\s*(tarde|noite)"
          r"|привет|здравствуй|добрый\s*(день|вечер)"
          r"|merhaba|selam|g[uü]nayd[nu]ın?"
          r"|مرحبا|السلام|أهلا|هلا"
          r"|नमस्ते|नमस्कार|हैलो"
          r"|你好|您好|哈囉|嗨"
          r"|こんにちは|おはよう|こんばんは|やあ"
          r"|안녕|반가워|하이")
_THANKS = (r"thanks?|thank\s*you|thx|ty\b"
           r"|gracias|merci|danke|grazie|obrigad[oa]|arigat[oō]"
           r"|спасибо|teşekk[uü]r|sağ\s*ol|teşekküller"
           r"|شكرا|धन्यवाद|शुक्रिया|谢谢|感謝|ありがとう|감사|고마워")
_BYE = (r"bye|goodbye|good\s*night|see\s*you|cya"
        r"|adi[oó]s|hasta\s*luego|au\s*revoir|tsch[uü]ss|auf\s*wiedersehen"
        r"|arrivederci|tchau|at[eé]\s*(logo|mais)|пока|до\s*свидания"
        r"|g[uü]le\s*g[uü]le|hoşça\s*kal|مع\s*السلامة|अलविदा|फिर\s*मिलेंगे"
        r"|再见|拜拜|さようなら|またね|안녕히|잘가")
_IDENTITY = (r"who\s+are\s*you|what\s+are\s*you|your\s+name|about\s+yourself"
             r"|qui[eé]n\s+eres|qu[eé]\s+eres|tu\s+nombre"
             r"|qui\s*es(-|\s)tu|ton\s+nom|wer\s+bist\s*du|dein\s+name"
             r"|chi\s+sei|come\s+ti\s+chiami|quem\s+(é|és|e)\s+voc[eê]|seu\s+nome"
             r"|кто\s+ты|как\s+тебя\s+зовут|sen\s+kimsin|ad[nu]ın\s+ne"
             r"|من\s+أنت|اسمك|आप\s+कौन|तुम\s+कौन|आपका\s+नाम"
             r"|你是谁|你是什麼|你的名字|あなたは誰|名前は|누구세요|이름이\s*뭐")
_HELP = (r"help|what\s+can\s*you\s*do|capabilit|how\s+do\s+you\s+work"
         r"|ayuda|qu[eé]\s+puedes\s+hacer|aide|que\s+peux(-|\s)tu\s+faire"
         r"|hilfe|was\s+kannst\s*du|aiuto|cosa\s+puoi\s+fare"
         r"|ajuda|o\s+que\s+voc[eê]\s+(pode|faz)|помощь|что\s+ты\s+умеешь"
         r"|yard[iı]m|ne\s+yapabilirsin|مساعدة|मदद|帮助|你能做什么|助けて|도움|뭐\s+할\s+수")

_BOUND = r"(?:\b|$)"   # word boundary or end (CJK has no \b inside words)
_RE_GREET = re.compile(rf"^\s*({_GREET}){_BOUND}", re.I)
_RE_THANKS = re.compile(rf"^\s*({_THANKS}){_BOUND}", re.I)
_RE_BYE = re.compile(rf"^\s*({_BYE}){_BOUND}", re.I)
_RE_IDENTITY = re.compile(rf"({_IDENTITY})", re.I)
_RE_HELP = re.compile(rf"({_HELP})", re.I)


def template_kind(text: str) -> Optional[str]:
    """Detect chit-chat kinds (identity/greeting/help/thanks/bye) in any
    supported language. Returns None when the text needs a real answer."""
    if not text:
        return None
    # Arabic diacritics (tashkeel) would break "من أنت" → "مَن أنت" matches.
    stripped = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    if _RE_IDENTITY.search(stripped):
        return "identity"
    if _RE_GREET.match(stripped) and len(text) < 60:
        return "greeting"
    if _RE_HELP.search(stripped):
        return "help"
    if _RE_THANKS.match(stripped) and len(text) < 40:
        return "thanks"
    if _RE_BYE.match(stripped) and len(text) < 40:
        return "bye"
    return None


# --------------------------------------------------------------------------- #
# Localized canned replies
# --------------------------------------------------------------------------- #
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "en": {
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
    },
    "es": {
        "identity": [
            "Soy NeuroChat — una pequeña red neuronal escrita desde cero en NumPy, y aprendo un poco con cada conversación.",
            "Soy un transformador hecho a mano (sin PyTorch, lo prometo) entrenado con nuestras propias charlas.",
        ],
        "greeting": [
            "¡Hola! Qué gusto verte. ¿Qué tienes en mente?",
            "¡Hola! Pregúntame lo que quieras — y si no sé algo, di 'buscar: …' y lo miro en la web.",
        ],
        "help": [
            "Puedo charlar, recordar nuestras conversaciones, buscar en la web datos recientes (di 'buscar: …') y reentrenarme con lo que hablamos. ¡Pruébame!",
        ],
        "thanks": [
            "¡Siempre! Para eso estoy.",
            "¡De nada! Vuelve con más preguntas.",
        ],
        "bye": [
            "¡Hasta pronto! Seguiré aprendiendo mientras no estoy.",
            "¡Adiós! Cada chat nos hace mejores.",
        ],
        "fallback": [
            "Todavía estoy entrenando mi red con nuestras conversaciones, así que no estoy seguro de eso. Prueba otra vez en un minuto, o di 'buscar: …' y lo miro en la web.",
            "Hmm, mi cerebrito aún no tiene eso claro — está aprendiendo ahora mismo. ¡Pregúntame en un rato!",
        ],
    },
    "fr": {
        "identity": [
            "Je suis NeuroChat — un tout petit réseau de neurones écrit from scratch en NumPy, et j'apprends un peu à chaque conversation.",
            "Je suis un transformateur fait main (sans PyTorch, promis) entraîné sur nos propres discussions.",
        ],
        "greeting": [
            "Salut ! Ravi de te voir. Quoi de neuf ?",
            "Bonjour ! Pose-moi une question — et si je ne sais pas, dis « chercher : … » et je cherche sur le web.",
        ],
        "help": [
            "Je peux discuter, me souvenir de nos conversations, chercher des infos récentes sur le web (« chercher : … ») et me ré-entraîner avec ce qu'on dit. Essaie-moi !",
        ],
        "thanks": [
            "Avec plaisir !",
            "Je t'en prie ! Reviens avec d'autres questions.",
        ],
        "bye": [
            "À bientôt ! Je continue à apprendre pendant ton absence.",
            "Salut ! Chaque discussion me rend meilleur.",
        ],
        "fallback": [
            "J'entraîne encore mon réseau sur nos conversations, donc je ne suis pas sûr de ça. Réessaie dans une minute, ou dis « chercher : … » pour que je cherche en ligne.",
            "Hmm, mon petit cerveau n'a pas encore la réponse — il apprend en ce moment même !",
        ],
    },
    "de": {
        "identity": [
            "Ich bin NeuroChat — ein winziges, von Hand in NumPy gebautes neuronales Netz, das mit jedem Chat ein bisschen klüger wird.",
            "Ich bin ein von Hand gebauter Transformer (ohne PyTorch, ehrlich), trainiert mit unseren eigenen Gesprächen.",
        ],
        "greeting": [
            "Hallo! Schön, dich zu sehen. Was geht dir durch den Kopf?",
            "Hallo! Frag mich alles — und wenn ich es nicht weiß, sag 'suche: …' und ich schaue im Netz nach.",
        ],
        "help": [
            "Ich kann chatten, unsre Gespräche merken, im Web nach frischen Fakten suchen ('suche: …') und mich mit allem neu trainieren. Probier mich aus!",
        ],
        "thanks": [
            "Jederzeit! Dafür bin ich da.",
            "Gern geschehen! Komm mit mehr Fragen zurück.",
        ],
        "bye": [
            "Bis bald! Ich lerne weiter, während du weg bist.",
            "Tschüss! Jeder Chat macht mich besser.",
        ],
        "fallback": [
            "Ich trainiere mein Netzwerk noch an unseren Gesprächen, daher bin ich mir da nicht sicher. Versuch es gleich nochmal, oder sag 'suche: …', dann suche ich im Web.",
            "Hmm, mein Mini-Gehirn hat das noch nicht raus — es lernt ja gerade. Frag mich bald wieder!",
        ],
    },
    "it": {
        "identity": [
            "Sono NeuroChat — una piccolissima rete neurale scritta da zero in NumPy, e imparo qualcosa a ogni chiacchierata.",
            "Sono un transformer scritto a mano (senza PyTorch, giuro) addestrato sulle nostre conversazioni.",
        ],
        "greeting": [
            "Ehi! Che piacere vederti. Cosa hai in mente?",
            "Ciao! Chiedimi qualsiasi cosa — se non lo so, dì 'cerca: …' e guardo online.",
        ],
        "help": [
            "Posso chiacchierare, ricordare le nostre conversazioni, cercare informazioni fresche sul web ('cerca: …') e riaddestrarmi con tutto ciò che diciamo. Provalo!",
        ],
        "thanks": [
            "Sempre a disposizione!",
            "Prego! Torna con altre domande.",
        ],
        "bye": [
            "A presto! Continuerò a imparare anche da lontano.",
            "Ciao! Ogni chat mi rende migliore.",
        ],
        "fallback": [
            "Sto ancora addestrando la mia rete sulle nostre conversazioni, quindi non ne sono sicuro. Riprova tra un minuto, o dì 'cerca: …' e cerco online.",
            "Hmm, il mio cervellino non l'ha ancora capito — sta imparando proprio adesso. Riprova fra poco!",
        ],
    },
    "pt": {
        "identity": [
            "Eu sou o NeuroChat — uma rede neural minúscula escrita do zero em NumPy, e fico um pouco mais esperto a cada conversa.",
            "Sou um transformer feito à mão (sem PyTorch, prometo) treinado com as nossas próprias conversas.",
        ],
        "greeting": [
            "Oi! Que bom te ver. O que você está pensando?",
            "Olá! Pergunte qualquer coisa — se eu não souber, diga 'buscar: …' que eu procuro na web.",
        ],
        "help": [
            "Posso conversar, lembrar das nossas conversas, buscar fatos recentes na web ('buscar: …') e me retreinar com tudo o que falamos. Testa aí!",
        ],
        "thanks": [
            "Sempre! É pra isso que eu estou aqui.",
            "Por nada! Volte com mais perguntas.",
        ],
        "bye": [
            "Até logo! Vou continuar aprendendo enquanto você está fora.",
            "Tchau! Cada papo me deixa melhor.",
        ],
        "fallback": [
            "Ainda estou treinando minha rede com as nossas conversas, então não tenho certeza disso. Tente de novo em um minuto, ou diga 'buscar: …' e eu pesquiso na web.",
            "Hmm, meu cérebro pequeno ainda não sacou isso — ele está aprendendo agora. Pergunte de novo já já!",
        ],
    },
    "ru": {
        "identity": [
            "Я NeuroChat — крошечная нейросеть, написанная с нуля на NumPy, и я становлюсь немного умнее с каждым разговором.",
            "Я трансформер, собранный вручную (без PyTorch, честно) и обученный на наших собственных разговорах.",
        ],
        "greeting": [
            "Привет! Рад тебя видеть. О чём думаешь?",
            "Привет! Спрашивай что угодно — а если не знаю, скажи «поиск: …» и я посмотрю в интернете.",
        ],
        "help": [
            "Я умею болтать, помнить наши разговоры, искать свежие факты в сети («поиск: …») и дообучаться на всём, что мы обсуждаем. Попробуй!",
        ],
        "thanks": [
            "Всегда пожалуйста!",
            "Не за что! Возвращайся с новыми вопросами.",
        ],
        "bye": [
            "До скорого! Пока тебя нет, я продолжу учиться.",
            "Пока! Каждый чат делает меня лучше.",
        ],
        "fallback": [
            "Я ещё обучаю свою сеть на наших разговорах, поэтому не уверен в ответе. Попробуй через минуту или скажи «поиск: …» — поищу в интернете.",
            "Хм, мой маленький мозг этого пока не понял — он учится прямо сейчас. Спроси чуть позже!",
        ],
    },
    "tr": {
        "identity": [
            "Ben NeuroChat — NumPy ile sıfırdan yazılmış minicik bir sinir ağıyım ve her sohbetle biraz daha akıllanıyorum.",
            "Ben elle yazılmış bir transformerim (PyTorch'suz, söz) ve kendi sohbetlerimizle eğitiliyorum.",
        ],
        "greeting": [
            "Selam! Seni görmek güzel. Aklından ne geçiyor?",
            "Merhaba! Bana her şeyi sor — bilmiyorsam 'ara: …' de, internete bakarım.",
        ],
        "help": [
            "Sohbet edebilir, konuşmalarımızı hatırlayabilir, web'den güncel bilgiler arayabilir ('ara: …') ve konuştuğumuz her şeyle kendimi yeniden eğitebilirim. Dene bakalım!",
        ],
        "thanks": [
            "Ne zaman! Ben buradayım.",
            "Rica ederim! Yeni sorularla geri gel.",
        ],
        "bye": [
            "Görüşürüz! Sen yokken öğrenmeye devam ederim.",
            "Hoşça kal! Her sohbet beni daha iyi yapıyor.",
        ],
        "fallback": [
            "Sinir ağımı hâlâ konuşmalarımızla eğitiyorum, o yüzden bundan emin değilim. Bir dakika sonra tekrar dene ya da 'ara: …' de, internete bakayım.",
            "Hmm, minicik beynim bunu henüz çözemedi — şu anda bile öğreniyor. Az sonra yine sor!",
        ],
    },
    "ar": {
        "identity": [
            "أنا نيوروشات — شبكة عصبية صغيرة مبنية من الصفر بلغة NumPy، وأتعلم قليلاً مع كل محادثة.",
        ],
        "greeting": [
            "مرحباً! سعيد برؤيتك. ما الذي يخطر ببالك؟",
            "أهلاً! اسألني أي شيء — وإن لم أعرف قل «ابحث: …» وسأبحث في الإنترنت.",
        ],
        "help": [
            "أستطيع الدردشة وتذكر محادثاتنا والبحث عن معلومات حديثة على الويب («ابحث: …») وإعادة تدريب نفسي على كل ما نتحدث عنه. جرّبني!",
        ],
        "thanks": [
            "في أي وقت!",
            "على الرحب والسعة! عد بمزيد من الأسئلة.",
        ],
        "bye": [
            "أراك قريباً! سأواصل التعلم في غيابك.",
            "وداعاً! كل محادثة تجعلني أفضل.",
        ],
        "fallback": [
            "ما زلت أدرّب شبكتي على محادثاتنا، لست متأكداً من هذا بعد. حاول بعد دقيقة، أو قل «ابحث: …» وسأبحث في الإنترنت.",
            "همم، عقلي الصغير لم يفهم هذا بعد — إنه يتعلم الآن. اسألني بعد قليل!",
        ],
    },
    "hi": {
        "identity": [
            "मैं NeuroChat हूँ — NumPy में बिल्कुल शुरुआत से लिखा गया एक छोटा न्यूरल नेटवर्क, और हर बातचीत से मैं थोड़ा सीख जाता हूँ।",
        ],
        "greeting": [
            "नमस्ते! आपको देखकर अच्छा लगा। क्या सोच रहे हैं?",
            "नमस्ते! कुछ भी पूछिए — अगर मुझे नहीं पता तो 'खोजो: …' कहिए, मैं वेब से देख लूँगा।",
        ],
        "help": [
            "मैं बातचीत कर सकता हूँ, हमारी बातें याद रख सकता हूँ, वेब से ताज़ा जानकारी खोज सकता हूँ ('खोजो: …') और खुद को दोबारा train कर सकता हूँ। आज़माइए!",
        ],
        "thanks": [
            "कभी भी! इसीलिए तो हूँ।",
            "आपका स्वागत है! और सवालों के साथ लौटिए।",
        ],
        "bye": [
            "फिर मिलेंगे! आपके बिना भी मैं सीखता रहूँगा।",
            "अलविदा! हर बातचीत मुझे बेहतर बनाती है।",
        ],
        "fallback": [
            "मैं अभी अपना न्यूरल नेटवर्क हमारी बातचीत पर train कर रहा हूँ, इसलिए इसके बारे में पक्का नहीं हूँ। एक मिनट बाद फिर पूछिए, या 'खोजो: …' कहिए — मैं वेब से देख लूँगा।",
        ],
    },
    "zh": {
        "identity": [
            "我是 NeuroChat — 一个完全用 NumPy 从零写出来的小型神经网络，每次聊天我都会变得更聪明一点。",
        ],
        "greeting": [
            "你好！很高兴见到你。想聊点什么？",
            "你好！问我任何问题 — 如果我不懂，就说「搜索： …」我去网上查查。",
        ],
        "help": [
            "我可以聊天、记住我们的对话、上网搜索最新信息（说「搜索： …」），并用我们聊过的内容自我训练。试试吧！",
        ],
        "thanks": [
            "不客气！随时找我。",
            "不客气！欢迎带着更多问题回来。",
        ],
        "bye": [
            "回头见！你不在的时候我也会继续学习。",
            "再见！每一次聊天都让我更棒。",
        ],
        "fallback": [
            "我还在用我们的对话训练我的神经网络，所以这个问题我还答不好。过一分钟再试试，或者说「搜索： …」我帮你上网查。",
        ],
    },
    "ja": {
        "identity": [
            "僕は NeuroChat — NumPy でゼロから書かれたとても小さなニューラルネットで、話すたびに少しずつ賢くなるんだ。",
        ],
        "greeting": [
            "やあ！会えて嬉しいよ。何か話したいことは？",
            "こんにちは！なんでも聞いて — 分からないことは「検索: …」って言ってくれたらネットで調べるよ。",
        ],
        "help": [
            "おしゃべりや会話の記憶、ウェブ検索（「検索: …」）、話した内容での再学習ができるよ。試してみて！",
        ],
        "thanks": [
            "いつでもどうぞ！",
            "どういたしまして！また質問してね。",
        ],
        "bye": [
            "またね！いない間も勉強してるよ。",
            "バイバイ！おしゃべりするたびに僕は少し良くなるんだ。",
        ],
        "fallback": [
            "まだ君との会話でネットワークを訓練中だから、これはうまく答えられないかも。少し後でもう一度聞いて、「検索: …」って言ってくれたらネットで調べるよ。",
        ],
    },
    "ko": {
        "identity": [
            "나는 NeuroChat — NumPy로 처음부터 직접 만든 아주 작은 신경망이야. 대화할 때마다 조금씩 똑똑해져.",
        ],
        "greeting": [
            "안녕! 만나서 반가워. 무슨 생각 하고 있어?",
            "안녕! 뭐든 물어봐 — 모르겠으면 '검색: …'이라고 하면 인터넷에서 찾아줄게.",
        ],
        "help": [
            "채팅하고, 우리 대화를 기억하고, 웹에서 최신 정보를 찾고('검색: …'), 대화한 내용으로 스스로 학습할 수 있어. 한번 써봐!",
        ],
        "thanks": [
            "언제든지! 그러라고 있는 거야.",
            "천만에요! 더 궁금한 게 있으면 또 와.",
        ],
        "bye": [
            "다음에 보자! 네가 없는 동안에도 공부할 거야.",
            "잘 가! 대화할 때마다 나은 챗봇이 되고 있어.",
        ],
        "fallback": [
            "아직 우리 대화로 신경망을 학습 중이라서 그건 잘 모르겠어. 1분 뒤에 다시 물어봐 줘, 아니면 '검색: …'라고 하면 인터넷에서 찾아볼게.",
        ],
    },
}


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def template_reply(text: str, kind: str, lang: str = "en") -> str:
    """Pick a deterministic (per input text) canned reply in the user's language.
    Falls back to English when a kind/lang combination isn't translated."""
    lang = lang if lang in TEMPLATES else "en"
    bank = TEMPLATES[lang].get(kind) or TEMPLATES["en"].get(kind) or TEMPLATES["en"]["fallback"]
    return bank[_stable_hash(text) % len(bank)]
