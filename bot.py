import ast
import base64
import hashlib
import html
import json
import math
import operator
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"
RAG_STORE = DATA_DIR / "rag_store.json"
EMBEDDING_DIMS = 384

RU = {
    "friend": "\u0434\u0440\u0443\u0433",
    "start": (
        "\u041f\u0440\u0438\u0432\u0435\u0442, {name}! \u0422\u0435\u043f\u0435\u0440\u044c \u044f \u043d\u0435 \u044d\u0445\u043e-\u0431\u043e\u0442.\n\n"
        "\u041f\u0438\u0448\u0438 \u043e\u0431\u044b\u0447\u043d\u044b\u043c \u0442\u0435\u043a\u0441\u0442\u043e\u043c: \u0432\u043e\u043f\u0440\u043e\u0441, \u043c\u0430\u0442\u0435\u043c\u0430\u0442\u0438\u043a\u0443, \u043e\u0446\u0435\u043d\u043a\u0443 \u041f\u041a-\u0441\u0431\u043e\u0440\u043a\u0438. "
        "\u0415\u0441\u043b\u0438 \u0434\u043e\u0431\u0430\u0432\u0438\u0448\u044c AI API \u043a\u043b\u044e\u0447 \u0432 .env, \u044f \u0441\u043c\u043e\u0433\u0443 \u043e\u0442\u0432\u0435\u0447\u0430\u0442\u044c \u043f\u043e\u0447\u0442\u0438 \u043d\u0430 \u0432\u0441\u0435."
    ),
    "help": (
        "\u042f \u0443\u043c\u0435\u044e:\n"
        "- \u0441\u0447\u0438\u0442\u0430\u0442\u044c \u043f\u0440\u0438\u043c\u0435\u0440\u044b \u0442\u0438\u043f\u0430 1+1, (5+3)*2\n"
        "- \u043e\u0446\u0435\u043d\u0438\u0432\u0430\u0442\u044c \u041f\u041a-\u0441\u0431\u043e\u0440\u043a\u0438\n"
        "- \u043e\u0442\u0432\u0435\u0447\u0430\u0442\u044c \u043d\u0430 \u0431\u0430\u0437\u043e\u0432\u044b\u0435 \u0432\u043e\u043f\u0440\u043e\u0441\u044b\n\n"
        "\u0414\u043b\u044f \u043f\u043e-\u043d\u0430\u0441\u0442\u043e\u044f\u0449\u0435\u043c\u0443 \u0441\u0432\u043e\u0431\u043e\u0434\u043d\u044b\u0445 \u043e\u0442\u0432\u0435\u0442\u043e\u0432 \u043d\u0443\u0436\u0435\u043d AI API \u043a\u043b\u044e\u0447: GROQ_API_KEY \u0438\u043b\u0438 OPENAI_API_KEY \u0432 .env."
    ),
    "unknown": (
        "\u042f \u043f\u043e\u043d\u044f\u043b \u0432\u043e\u043f\u0440\u043e\u0441, \u043d\u043e \u0431\u0435\u0437 AI-\u043a\u043b\u044e\u0447\u0430 \u043d\u0435 \u0445\u043e\u0447\u0443 \u0432\u0440\u0430\u0442\u044c. "
        "\u041c\u043e\u0433\u0443 \u0442\u043e\u0447\u043d\u043e \u043f\u043e\u0441\u0447\u0438\u0442\u0430\u0442\u044c \u043c\u0430\u0442\u0435\u043c\u0430\u0442\u0438\u043a\u0443 \u0438\u043b\u0438 \u043e\u0446\u0435\u043d\u0438\u0442\u044c \u041f\u041a-\u0441\u0431\u043e\u0440\u043a\u0443. "
        "\u0427\u0442\u043e\u0431\u044b \u044f \u043e\u0442\u0432\u0435\u0447\u0430\u043b \u043a\u0430\u043a \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u044b\u0439 \u0447\u0430\u0442-\u0431\u043e\u0442 \u043d\u0430 \u043b\u044e\u0431\u0443\u044e \u0442\u0435\u043c\u0443, \u0434\u043e\u0431\u0430\u0432\u044c GROQ_API_KEY \u0432 .env."
    ),
}


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def load_env(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def post_json(url, payload, headers=None, timeout=40):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "telegram-bot/1.0",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body.strip() else None
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {details}") from error


def telegram_request(token, method, payload=None, timeout=35):
    url = TELEGRAM_API.format(token=token, method=method)
    parsed = post_json(url, payload or {}, timeout=timeout)
    if not parsed.get("ok"):
        raise RuntimeError(parsed)
    return parsed["result"]


def telegram_file_url(token, file_id):
    file_info = telegram_request(token, "getFile", {"file_id": file_id}, timeout=20)
    return f"https://api.telegram.org/file/bot{token}/{file_info['file_path']}"


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "telegram-bot/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def send_message(token, chat_id, text):
    if len(text) > 3900:
        text = text[:3900].rstrip() + "\n..."
    return telegram_request(
        token,
        "sendMessage",
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
    )


def post_supabase(path, payload):
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        return None

    return post_json(
        f"{url}/rest/v1/{path}",
        payload,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=minimal",
        },
        timeout=30,
    )


def load_rag_store():
    if not RAG_STORE.exists():
        return []
    try:
        return json.loads(RAG_STORE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_rag_store(items):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAG_STORE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def text_embedding(text, dims=EMBEDDING_DIMS):
    vector = [0.0] * dims
    words = re.findall(r"[\w]+", normalize_text(text), flags=re.UNICODE)
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))


def chunk_text(text, size=1300, overlap=180):
    clean = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end) if end == len(clean) else end - overlap
    return chunks


def extract_pdf_text(path):
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def add_document_to_rag(source, text):
    chunks = chunk_text(text)
    store = load_rag_store()
    rows = []
    for index, chunk in enumerate(chunks):
        item = {
            "id": hashlib.sha256(f"{source}:{index}:{chunk[:80]}".encode("utf-8")).hexdigest(),
            "source": source,
            "chunk_index": index,
            "content": chunk,
            "embedding": text_embedding(chunk),
        }
        store.append(item)
        rows.append(item)

    save_rag_store(store)
    try:
        post_supabase(
            "documents",
            [
                {
                    "id": row["id"],
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "embedding": row["embedding"],
                }
                for row in rows
            ],
        )
    except Exception as error:
        print(f"Supabase upsert error: {error}")
    return len(chunks)


def rag_search(query, limit=5):
    query_embedding = text_embedding(query)
    local_results = []
    for item in load_rag_store():
        score = cosine_similarity(query_embedding, item.get("embedding", []))
        local_results.append((score, item))

    local_results.sort(key=lambda pair: pair[0], reverse=True)
    results = [
        {"source": item["source"], "content": item["content"], "score": score}
        for score, item in local_results[:limit]
        if score > 0.05
    ]

    try:
        remote = post_supabase(
            "rpc/match_documents",
            {"query_embedding": query_embedding, "match_count": limit},
        )
        if isinstance(remote, list):
            for item in remote:
                results.append(
                    {
                        "source": item.get("source", "supabase"),
                        "content": item.get("content", ""),
                        "score": item.get("similarity", 0),
                    }
                )
    except Exception as error:
        print(f"Supabase search error: {error}")

    seen = set()
    unique = []
    for item in sorted(results, key=lambda row: row.get("score", 0), reverse=True):
        key = item["content"][:160]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:limit]


def format_rag_context(results):
    if not results:
        return ""
    parts = []
    for index, item in enumerate(results, 1):
        parts.append(f"[{index}] {item['source']}: {item['content']}")
    return "\n\n".join(parts)


def web_search(query, limit=5):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        page = response.read().decode("utf-8", errors="replace")

    results = []
    blocks = re.split(r'<li class="b_algo"', page)[1:]
    for block in blocks:
        title_match = re.search(r'<h2.*?<a[^>]*href="(?P<url>.*?)"[^>]*>(?P<title>.*?)</a>', block, re.DOTALL)
        snippet_match = re.search(r"<p[^>]*>(?P<snippet>.*?)</p>", block, re.DOTALL)
        if not title_match:
            continue

        final_url = decode_bing_url(html.unescape(title_match.group("url")))
        title = html.unescape(re.sub(r"<.*?>", "", title_match.group("title")))
        snippet = ""
        if snippet_match:
            snippet = html.unescape(re.sub(r"<.*?>", "", snippet_match.group("snippet")))
        results.append({"title": title.strip(), "url": final_url, "snippet": snippet.strip()})
        if len(results) >= limit:
            break
    return results[:limit]


def enrich_web_query(query):
    n = normalize_text(query)
    current_words = [
        "\u043d\u043e\u0432\u043e\u0441\u0442",
        "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
        "\u0441\u0435\u0439\u0447\u0430\u0441",
        "\u0441\u0432\u0435\u0436",
        "\u0430\u043a\u0442\u0443\u0430\u043b",
        "\u043a\u0443\u0440\u0441",
        "2026",
        "latest",
        "news",
        "current",
    ]
    if any(word in n for word in current_words):
        return f"{query} {datetime.now().strftime('%Y-%m-%d')}"
    return query


def decode_bing_url(url):
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    encoded = params.get("u", [None])[0]
    if not encoded:
        return url
    if encoded.startswith("a1"):
        encoded = encoded[2:]
    try:
        padding = "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode((encoded + padding).encode("ascii")).decode("utf-8")
    except Exception:
        return url


def format_web_context(results):
    if not results:
        return ""
    return "\n".join(
        f"[{index}] {item['title']} - {item['snippet']} ({item['url']})"
        for index, item in enumerate(results, 1)
    )


def normalize_text(text):
    text = text.lower().replace("\u0451", "\u0435")
    text = re.sub(r"[^\w\s/+*().,%:-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = safe_eval_node(node.left)
        right = safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Power is too large")
        return SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](safe_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def try_math(text):
    expression = text.strip().replace(",", ".")
    if not re.fullmatch(r"[0-9\s.+\-*/()%]+", expression):
        return None
    if not re.search(r"\d\s*[+\-*/%]\s*\d", expression):
        return None
    expression = expression.replace("%", "/100")
    value = safe_eval_node(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and math.isfinite(value):
        value = round(value, 10)
    return f"\u041e\u0442\u0432\u0435\u0442: {value}"


def evaluate_pc_build(text):
    n = normalize_text(text)
    looks_like_build = any(word in n for word in ["\u0441\u0431\u043e\u0440\u043a", "\u043f\u043a", "rtx", "gtx", "ddr", "hdd", "ssd", "\u043f\u0435\u043d\u0442\u0438\u0443\u043c", "pentium"])
    asks_rating = any(word in n for word in ["\u043e\u0446\u0435\u043d", "\u043d\u043e\u0440\u043c", "\u0445\u043e\u0440\u043e\u0448", "\u043a\u0430\u043a", "\u0441\u0431\u043e\u0440\u043a"])
    if not (looks_like_build and asks_rating):
        return None

    issues = []
    good = []

    if "pentium" in n or "\u043f\u0435\u043d\u0442\u0438\u0443\u043c" in n:
        issues.append("\u041f\u0440\u043e\u0446\u0435\u0441\u0441\u043e\u0440 Pentium \u043e\u0447\u0435\u043d\u044c \u0441\u043b\u0430\u0431\u044b\u0439 \u0434\u043b\u044f \u0441\u043e\u0432\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0445 \u0438\u0433\u0440 \u0438 \u0442\u043e\u0447\u043d\u043e \u0431\u0443\u0434\u0435\u0442 \u0443\u043f\u0438\u0440\u0430\u0442\u044c\u0441\u044f \u0432 RTX 5090.")
    if re.search(r"\b1\s*(gb|g|гб|гиг|гигa|гига|гб)\b", n) or "\u043e\u0437\u0443 1" in n or "1\u0433\u0431" in n:
        issues.append("1 \u0413\u0411 \u041e\u0417\u0423 \u0441\u0435\u0439\u0447\u0430\u0441 \u043a\u0440\u0430\u0439\u043d\u0435 \u043c\u0430\u043b\u043e. \u0414\u0430\u0436\u0435 Windows \u0438 \u0431\u0440\u0430\u0443\u0437\u0435\u0440\u0443 \u0431\u0443\u0434\u0435\u0442 \u0442\u044f\u0436\u0435\u043b\u043e.")
    if "ddr5" in n and ("pentium" in n or "\u043f\u0435\u043d\u0442\u0438\u0443\u043c" in n):
        issues.append("DDR5 \u0441 Pentium \u0437\u0430\u0432\u0438\u0441\u0438\u0442 \u043e\u0442 \u0442\u043e\u0447\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438 CPU \u0438 \u043f\u043b\u0430\u0442\u044b; \u043c\u043d\u043e\u0433\u0438\u0435 \u0441\u0442\u0430\u0440\u044b\u0435 Pentium \u0441 DDR5 \u043d\u0435\u0441\u043e\u0432\u043c\u0435\u0441\u0442\u0438\u043c\u044b.")
    if "hdd" in n or "\u0448\u0434\u0434" in n:
        issues.append("HDD \u043d\u0430 64 \u0413\u0411 - \u043e\u0447\u0435\u043d\u044c \u043c\u0430\u043b\u043e \u0438 \u043c\u0435\u0434\u043b\u0435\u043d\u043d\u043e. \u041b\u0443\u0447\u0448\u0435 SSD \u0445\u043e\u0442\u044f \u0431\u044b 500 \u0413\u0411.")
    if "5090" in n:
        good.append("RTX 5090 - \u043e\u0447\u0435\u043d\u044c \u043c\u043e\u0449\u043d\u0430\u044f \u0432\u0438\u0434\u0435\u043e\u043a\u0430\u0440\u0442\u0430.")
        issues.append("\u0414\u043b\u044f RTX 5090 \u043d\u0443\u0436\u043d\u044b \u0441\u0438\u043b\u044c\u043d\u044b\u0439 CPU, \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u0430\u044f \u043f\u043b\u0430\u0442\u0430, \u0445\u043e\u0440\u043e\u0448\u0438\u0439 \u0431\u043b\u043e\u043a \u043f\u0438\u0442\u0430\u043d\u0438\u044f \u0438 \u043f\u0440\u043e\u0434\u0443\u0432.")

    if not issues:
        return "\u041f\u043e \u043e\u043f\u0438\u0441\u0430\u043d\u0438\u044e \u043c\u0430\u043b\u043e \u0434\u0430\u043d\u043d\u044b\u0445. \u0421\u043a\u0438\u043d\u044c CPU, GPU, RAM, \u0434\u0438\u0441\u043a, \u043f\u043b\u0430\u0442\u0443 \u0438 \u0431\u043b\u043e\u043a \u043f\u0438\u0442\u0430\u043d\u0438\u044f - \u043e\u0446\u0435\u043d\u044e \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u043e."

    answer = "\u041a\u043e\u0440\u043e\u0442\u043a\u043e: \u0441\u0431\u043e\u0440\u043a\u0430 \u043f\u043b\u043e\u0445\u043e \u0441\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u0430.\n\n"
    if good:
        answer += "\u041f\u043b\u044e\u0441: " + " ".join(good) + "\n\n"
    answer += "\u041f\u0440\u043e\u0431\u043b\u0435\u043c\u044b:\n- " + "\n- ".join(issues)
    answer += "\n\n\u0418\u0442\u043e\u0433: \u0442\u0430\u043a \u0441\u043e\u0431\u0438\u0440\u0430\u0442\u044c \u043d\u0435 \u0441\u0442\u043e\u0438\u0442. \u041b\u0443\u0447\u0448\u0435 \u0432\u0437\u044f\u0442\u044c \u0431\u043e\u043b\u0435\u0435 \u0441\u0438\u043b\u044c\u043d\u044b\u0439 CPU, 16-32 \u0413\u0411 RAM, SSD \u0438 \u0441\u043e\u0432\u043c\u0435\u0441\u0442\u0438\u043c\u0443\u044e \u043f\u043b\u0430\u0442\u0443."
    return answer


def completion_messages(messages, max_tokens=900, temperature=0.7):
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    api_key = groq_key or openai_key
    if not api_key:
        return None

    url = GROQ_URL if groq_key else OPENAI_URL
    model = os.environ.get("GROQ_MODEL" if groq_key else "AI_MODEL")
    if not model:
        model = "openai/gpt-oss-120b" if groq_key else "gpt-4.1-mini"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tool_choice": "none",
    }
    parsed = post_json(
        url,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    return parsed["choices"][0]["message"]["content"].strip()


def ai_reply(text, rag_context="", web_context=""):
    today = datetime.now().strftime("%Y-%m-%d")
    system = (
        "\u0422\u044b \u0443\u043c\u043d\u044b\u0439 Telegram-\u0431\u043e\u0442. "
        "\u041e\u0442\u0432\u0435\u0447\u0430\u0439 \u043f\u043e-\u0440\u0443\u0441\u0441\u043a\u0438, \u0441\u043c\u0435\u043b\u043e, \u043f\u043e \u0434\u0435\u043b\u0443, \u0431\u0435\u0437 \u0442\u0443\u043f\u044b\u0445 \u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d\u0438\u0439. "
        f"\u0421\u0435\u0433\u043e\u0434\u043d\u044f: {today}. \u0415\u0441\u043b\u0438 \u0443\u043f\u043e\u043c\u0438\u043d\u0430\u0435\u0448\u044c \u0442\u0435\u043a\u0443\u0449\u0443\u044e \u0434\u0430\u0442\u0443, \u043f\u0438\u0448\u0438 \u0438\u043c\u0435\u043d\u043d\u043e {today}, \u043d\u0435 \u0437\u0430\u043c\u0435\u043d\u044f\u0439 \u0435\u0435 \u0434\u0440\u0443\u0433\u043e\u0439 \u0434\u0430\u0442\u043e\u0439. "
        "\u0423 \u0442\u0435\u0431\u044f \u043d\u0435\u0442 \u0438\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442\u043e\u0432, \u0444\u0443\u043d\u043a\u0446\u0438\u0439 \u0438 tool calls. \u041d\u0438\u043a\u043e\u0433\u0434\u0430 \u043d\u0435 \u0432\u044b\u0437\u044b\u0432\u0430\u0439 web.run, browser, search \u0438\u043b\u0438 \u0434\u0440\u0443\u0433\u0438\u0435 tools. "
        "\u0412\u0441\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0443\u0436\u0435 \u0434\u0430\u043d\u044b \u0432 \u0442\u0435\u043a\u0441\u0442\u0435 \u043d\u0438\u0436\u0435; \u0440\u0430\u0431\u043e\u0442\u0430\u0439 \u0442\u043e\u043b\u044c\u043a\u043e \u0441 \u043d\u0438\u043c\u0438. "
        "\u0415\u0441\u043b\u0438 \u0435\u0441\u0442\u044c RAG/Web-\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442, \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 \u0435\u0433\u043e, \u043d\u043e \u043d\u0435 \u0432\u044b\u0434\u0443\u043c\u044b\u0432\u0430\u0439 \u0444\u0430\u043a\u0442\u044b. "
        "\u0414\u043b\u044f \u0441\u0432\u0435\u0436\u0438\u0445 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439, \u0446\u0435\u043d, \u043a\u0443\u0440\u0441\u043e\u0432, \u0434\u0430\u0442 \u0438 \u0442\u0435\u043a\u0443\u0449\u0438\u0445 \u0441\u043e\u0431\u044b\u0442\u0438\u0439 \u043e\u043f\u0438\u0440\u0430\u0439\u0441\u044f \u043d\u0430 Web context, \u0430 \u043d\u0435 \u043d\u0430 \u043f\u0430\u043c\u044f\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u0438. "
        "\u0414\u043b\u0438\u043d\u043d\u044b\u0435 \u0440\u0430\u0441\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f \u0441\u043a\u0440\u044b\u0432\u0430\u0439, \u0434\u0430\u0432\u0430\u0439 \u0438\u0442\u043e\u0433\u043e\u0432\u044b\u0439 \u043e\u0442\u0432\u0435\u0442."
    )
    context = ""
    if rag_context:
        context += f"\n\nRAG context:\n{rag_context}"
    if web_context:
        context += f"\n\nWeb context:\n{web_context}"

    plan = completion_messages(
        [
            {"role": "system", "content": system + "\n\u042d\u0442\u043e chain-1: \u043a\u0440\u0430\u0442\u043a\u043e \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0438, \u0447\u0442\u043e \u043d\u0443\u0436\u043d\u043e \u0443\u0447\u0435\u0441\u0442\u044c. \u041d\u0435 \u043e\u0442\u0432\u0435\u0447\u0430\u0439 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044e."},
            {"role": "user", "content": text + context},
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return completion_messages(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"\u0412\u043e\u043f\u0440\u043e\u0441: {text}{context}\n\nChain-1 notes:\n{plan}\n\n\u0414\u0430\u0439 \u0444\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0439 \u043e\u0442\u0432\u0435\u0442."},
        ],
        max_tokens=1100,
        temperature=0.7,
    )


def local_reply(text, first_name):
    normalized = normalize_text(text)

    if text == "/start":
        return RU["start"].format(name=first_name)
    if text == "/help":
        return RU["help"]
    if text.startswith("/echo"):
        return text.removeprefix("/echo").strip() or "\u041d\u0430\u043f\u0438\u0448\u0438 \u0442\u0435\u043a\u0441\u0442 \u043f\u043e\u0441\u043b\u0435 /echo."
    if text.startswith("/"):
        return "\u0422\u0430\u043a\u043e\u0439 \u043a\u043e\u043c\u0430\u043d\u0434\u044b \u043d\u0435\u0442. \u041f\u0440\u043e\u0441\u0442\u043e \u043d\u0430\u043f\u0438\u0448\u0438 \u0432\u043e\u043f\u0440\u043e\u0441 \u0442\u0435\u043a\u0441\u0442\u043e\u043c."

    math_answer = try_math(text)
    if math_answer:
        return math_answer

    pc_answer = evaluate_pc_build(text)
    if pc_answer:
        return pc_answer

    if any(word in normalized for word in ["\u043f\u0440\u0438\u0432\u0435\u0442", "\u0441\u0430\u043b\u0430\u043c", "hello", "hi"]):
        return "\u041f\u0440\u0438\u0432\u0435\u0442! \u041f\u0438\u0448\u0438 \u0432\u043e\u043f\u0440\u043e\u0441, \u0440\u0430\u0437\u0431\u0435\u0440\u0435\u043c."

    if "\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u043f\u043a" in normalized or "\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u043a\u043e\u043c\u043f\u044c\u044e\u0442\u0435\u0440" in normalized:
        return (
            "\u041f\u041a - \u044d\u0442\u043e \u043f\u0435\u0440\u0441\u043e\u043d\u0430\u043b\u044c\u043d\u044b\u0439 \u043a\u043e\u043c\u043f\u044c\u044e\u0442\u0435\u0440: \u0443\u0441\u0442\u0440\u043e\u0439\u0441\u0442\u0432\u043e \u0434\u043b\u044f \u0440\u0430\u0431\u043e\u0442\u044b, \u0443\u0447\u0435\u0431\u044b, \u0438\u0433\u0440, \u0438\u043d\u0442\u0435\u0440\u043d\u0435\u0442\u0430 \u0438 \u0444\u0430\u0439\u043b\u043e\u0432. "
            "\u0413\u043b\u0430\u0432\u043d\u044b\u0435 \u0447\u0430\u0441\u0442\u0438: CPU, RAM, \u0434\u0438\u0441\u043a SSD/HDD, GPU, \u043c\u0430\u0442\u0435\u0440\u0438\u043d\u0441\u043a\u0430\u044f \u043f\u043b\u0430\u0442\u0430, \u0431\u043b\u043e\u043a \u043f\u0438\u0442\u0430\u043d\u0438\u044f \u0438 \u043a\u043e\u0440\u043f\u0443\u0441."
        )

    return RU["unknown"]


def answer_with_rag(query):
    results = rag_search(query)
    context = format_rag_context(results)
    if not context:
        return "\u0412 RAG \u043f\u043e\u043a\u0430 \u043d\u0435\u0442 \u043f\u043e\u0434\u0445\u043e\u0434\u044f\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445. \u041e\u0442\u043f\u0440\u0430\u0432\u044c PDF \u0432 \u0447\u0430\u0442, \u0438 \u044f \u0435\u0433\u043e \u043f\u0440\u043e\u0438\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u0443\u044e."
    try:
        answer = ai_reply(query, rag_context=context)
        if answer:
            return answer
    except Exception as error:
        print(f"RAG AI error: {error}")
    return "\u041d\u0430\u0448\u0435\u043b \u0432 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445:\n\n" + context[:3000]


def answer_with_web(query):
    try:
        results = web_search(enrich_web_query(query))
    except Exception as error:
        return f"\u041d\u0435 \u0441\u043c\u043e\u0433 \u043d\u0430\u0439\u0442\u0438 \u0432 web: {error}"
    context = format_web_context(results)
    if not context:
        return "\u0412 web \u043d\u0438\u0447\u0435\u0433\u043e \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u043d\u0435 \u043d\u0430\u0448\u0435\u043b."
    try:
        answer = ai_reply(query, web_context=context)
        if answer:
            return answer + "\n\n\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438:\n" + "\n".join(
                f"- {item['url']}" for item in results if item.get("url")
            )
    except Exception as error:
        print(f"Web AI error: {error}")
    return "\u041d\u0430\u0448\u0435\u043b:\n\n" + context[:3000]


def should_use_web(text):
    if os.environ.get("AUTO_WEB", "1").strip().lower() not in {"0", "false", "no", "off"}:
        return True

    n = normalize_text(text)
    return any(
        word in n
        for word in [
            "\u0432 \u0438\u043d\u0442\u0435\u0440\u043d\u0435\u0442\u0435",
            "\u043d\u0430\u0439\u0434\u0438",
            "\u043f\u043e\u0438\u0449\u0438",
            "\u0441\u0432\u0435\u0436",
            "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
            "\u043d\u043e\u0432\u043e\u0441\u0442",
            "\u043a\u0443\u0440\u0441",
        ]
    )


def is_quick_smalltalk(text):
    n = normalize_text(text)
    return n in {
        "\u043f\u0440\u0438\u0432\u0435\u0442",
        "\u0441\u0430\u043b\u0430\u043c",
        "\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439",
        "\u0441\u043f\u0430\u0441\u0438\u0431\u043e",
        "\u0440\u0430\u0445\u043c\u0435\u0442",
        "hi",
        "hello",
    }


def build_reply(message):
    text = (message.get("text") or "").strip()
    first_name = (message.get("from") or {}).get("first_name") or RU["friend"]

    if not text:
        return "\u042f \u043f\u043e\u043a\u0430 \u043f\u043e\u043d\u0438\u043c\u0430\u044e \u0442\u043e\u043b\u044c\u043a\u043e \u0442\u0435\u043a\u0441\u0442."

    if text in {"/start", "/help"} or text.startswith("/echo"):
        return local_reply(text, first_name)

    if text.startswith("/rag"):
        query = text.removeprefix("/rag").strip()
        return answer_with_rag(query or "\u0447\u0442\u043e \u0432 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445?")

    if text.startswith("/web"):
        query = text.removeprefix("/web").strip()
        return answer_with_web(query or "\u043d\u043e\u0432\u043e\u0441\u0442\u0438")

    if text.startswith("/"):
        return local_reply(text, first_name)

    math_answer = try_math(text)
    if math_answer:
        return math_answer

    if is_quick_smalltalk(text):
        return local_reply(text, first_name)

    rag_context = format_rag_context(rag_search(text, limit=3))
    web_context = ""
    if should_use_web(text):
        try:
            web_context = format_web_context(web_search(enrich_web_query(text), limit=4))
        except Exception as error:
            print(f"Auto web error: {error}")

    try:
        answer = ai_reply(text, rag_context=rag_context, web_context=web_context)
        if answer:
            return answer
    except Exception as error:
        print(f"AI error: {error}")
        if web_context:
            return "\u042f \u043d\u0430\u0448\u0435\u043b \u0441\u0432\u0435\u0436\u0438\u0435 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438, \u043d\u043e AI-\u043e\u0442\u0432\u0435\u0442 \u0441\u0435\u0439\u0447\u0430\u0441 \u0441\u0431\u043e\u0439\u043d\u0443\u043b. \u0412\u043e\u0442 \u0447\u0442\u043e \u043d\u0430\u0448\u0435\u043b:\n\n" + web_context[:3200]

    return local_reply(text, first_name)


def handle_document_message(token, message):
    document = message.get("document")
    if not document:
        return None

    filename = document.get("file_name") or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        return "\u041f\u043e\u043a\u0430 RAG \u0438\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u0443\u0435\u0442 \u0442\u043e\u043b\u044c\u043a\u043e PDF."

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    path = DOCS_DIR / f"{int(time.time())}_{safe_name}"
    download_file(telegram_file_url(token, document["file_id"]), path)
    text = extract_pdf_text(path)
    if not text:
        return "\u042f \u0441\u043a\u0430\u0447\u0430\u043b PDF, \u043d\u043e \u043d\u0435 \u0441\u043c\u043e\u0433 \u0438\u0437\u0432\u043b\u0435\u0447\u044c \u0442\u0435\u043a\u0441\u0442. \u0412\u043e\u0437\u043c\u043e\u0436\u043d\u043e, \u044d\u0442\u043e \u0441\u043a\u0430\u043d-\u043a\u0430\u0440\u0442\u0438\u043d\u043a\u0430."

    count = add_document_to_rag(filename, text)
    supabase_note = "\u0418 \u0432 Supabase." if os.environ.get("SUPABASE_URL") else "\u041f\u043e\u043a\u0430 \u0432 \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0439 RAG-store; Supabase \u0432\u043a\u043b\u044e\u0447\u0438\u0442\u0441\u044f, \u043a\u043e\u0433\u0434\u0430 \u0434\u043e\u0431\u0430\u0432\u0438\u0448\u044c SUPABASE_URL \u0438 \u043a\u043b\u044e\u0447."
    return f"PDF \u0433\u043e\u0442\u043e\u0432: {filename}\n\u0418\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043e \u0447\u0430\u043d\u043a\u043e\u0432: {count}\n{supabase_note}\n\n\u0422\u0435\u043f\u0435\u0440\u044c \u0441\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u0439: /rag \u0442\u0432\u043e\u0439 \u0432\u043e\u043f\u0440\u043e\u0441"


def main():
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env or environment variables.")

    offset = None
    pending_updates = telegram_request(token, "getUpdates", payload={"timeout": 0}, timeout=10)
    if pending_updates:
        offset = pending_updates[-1]["update_id"] + 1

    print("Bot is running. Press Ctrl+C to stop.")

    while True:
        try:
            payload = {"timeout": 30}
            if offset is not None:
                payload["offset"] = offset

            updates = telegram_request(token, "getUpdates", payload=payload, timeout=40)
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue

                chat_id = message["chat"]["id"]
                document_reply = handle_document_message(token, message)
                if document_reply:
                    send_message(token, chat_id, document_reply)
                else:
                    send_message(token, chat_id, build_reply(message))

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"Network error: {error}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as error:
            print(f"Error: {error}. Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    main()
