import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "assistant.db"
PUBLIC_DIR = BASE_DIR / "public"

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Smart Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    title: str | None = None


class UpdateSettingsRequest(BaseModel):
    title: str | None = None
    persona: str | None = None
    memory: str | None = None


class ChatRequest(BaseModel):
    conversationId: str
    message: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              persona TEXT NOT NULL,
              memory TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


def provider_config() -> dict:
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    presets = {
        "openai": {"baseUrl": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
        "deepseek": {"baseUrl": "https://api.deepseek.com", "model": "deepseek-chat"},
        "doubao": {"baseUrl": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-seed-1-6"},
        "custom": {"baseUrl": os.getenv("AI_BASE_URL", ""), "model": os.getenv("AI_MODEL", "")},
    }
    preset = presets.get(provider, presets["custom"])
    api_key = os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    return {
        "provider": provider,
        "baseUrl": os.getenv("AI_BASE_URL", preset["baseUrl"]).rstrip("/"),
        "model": os.getenv("AI_MODEL", preset["model"]),
        "configured": bool(api_key),
        "apiKey": api_key,
    }


def ensure_default_conversation() -> str:
    with connect() as db:
        row = db.execute("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1").fetchone()
        if row:
            return row["id"]

        conversation_id = str(uuid.uuid4())
        created_at = utc_now()
        db.execute(
            """
            INSERT INTO conversations (id, title, persona, memory, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                "默认对话",
                "你是一个专业、清晰、行动导向的 AI 应用开发助手。",
                "用户正在准备 AI 应用开发工程师方向项目。",
                created_at,
                created_at,
            ),
        )
        return conversation_id


def get_conversation(conversation_id: str) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return row_to_dict(row)


def get_messages(conversation_id: str, limit: int = 20) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT role, content, created_at FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]


def insert_message(conversation_id: str, role: str, content: str) -> None:
    with connect() as db:
        db.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), conversation_id, role, content, utc_now()),
        )
        db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (utc_now(), conversation_id))


def build_messages(conversation: dict, history: list[dict], user_message: str) -> list[dict]:
    system_prompt = "\n".join(
        [
            conversation["persona"],
            "",
            "长期记忆：",
            conversation["memory"] or "暂无。",
            "",
            "要求：回答要自然、具体、可执行；涉及代码时优先给出能落地的实现建议。",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        *[{"role": item["role"], "content": item["content"]} for item in history],
        {"role": "user", "content": user_message},
    ]


def call_model(messages: list[dict]) -> str | None:
    config = provider_config()
    if not config["configured"] or not config["baseUrl"] or not config["model"]:
        return None

    payload = json.dumps(
        {"model": config["model"], "messages": messages, "temperature": 0.6},
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        f"{config['baseUrl']}/chat/completions",
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {config['apiKey']}", "Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Model request failed: {exc.code} {detail}") from exc

    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def local_reply(message: str) -> str:
    lower = message.lower()
    if "rag" in lower or "知识库" in message:
        return "可以。RAG 项目建议按“文档解析 -> 文本分块 -> 检索召回 -> Prompt 组装 -> 引用展示 -> 评估优化”这条链路写，面试时重点讲清楚如何降低幻觉、如何调 chunk size/top-k、如何处理检索为空。"
    if "简历" in message or "项目" in message:
        return "建议把项目描述写成：业务问题、技术方案、你的具体实现、可验证结果。AI 应用岗位尤其看重模型接入、工程闭环、异常处理、部署和评估，而不是只写“调用了 API”。"
    return f"我已理解你的问题：“{message}”。当前没有配置真实 API Key，所以返回本地兜底建议：把目标拆成输入、处理、输出、验证四步，会更容易落地。"


@app.on_event("startup")
def startup() -> None:
    init_db()
    ensure_default_conversation()


@app.get("/api/config")
def config() -> dict:
    current = provider_config()
    return {
        "provider": current["provider"],
        "baseUrl": current["baseUrl"],
        "model": current["model"],
        "configured": current["configured"],
    }


@app.get("/api/conversations")
def conversations() -> dict:
    ensure_default_conversation()
    with connect() as db:
        rows = db.execute(
            """
            SELECT id, title, persona, memory, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return {"conversations": [dict(row) for row in rows]}


@app.post("/api/conversations")
def create_conversation(payload: CreateConversationRequest) -> dict:
    title = (payload.title or "新对话").strip()[:60] or "新对话"
    conversation_id = str(uuid.uuid4())
    created_at = utc_now()
    with connect() as db:
        db.execute(
            """
            INSERT INTO conversations (id, title, persona, memory, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                title,
                "你是一个专业、清晰、行动导向的 AI 应用开发助手。",
                "",
                created_at,
                created_at,
            ),
        )
    return {"id": conversation_id}


@app.get("/api/conversations/{conversation_id}/messages")
def messages(conversation_id: str) -> dict:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在。")
    return {"conversation": conversation, "messages": get_messages(conversation_id, 200)}


@app.patch("/api/conversations/{conversation_id}/settings")
def update_settings(conversation_id: str, payload: UpdateSettingsRequest) -> dict:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在。")

    title = (payload.title if payload.title is not None else conversation["title"]).strip()[:80] or "未命名对话"
    persona = (payload.persona if payload.persona is not None else conversation["persona"]).strip()
    memory = (payload.memory if payload.memory is not None else conversation["memory"]).strip()

    with connect() as db:
        db.execute(
            """
            UPDATE conversations SET title = ?, persona = ?, memory = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, persona, memory, utc_now(), conversation_id),
        )
    return {"ok": True}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> JSONResponse:
    conversation_id = payload.conversationId.strip()
    content = payload.message.strip()
    if not conversation_id or not content:
        raise HTTPException(status_code=400, detail="conversationId 和 message 不能为空。")

    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在。")

    insert_message(conversation_id, "user", content)
    history = get_messages(conversation_id, 18)
    if history and history[-1]["role"] == "user" and history[-1]["content"] == content:
        history = history[:-1]

    try:
        model_answer = call_model(build_messages(conversation, history, content))
        answer = model_answer or local_reply(content)
        insert_message(conversation_id, "assistant", answer)
        return JSONResponse({"answer": answer, "mode": "llm" if model_answer else "local"})
    except RuntimeError as exc:
        answer = local_reply(content)
        insert_message(conversation_id, "assistant", answer)
        return JSONResponse(status_code=502, content={"error": str(exc), "fallback": answer, "mode": "local"})


app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
