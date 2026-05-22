"""Configuration constants and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")

RAW_DIR = REPO_ROOT / "raw"
WIKI_DIR = REPO_ROOT / "wiki"

KB_SOURCES = {
    "raw_customer_service": RAW_DIR / "客服",
    "raw_official_tutorial": RAW_DIR / "官方教程",
    "wiki": WIKI_DIR,
}

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25.pkl"
INDEX_META_PATH = DATA_DIR / "index_meta.json"
QA_LOG_PATH = DATA_DIR / "qa_log.jsonl"

EMBED_MODEL_NAME = "BAAI/bge-m3"
CHROMA_COLLECTION = "superex_kb"

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

TOP_K = 5
SIM_THRESHOLD = 0.50
HYBRID_VECTOR_WEIGHT = 0.7
MAX_CTX_TURNS = 5

ANSWER_MODEL = "llama-3.3-70b-versatile"
ROUTER_MODEL = "llama-3.1-8b-instant"
DEEPSEEK_MODEL = "deepseek-chat"

TELEGRAM_MAX_MESSAGE_LEN = 4000

ENABLE_WEAK_TRIGGER = False  # P0: off; P1: on


@dataclass(frozen=True)
class Secrets:
    bot_token: str
    groq_api_key: str
    deepseek_api_key: str | None


def load_secrets() -> Secrets:
    bot_token = os.environ.get("BOT_TOKEN", "").strip()
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or None
    if not bot_token:
        raise RuntimeError("BOT_TOKEN not set in .env")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    return Secrets(
        bot_token=bot_token,
        groq_api_key=groq_api_key,
        deepseek_api_key=deepseek_api_key,
    )


@dataclass(frozen=True)
class RuntimeConfig:
    allowed_chats: frozenset[int] = field(default_factory=frozenset)
    admin_user_ids: frozenset[int] = field(default_factory=frozenset)


def load_runtime_config() -> RuntimeConfig:
    allowed = os.environ.get("AGENT_BOT_ALLOWED_CHATS", "").strip()
    admins = os.environ.get("AGENT_BOT_ADMINS", "").strip()
    return RuntimeConfig(
        allowed_chats=frozenset(int(x) for x in allowed.split(",") if x.strip()),
        admin_user_ids=frozenset(int(x) for x in admins.split(",") if x.strip()),
    )
