"""Configuration constants and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
_parents = Path(__file__).resolve().parents
REPO_ROOT = _parents[2] if len(_parents) > 2 else PROJECT_ROOT

load_dotenv(REPO_ROOT / ".env")

RAW_DIR = REPO_ROOT / "raw"
WIKI_DIR = REPO_ROOT / "wiki"
OUTPUT_DIR = REPO_ROOT / "output"
# Crawled from support.superex.com; see scripts/crawl_help_center.py.
HELP_CENTER_DIR = RAW_DIR / "帮助中心"

KB_SOURCES = {
    "raw_customer_service": RAW_DIR / "客服",
    "raw_official_tutorial": RAW_DIR / "官方教程",
    "raw_help_center": HELP_CENTER_DIR,
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

# How much of a retrieved chunk reaches the model. This was 1200, which silently
# clipped 51% of the corpus — every long chunk lost its tail. The VIP fee tables
# run 1364 and 1385 characters, so VIP 9 fell off the end and the bot reported
# that it only had "VIP levels 0 through 8" while the data was sitting in the
# index. Truncating a table mid-row is worse than a longer prompt: the model is
# told never to invent a figure, so a clipped row becomes a refusal.
#
# 3600 clears the longest chunk in the corpus (3454). Worst case is TOP_K * this,
# which every model in use handles comfortably.
CONTEXT_SNIPPET_CHARS = 3600

ANSWER_MODEL = "llama-3.3-70b-versatile"
ROUTER_MODEL = "llama-3.1-8b-instant"
DEEPSEEK_MODEL = "deepseek-chat"

# Real-time price lookup. Price questions bypass the static index entirely — the
# index once answered "BTC = 100000" from a help-center worked example — and hit
# a live feed instead. Primary source is SuperEx's own public market API, so the
# bot quotes exactly what users see on the platform and covers all ~710 listed
# pairs including ET. CoinGecko and Binance are reliability fallbacks only.
SUPEREX_API_BASE = "https://api.superex.com"
SUPEREX_SUMMARY_PATH = "/spot/public/v3/summary"  # all pairs, no auth
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BINANCE_BASE = "https://api.binance.com"
PRICE_CACHE_TTL = 45.0  # seconds; a support bot does not need tick-level freshness
PRICE_HTTP_TIMEOUT = 8.0  # the full-summary payload takes ~2.5s; leave headroom
SUPEREX_MARKETS_URL = "https://www.superex.com/markets"

TELEGRAM_MAX_MESSAGE_LEN = 4000
# Split the markdown source below the hard limit: converting to HTML adds tags,
# so a 4000-char source can exceed Telegram's 4096-char ceiling once rendered.
TELEGRAM_SPLIT_LEN = 3500

# Retrieval scoring. Chunks whose body is just "see this link" match support
# questions lexically but answer nothing, so they must not crowd out real
# content — the penalty demotes them without hiding them entirely.
LINK_ONLY_PENALTY = 0.6
# Social-media marketing copy is not a support answer; damp it below the FAQ
# and tutorial sources. Types not listed here default to 1.0.
DOC_TYPE_WEIGHTS: dict[str, float] = {"cs-script": 0.85}

# The Help Center is crawled in 11 locales, and the same fact therefore exists as
# 11 separate articles with distinct ids — text dedup cannot fold them together.
# When the user's language is one we crawled, a same-language article is known to
# exist, so demote the other translations instead of letting them eat TOP_K.
# Languages outside this set (tr, ar, id, …) are exempt: cross-lingual retrieval
# is the only way those users get an answer at all.
HELP_CENTER_LANGS: frozenset[str] = frozenset(
    {"en", "zh", "ru", "fa", "vi", "es", "fr", "ja", "ko", "pt", "uk"}
)
LANG_MISMATCH_PENALTY = 0.75

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
