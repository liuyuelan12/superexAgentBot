"""LLM-based router: classifies, detects language, rewrites query."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from llm.client import ChatMessage, LLMClient
from llm.prompts import ROUTER_PROMPT

from .context import Turn, format_history

logger = logging.getLogger(__name__)

_SYS = "You output a single JSON object and nothing else. Do not wrap in markdown."


@dataclass(frozen=True)
class RouterDecision:
    is_question: bool
    lang: str
    rewritten_query: str
    raw: str
    fallback: bool


_CJK_RE = re.compile(r"[一-鿿]")
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def _heuristic_lang(text: str) -> str:
    if _CJK_RE.search(text):
        return "zh"
    if _ARABIC_RE.search(text):
        return "fa"
    if _CYRILLIC_RE.search(text):
        return "ru"
    return "en"


async def route(
    llm: LLMClient,
    text: str,
    history: list[Turn],
    *,
    assume_question: bool = False,
) -> RouterDecision:
    """Call the small router model; on failure fall back to heuristics."""
    text = text.strip()
    if not text:
        return RouterDecision(
            is_question=False,
            lang="en",
            rewritten_query="",
            raw="",
            fallback=False,
        )

    prompt = ROUTER_PROMPT.format(text=text, history=format_history(history))
    messages = [
        ChatMessage(role="system", content=_SYS),
        ChatMessage(role="user", content=prompt),
    ]
    try:
        data, result = await llm.chat_json(messages, purpose="router")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Router LLM failed: %s; using heuristics", exc)
        return RouterDecision(
            is_question=assume_question,
            lang=_heuristic_lang(text),
            rewritten_query=text,
            raw="",
            fallback=True,
        )

    is_question = bool(data.get("is_question", assume_question))
    lang = str(data.get("lang") or _heuristic_lang(text)).strip()
    rewritten = str(data.get("rewritten_query") or text).strip()
    if assume_question:
        is_question = True
    return RouterDecision(
        is_question=is_question,
        lang=lang or _heuristic_lang(text),
        rewritten_query=rewritten or text,
        raw=result.text,
        fallback=result.fallback,
    )
