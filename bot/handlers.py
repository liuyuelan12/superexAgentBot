"""Telegram handlers: /ask, /start, /help, generic on_message."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from telegram import Message, Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import ContextTypes

from config import (
    ENABLE_WEAK_TRIGGER,
    MAX_CTX_TURNS,
    QA_LOG_PATH,
    SIM_THRESHOLD,
    TELEGRAM_MAX_MESSAGE_LEN,
    TOP_K,
    RuntimeConfig,
)
from kb.retriever import Hit, Retriever
from llm.client import ChatMessage, LLMClient
from llm.prompts import SYSTEM_ANSWER, refusal_for

from .context import build_history
from .router import RouterDecision, route

logger = logging.getLogger(__name__)


def _mentions_bot(msg: Message, bot_username: Optional[str]) -> bool:
    if not bot_username or not msg.text or not msg.entities:
        return False
    tag = f"@{bot_username.lower()}"
    for ent in msg.entities:
        if ent.type == "mention":
            slice_ = msg.text[ent.offset : ent.offset + ent.length].lower()
            if slice_ == tag:
                return True
    return False


def _strip_mention(text: str, bot_username: Optional[str]) -> str:
    if not bot_username:
        return text
    return re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE).strip()


def _format_hits(hits: list[Hit]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        if hit.metadata.get("type") == "wiki":
            cite = "[[%s]]" % hit.basename().rsplit(".", 1)[0]
        else:
            cite = hit.basename()
        text = hit.text.strip()
        if len(text) > 1200:
            text = text[:1200] + " …"
        blocks.append(f"[{i}] (source: {cite}) [{hit.lang or '-'}] {text}")
    return "\n\n".join(blocks)


def _format_hit_paths(hits: list[Hit]) -> list[dict]:
    return [
        {
            "doc_id": h.doc_id,
            "source": h.source,
            "section": h.section,
            "lang": h.lang,
            "vec_sim": round(h.vec_sim, 4),
            "bm25_norm": round(h.bm25_norm, 4),
            "score": round(h.score, 4),
        }
        for h in hits
    ]


async def _reply_long(msg: Message, text: str) -> None:
    text = text.strip()
    if not text:
        return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LEN:
        await msg.reply_text(text, disable_web_page_preview=True)
        return
    parts: list[str] = []
    remaining = text
    while remaining:
        cut = remaining[:TELEGRAM_MAX_MESSAGE_LEN]
        if len(remaining) > TELEGRAM_MAX_MESSAGE_LEN:
            split_at = cut.rfind("\n\n")
            if split_at < TELEGRAM_MAX_MESSAGE_LEN // 2:
                split_at = cut.rfind("\n")
            if split_at < TELEGRAM_MAX_MESSAGE_LEN // 2:
                split_at = cut.rfind(" ")
            if split_at > 0:
                cut = remaining[:split_at]
        parts.append(cut.rstrip())
        remaining = remaining[len(cut):].lstrip()
    total = len(parts)
    for i, part in enumerate(parts, 1):
        await msg.reply_text(f"{part}\n\n({i}/{total})", disable_web_page_preview=True)


def _log_qa(record: dict) -> None:
    QA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QA_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class AskHandler:
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        runtime: RuntimeConfig,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._runtime = runtime

    def _chat_allowed(self, chat_id: int, chat_type: str) -> bool:
        if chat_type == ChatType.PRIVATE:
            return True
        if not self._runtime.allowed_chats:
            return True
        return chat_id in self._runtime.allowed_chats

    async def _answer(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: str,
        triggered_by: str,
        decision: Optional[RouterDecision] = None,
    ) -> None:
        msg = update.effective_message
        if msg is None:
            return
        query = (query or "").strip()
        if not query:
            return
        bot_id = context.bot.id

        await context.bot.send_chat_action(
            chat_id=msg.chat.id, action=ChatAction.TYPING
        )
        history = build_history(msg, bot_id)

        if decision is None:
            decision = await route(self._llm, query, history, assume_question=True)

        start = time.monotonic()
        hits = self._retriever.search(
            decision.rewritten_query or query,
            top_k=TOP_K,
            lang_boost=decision.lang,
        )
        top_vec = hits[0].vec_sim if hits else 0.0
        top_score = hits[0].score if hits else 0.0

        logger.info(
            "ask trigger=%s lang=%s top_vec=%.3f q=%r rew=%r hits=%s",
            triggered_by,
            decision.lang,
            top_vec,
            query[:120],
            (decision.rewritten_query or "")[:120],
            [
                f"{h.basename()[:30]}|sec={h.section[:25]}|{h.lang}|s={h.score:.2f}/v={h.vec_sim:.2f}"
                for h in hits[:5]
            ],
        )

        record = {
            "ts": int(time.time()),
            "chat_id": msg.chat.id,
            "chat_type": str(msg.chat.type),
            "user_id": msg.from_user.id if msg.from_user else None,
            "triggered_by": triggered_by,
            "query": query,
            "rewritten": decision.rewritten_query,
            "lang": decision.lang,
            "top_vec": round(top_vec, 4),
            "top_score": round(top_score, 4),
            "hits": _format_hit_paths(hits),
        }

        if not hits or top_vec < SIM_THRESHOLD:
            answer = refusal_for(decision.lang)
            await _reply_long(msg, answer)
            record.update(
                {
                    "answered": False,
                    "reason": "below_threshold" if hits else "no_hits",
                    "latency_ms": int((time.monotonic() - start) * 1000),
                    "fallback_llm": False,
                }
            )
            _log_qa(record)
            return

        context_block = _format_hits(hits)
        system_text = SYSTEM_ANSWER.format(lang=decision.lang, context=context_block)
        messages = [ChatMessage(role="system", content=system_text)]
        for turn in history[-MAX_CTX_TURNS:]:
            messages.append(ChatMessage(role=turn.role, content=turn.text))
        messages.append(ChatMessage(role="user", content=query))

        try:
            result = await self._llm.chat(
                messages,
                purpose="answer",
                temperature=0.2,
                max_tokens=1024,
            )
            answer = (result.text or "").strip() or refusal_for(decision.lang)
            fallback_llm = result.fallback
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM call failed entirely: %s", exc)
            answer = refusal_for(decision.lang)
            fallback_llm = False

        await _reply_long(msg, answer)
        record.update(
            {
                "answered": True,
                "latency_ms": int((time.monotonic() - start) * 1000),
                "fallback_llm": fallback_llm,
            }
        )
        _log_qa(record)

    async def on_ask(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        if msg is None or msg.from_user is None:
            return
        if msg.from_user.is_bot:
            return
        if not self._chat_allowed(msg.chat.id, msg.chat.type):
            return

        if context.args:
            query = " ".join(context.args).strip()
        else:
            query = (msg.text or "").lstrip()
            if query.lower().startswith("/ask"):
                query = query[4:].lstrip()
                if query.startswith("@" + (context.bot.username or "")):
                    query = query[len(context.bot.username or "") + 1 :].lstrip()
        if not query:
            await msg.reply_text("Usage: /ask <your question>")
            return
        await self._answer(update, context, query, triggered_by="ask")

    async def on_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        if msg is None or msg.from_user is None:
            return
        if msg.from_user.is_bot:
            return
        text = (msg.text or "").strip()
        if not text:
            return
        if text.startswith("/"):
            return
        if not self._chat_allowed(msg.chat.id, msg.chat.type):
            return

        bot_id = context.bot.id
        bot_username = context.bot.username

        is_private = msg.chat.type == ChatType.PRIVATE
        is_mention = _mentions_bot(msg, bot_username)
        is_reply_to_bot = bool(
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.id == bot_id
        )

        if is_private or is_mention or is_reply_to_bot:
            query = _strip_mention(text, bot_username) if is_mention else text
            await self._answer(update, context, query, triggered_by="strong")
            return

        if not ENABLE_WEAK_TRIGGER:
            return

        history = build_history(msg, bot_id)
        decision = await route(self._llm, text, history)
        if not decision.is_question:
            return
        await self._answer(
            update, context, text, triggered_by="weak", decision=decision
        )


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None:
        return
    await msg.reply_text(
        "你好！我是 SuperEx 的 AI 客服。\n\n"
        "• 私聊我直接发问\n"
        "• 群里 @我 或者 reply 我的消息\n"
        "• 也可以用 /ask <问题>\n\n"
        "我支持中、英、波斯、俄、越南、西班牙、法等语言。"
    )


async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None:
        return
    await msg.reply_text(
        "命令:\n"
        "/ask <question> — 询问\n"
        "/start — 简介\n"
        "/help — 本帮助\n\n"
        "群里使用 @我 或 reply 我的消息也会触发。"
    )
