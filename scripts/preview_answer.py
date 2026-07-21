"""Offline answer preview — no Telegram token required.

Runs the real retrieval + LLM path for a question and prints:
  1. the merged TOP-K hits (matching handlers._answer's dual-retrieval merge),
  2. the raw model answer,
  3. the exact HTML that would be sent to Telegram.

Usage:
    python -m scripts.preview_answer "如果我是superex vip6 我的手续费是多少"
    python -m scripts.preview_answer --lang zh "现货手续费是多少"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from bot.handlers import _format_hits  # noqa: E402
from bot.router import route  # noqa: E402
from bot.telegram_format import md_to_telegram_html  # noqa: E402
from config import MAX_CTX_TURNS, SIM_THRESHOLD, TOP_K  # noqa: E402
from kb.retriever import Retriever  # noqa: E402
from llm.client import ChatMessage, LLMClient  # noqa: E402
from llm.prompts import SYSTEM_ANSWER, refusal_for  # noqa: E402


async def preview(query: str, lang_override: str | None) -> None:
    llm = LLMClient()
    retriever = Retriever()

    decision = await route(llm, query, [], assume_question=True)
    lang = lang_override or decision.lang
    print(f"\n=== Q: {query!r}  (lang={lang}, rewritten={decision.rewritten_query!r})\n")

    hits_orig = retriever.search(query, top_k=TOP_K, lang_boost=lang)
    hits_rew = []
    if decision.rewritten_query and decision.rewritten_query.strip() != query.strip():
        hits_rew = retriever.search(decision.rewritten_query, top_k=TOP_K, lang_boost=lang)
    merged: dict = {}
    for h in hits_orig + hits_rew:
        cur = merged.get(h.doc_id)
        if cur is None or cur.score < h.score:
            merged[h.doc_id] = h
    hits = sorted(merged.values(), key=lambda h: -h.score)[:TOP_K]

    print("--- TOP hits ---")
    for i, h in enumerate(hits, 1):
        flag = " [LINK_ONLY]" if h.metadata.get("link_only") else ""
        print(
            f"[{i}] s={h.score:.3f} v={h.vec_sim:.3f} b={h.bm25_norm:.3f} "
            f"{h.metadata.get('type','')}{flag} | {h.basename()[:36]} | {h.section[:34]}"
        )
        print(f"     {h.text[:110].replace(chr(10),' / ')}")

    top_vec = hits[0].vec_sim if hits else 0.0
    if not hits or top_vec < SIM_THRESHOLD:
        answer = refusal_for(lang)
        print(f"\n--- REFUSAL (top_vec={top_vec:.3f} < {SIM_THRESHOLD}) ---\n{answer}")
    else:
        system_text = SYSTEM_ANSWER.format(lang=lang, context=_format_hits(hits))
        messages = [
            ChatMessage(role="system", content=system_text),
            ChatMessage(role="user", content=query),
        ]
        result = await llm.chat(messages, purpose="answer", temperature=0.2, max_tokens=1024)
        answer = (result.text or "").strip() or refusal_for(lang)
        print(f"\n--- RAW ANSWER (provider={result.provider}) ---\n{answer}")

    print("\n--- TELEGRAM HTML ---")
    print(md_to_telegram_html(answer))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--lang", default=None)
    args = parser.parse_args()
    asyncio.run(preview(args.query, args.lang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
