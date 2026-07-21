"""Compare the crawled Help Center against the existing KB and report conflicts.

Only zh-hk and en-001 are compared: the existing wiki and 客服 material is written
in Chinese and English, so diffing a Persian translation against it would produce
noise, not findings.

Outputs output/superex-wiki-conflict-report.md in the task-book table format.

Usage:
    python -m scripts.detect_conflicts              # deterministic + LLM adjudication
    python -m scripts.detect_conflicts --no-llm     # candidates only, zero LLM cost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s", level=logging.INFO
)
logger = logging.getLogger("detect_conflicts")

from config import OUTPUT_DIR  # noqa: E402
from kb.conflicts import ConflictCandidate, find_candidates  # noqa: E402
from kb.loader import load_all  # noqa: E402
from llm.client import ChatMessage, LLMClient  # noqa: E402

REPORT_PATH = OUTPUT_DIR / "superex-wiki-conflict-report.md"
COMPARE_LANGS = {"zh-TW", "zh-CN", "en"}

JUDGE_PROMPT = """You compare two statements about the SuperEx exchange.

OLD (existing internal knowledge base):
{old}

NEW (official SuperEx Help Center, updated {updated}):
{new}

Both mention the unit "{unit}". Decide whether they genuinely CONTRADICT each
other about the same thing.

Answer conflict=false unless ALL of these hold:
- BOTH texts explicitly state a value for the SAME named parameter.
- It is the same product and the same programme (referral rebate, KOL rebate and
  Free Market copy-pair dividends are DIFFERENT programmes — never compare them).
- The stated values genuinely differ.

Answer conflict=false if the NEW text does not actually state a value for the
parameter the OLD text is talking about. Never infer, complete or guess a value
that is not literally written in the text — quote only what is present.
Also not a conflict: different VIP tiers, the same value written differently,
or one text giving a range that contains the other's value.

Return one JSON object, nothing else:
{{"conflict": <bool>,
  "type": "<参数变化|规则变化|流程变化|产品说明变化|无冲突>",
  "subject": "<what the parameter is, in Chinese, <=25 chars>",
  "old_value": "<value stated in OLD, <=30 chars>",
  "new_value": "<value stated in NEW, <=30 chars>",
  "confidence": <0.0-1.0>}}"""


async def adjudicate(
    llm: LLMClient, cand: ConflictCandidate
) -> dict | None:
    old_text = " / ".join(c.sentence for c in cand.old_claims)
    new_text = " / ".join(c.sentence for c in cand.new_claims)
    prompt = JUDGE_PROMPT.format(
        old=old_text[:900],
        new=new_text[:900],
        unit=cand.unit,
        updated=str(cand.new_doc.extras.get("updated_at", "unknown"))[:10],
    )
    try:
        data, _ = await llm.chat_json(
            [
                ChatMessage(role="system", content="You output one JSON object only."),
                ChatMessage(role="user", content=prompt),
            ],
            purpose="answer",
            temperature=0.0,
            max_tokens=400,
        )
        return data or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge failed: %s", exc)
        return None


def render_report(rows: list[dict], stats: dict) -> str:
    lines = [
        "# SuperEx 新旧知识库冲突报告",
        "",
        f"> 生成时间:{stats['generated']}",
        "> 新内容来源:support.superex.com 帮助中心(官方,带 updated_at)",
        "> 旧内容来源:raw/客服、raw/官方教程、wiki/",
        "",
        "## 处理原则",
        "",
        "1. **以 SuperEx 官网 / 最新链接为准** —— 官方帮助中心带 `updated_at`,可核验时效。",
        "2. 无法判断的标记 `人工审核`,不自动覆盖。",
        "3. 本报告不直接改写旧 Wiki;确认后再执行清洗。",
        "",
        "## 统计",
        "",
        f"- 官方帮助中心参与比对的块:{stats['new_chunks']}",
        f"- 现有知识库参与比对的块:{stats['old_chunks']}",
        f"- 数值分歧候选:{stats['candidates']}",
        f"- LLM 判定为真实冲突:{stats['confirmed']}",
        "",
    ]
    if not rows:
        lines += ["## 结论", "", "未发现需要处理的事实冲突。", ""]
        return "\n".join(lines)

    lines += [
        "## Conflict Report",
        "",
        "| 模块 | 冲突类型 | 旧内容 | 新内容 | 官方更新时间 | 推荐版本 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        def esc(x: str) -> str:
            return str(x).replace("|", "\\|").replace("\n", " ")[:120]

        lines.append(
            f"| {esc(r['subject'])} | {esc(r['type'])} | {esc(r['old_value'])} "
            f"| {esc(r['new_value'])} | {esc(r['updated_at'])[:10]} | {esc(r['recommend'])} |"
        )
    lines.append("")
    lines += ["## 明细与出处", ""]
    for i, r in enumerate(rows, 1):
        lines += [
            f"### {i}. {r['subject']}  ({r['type']})",
            "",
            f"- **旧**:{r['old_snippet']}",
            f"  - 出处:`{r['old_source']}`",
            f"- **新(官方)**:{r['new_snippet']}",
            f"  - 出处:{r['new_url'] or '`' + r['new_source'] + '`'}",
            f"  - 官方更新时间:{r['updated_at'][:10]}",
            f"- **推荐**:{r['recommend']}(判定置信度 {r['confidence']})",
            "",
        ]
    return "\n".join(lines)


async def run(use_llm: bool, limit: int | None) -> int:
    logger.info("Loading knowledge base…")
    docs = load_all()
    new_docs = [
        d
        for d in docs
        if d.type == "help_center" and d.lang in COMPARE_LANGS
    ]
    old_docs = [d for d in docs if d.type != "help_center"]
    logger.info("official=%d  existing=%d", len(new_docs), len(old_docs))

    candidates = find_candidates(old_docs, new_docs)
    if limit:
        candidates = candidates[:limit]

    rows: list[dict] = []
    if use_llm and candidates:
        llm = LLMClient()
        for i, cand in enumerate(candidates, 1):
            verdict = await adjudicate(llm, cand)
            if i % 10 == 0:
                logger.info("adjudicated %d/%d", i, len(candidates))
            if not verdict or not verdict.get("conflict"):
                continue
            updated = str(cand.new_doc.extras.get("updated_at", ""))
            confidence = float(verdict.get("confidence") or 0)
            rows.append(
                {
                    "subject": verdict.get("subject") or cand.topic,
                    "type": verdict.get("type") or "参数变化",
                    "old_value": verdict.get("old_value", ""),
                    "new_value": verdict.get("new_value", ""),
                    "updated_at": updated,
                    "recommend": "采用官方新值" if confidence >= 0.6 else "人工审核",
                    "confidence": round(confidence, 2),
                    "old_snippet": cand.old_claims[0].sentence,
                    "new_snippet": cand.new_claims[0].sentence,
                    "old_source": cand.old_doc.source,
                    "new_source": cand.new_doc.source,
                    "new_url": str(cand.new_doc.extras.get("url", "")),
                }
            )

    import datetime

    stats = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "new_chunks": len(new_docs),
        "old_chunks": len(old_docs),
        "candidates": len(candidates),
        "confirmed": len(rows),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(rows, stats), encoding="utf-8")
    (OUTPUT_DIR / "superex-wiki-conflicts.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("candidates=%d confirmed=%d -> %s", len(candidates), len(rows), REPORT_PATH)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM adjudication")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(run(use_llm=not args.no_llm, limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
