"""KB coverage audit: run the retriever against a canonical list of
common crypto-exchange support questions, then LLM-judge whether the
retrieved chunks actually answer each question. Outputs a CSV report.

Output: /Users/ericc/Desktop/SuperEx/output/kb_gap_analysis.csv
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.WARNING)

from kb.retriever import Retriever
from llm.client import ChatMessage, LLMClient

# Categories of canonical questions. Each row: (category, zh, en, lang_for_audit)
QUESTIONS: list[tuple[str, str, str]] = [
    # ───────── 注册 / KYC ─────────
    ("注册/KYC", "如何注册 SuperEx 账户", "How to register a SuperEx account"),
    ("注册/KYC", "注册需要提供哪些信息", "What information do I need to register"),
    ("注册/KYC", "什么是 KYC", "What is KYC"),
    ("注册/KYC", "KYC 等级分几级，各级权限是什么", "KYC tiers and their limits"),
    ("注册/KYC", "KYC 提交后多久审核", "KYC review time"),
    ("注册/KYC", "KYC 被拒怎么办", "What to do if KYC is rejected"),
    ("注册/KYC", "可以注册多个账户吗", "Can I have multiple accounts"),
    ("注册/KYC", "如何修改绑定邮箱或手机号", "Change registered email or phone"),
    ("注册/KYC", "如何注销账户", "How to delete account"),
    ("注册/KYC", "邀请码 / 推荐人是什么", "What is invite code / referrer"),
    # ───────── 账户安全 ─────────
    ("账户安全", "如何开启 Google 二次验证 2FA", "How to enable Google Authenticator 2FA"),
    ("账户安全", "2FA 丢失了怎么办", "Lost 2FA, how to recover"),
    ("账户安全", "忘记登录密码怎么办", "Forgot login password"),
    ("账户安全", "忘记资金密码怎么办", "Forgot fund password"),
    ("账户安全", "账户被盗怎么办", "Account compromised what to do"),
    ("账户安全", "如何识别钓鱼邮件", "How to spot phishing emails"),
    ("账户安全", "如何防止社交工程诈骗", "Avoid social engineering scams"),
    ("账户安全", "异地登录提醒怎么设置", "Set up unusual login alerts"),
    ("账户安全", "防钓鱼码 anti-phishing code 是什么", "What is anti-phishing code"),
    ("账户安全", "如何创建和管理 API 密钥", "Create and manage API keys"),
    # ───────── 充值 ─────────
    ("充值", "如何充值", "How to deposit"),
    ("充值", "支持哪些币种充值", "What coins can I deposit"),
    ("充值", "支持哪些充值网络（ERC20 TRC20 BEP20）", "Supported deposit networks (ERC20 TRC20 BEP20)"),
    ("充值", "充值最低数量是多少", "Minimum deposit amount"),
    ("充值", "充值是否收手续费", "Are there deposit fees"),
    ("充值", "充值多久到账", "Deposit confirmation time"),
    ("充值", "充值未到账怎么办", "Deposit not credited"),
    ("充值", "充错网络怎么办", "Deposited via wrong network"),
    ("充值", "充错地址怎么办", "Deposited to wrong address"),
    ("充值", "充值地址会变化吗", "Does deposit address change"),
    # ───────── 提现 ─────────
    ("提现", "如何提现", "How to withdraw"),
    ("提现", "提现限额是多少", "Withdrawal limit"),
    ("提现", "提现手续费是多少", "Withdrawal fees"),
    ("提现", "提现处理时间多久", "Withdrawal processing time"),
    ("提现", "提现未到账怎么办", "Withdrawal not received"),
    ("提现", "提现被拒绝怎么办", "Withdrawal rejected"),
    ("提现", "提现地址白名单怎么设置", "Withdrawal address whitelist"),
    ("提现", "提现需要 2FA 或验证码吗", "Does withdrawal need 2FA"),
    ("提现", "提币错误地址或错误网络怎么办", "Withdrew to wrong address or wrong chain"),
    ("提现", "提币是否需要 KYC", "Does withdrawal need KYC"),
    # ───────── 现货交易 ─────────
    ("现货交易", "如何买币 / 卖币", "How to buy or sell coins"),
    ("现货交易", "限价单 / 市价单 / 止损单 区别", "Limit order vs market order vs stop loss"),
    ("现货交易", "现货手续费是多少", "Spot trading fees"),
    ("现货交易", "交易对有哪些", "Available trading pairs"),
    ("现货交易", "现货下单失败原因", "Why spot order failed"),
    ("现货交易", "现货如何撤单", "How to cancel a spot order"),
    ("现货交易", "K 线和技术指标怎么看", "How to read K-line and indicators"),
    ("现货交易", "现货网格交易怎么用", "How to use spot grid trading"),
    # ───────── 合约 / 期货 ─────────
    ("合约/期货", "什么是永续合约", "What is a perpetual contract"),
    ("合约/期货", "杠杆是什么，怎么调", "What is leverage and how to adjust"),
    ("合约/期货", "全仓和逐仓的区别", "Cross vs isolated margin"),
    ("合约/期货", "标记价、指数价、最新价区别", "Mark / index / last price"),
    ("合约/期货", "资金费率是什么，怎么计算", "What is funding rate"),
    ("合约/期货", "强平 / 爆仓机制", "Liquidation mechanism"),
    ("合约/期货", "维持保证金率 MMR", "Maintenance margin rate"),
    ("合约/期货", "合约手续费", "Futures trading fees"),
    ("合约/期货", "止盈止损怎么设置", "How to set TP/SL"),
    ("合约/期货", "合约持仓在哪里查询", "Where to view futures positions"),
    ("合约/期货", "合约爆仓会扣到现货账户吗", "Does liquidation deduct from spot account"),
    ("合约/期货", "跟单交易 copy trading 怎么用", "How to use copy trading"),
    # ───────── 理财 / Earn ─────────
    ("理财/Earn", "SuperEx 有理财产品吗", "Does SuperEx have Earn products"),
    ("理财/Earn", "理财年化收益怎么算", "How is Earn APR calculated"),
    ("理财/Earn", "理财如何赎回", "How to redeem Earn"),
    ("理财/Earn", "什么是体验金", "What is futures bonus"),
    ("理财/Earn", "ET 平台币有什么用", "What is ET token used for"),
    ("理财/Earn", "如何参与质押 staking", "How to stake"),
    # ───────── SuperEx 特色 ─────────
    ("平台特色", "什么是全币种合约 Index Futures", "What is index futures"),
    ("平台特色", "什么是 Free Market", "What is Free Market"),
    ("平台特色", "什么是 Free Market AMM", "What is Free Market AMM"),
    ("平台特色", "什么是 1USD 活动", "What is 1USD campaign"),
    ("平台特色", "Web3 钱包怎么接入", "How to connect Web3 wallet"),
    ("平台特色", "返佣 / 推荐返佣怎么算", "Referral rebate"),
    ("平台特色", "公告和上新币去哪看", "Where to see announcements and new listings"),
    ("平台特色", "无常损失是什么 (AMM 相关)", "What is impermanent loss"),
    # ───────── 费用 ─────────
    ("费用", "VIP 等级和手续费折扣", "VIP tiers and fee discounts"),
    ("费用", "提币网络费如何计算", "How are withdrawal network fees calculated"),
    ("费用", "是否有隐藏费用", "Are there any hidden fees"),
    # ───────── 客服 / 申诉 ─────────
    ("客服/申诉", "如何联系人工客服", "How to contact human support"),
    ("客服/申诉", "如何提交申诉或投诉", "How to file a complaint"),
    ("客服/申诉", "账户被冻结怎么办", "Account is frozen what to do"),
    ("客服/申诉", "资产被错误冻结如何解冻", "How to unfreeze assets"),
    ("客服/申诉", "交易对错误下单可以撤回吗", "Can I reverse a wrong trade"),
    ("客服/申诉", "如何举报诈骗群组或假客服", "How to report scams"),
]


TOP_K = 5

JUDGE_SYSTEM = """You are auditing a customer-service knowledge base for a crypto exchange.
Given a user QUESTION and the top retrieved CHUNKS, decide whether a support agent could
answer the question concretely using ONLY these chunks.

Return strict JSON: {"verdict": "yes" | "partial" | "no", "reason": "<one short sentence in Chinese>", "missing": "<if not yes, what specific fact is missing, in Chinese; else empty>"}

verdict = "yes"     : chunks contain a direct, actionable answer covering all parts of the question.
verdict = "partial" : chunks are topically related and answer SOME part, but key concrete facts (numbers, steps, names) are missing or only tangential.
verdict = "no"      : chunks do not answer the question at all, only the topic is loosely related.
"""

JUDGE_USER_TEMPLATE = """QUESTION (Chinese): {zh}
QUESTION (English): {en}

CHUNKS:
{chunks}

Respond with JSON only.
"""


async def judge_one(
    llm: LLMClient,
    zh: str,
    en: str,
    hits: list,
) -> tuple[str, str, str]:
    chunks_text = "\n\n".join(
        f"[{i + 1}] (vec={h.vec_sim:.2f} src={h.basename()[:40]} sec={h.section[:40]})\n{h.text[:600]}"
        for i, h in enumerate(hits[:TOP_K])
    )
    if not chunks_text:
        return "no", "no hits returned", "全部信息"
    user_msg = JUDGE_USER_TEMPLATE.format(zh=zh, en=en, chunks=chunks_text)
    messages = [
        ChatMessage(role="system", content=JUDGE_SYSTEM),
        ChatMessage(role="user", content=user_msg),
    ]
    try:
        data, _ = await llm.chat_json(
            messages, purpose="answer", temperature=0.0, max_tokens=300
        )
    except Exception as exc:
        return "error", f"LLM error: {exc!s:.80}", ""
    verdict = str(data.get("verdict", "error")).lower()
    reason = str(data.get("reason", ""))
    missing = str(data.get("missing", ""))
    return verdict, reason, missing


async def amain() -> None:
    retriever = Retriever()
    llm = LLMClient()
    out_path = ROOT.parent.parent / "output" / "kb_gap_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = [
        [
            "分类",
            "问题（中文）",
            "问题（英文）",
            "覆盖判定",
            "理由",
            "缺失要点",
            "top_vec",
            "top命中源",
            "top命中片段(80字)",
        ]
    ]
    label_for = {
        "yes": "✓ 已覆盖",
        "partial": "△ 部分",
        "no": "✗ 缺失",
        "error": "? 错误",
    }
    counts: dict[str, int] = {v: 0 for v in label_for.values()}

    started = time.monotonic()
    for idx, (category, zh, en) in enumerate(QUESTIONS, 1):
        hits_zh = retriever.search(zh, top_k=TOP_K, lang_boost="zh")
        hits_en = retriever.search(en, top_k=TOP_K, lang_boost="en")
        merged: dict = {}
        for h in hits_zh + hits_en:
            existing = merged.get(h.doc_id)
            if existing is None or existing.score < h.score:
                merged[h.doc_id] = h
        all_hits = sorted(merged.values(), key=lambda h: -h.vec_sim)
        top_vec = all_hits[0].vec_sim if all_hits else 0.0
        top_src = all_hits[0].basename() if all_hits else ""
        top_text = (all_hits[0].text[:80].replace("\n", " ") if all_hits else "")

        verdict, reason, missing = await judge_one(llm, zh, en, all_hits)
        label = label_for.get(verdict, "? 错误")
        counts[label] += 1
        rows.append(
            [
                category, zh, en, label, reason, missing,
                f"{top_vec:.3f}", top_src, top_text,
            ]
        )
        if idx % 5 == 0:
            elapsed = time.monotonic() - started
            print(f"[{idx}/{len(QUESTIONS)}] elapsed={elapsed:.0f}s | last={label} | {zh[:30]}")

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    total = sum(counts.values())
    print(f"\nWrote {len(rows) - 1} rows to {out_path}")
    print("Summary (LLM-judged):")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v}/{total} ({v / total * 100:.1f}%)")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
