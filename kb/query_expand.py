"""Deterministic bilingual query expansion for cross-language retrieval.

The knowledge base is mixed-language: the highest-authority curated pages
(fee schedule, account rules, support supplement) are Chinese-only, while the
Help Center carries the same topics in eleven languages.

BM25 cannot bridge that gap. ``kb.tokenize`` splits CJK per character and Latin
per word, so an English query shares *zero* tokens with a Chinese document. In
:func:`kb.retriever.score_pool` a document found only by vector search keeps
``bm25_norm = 0.0``, so its blended score is capped at ``HYBRID_VECTOR_WEIGHT *
vec_sim``. A Chinese page can therefore be the single most semantically similar
chunk in the corpus and still lose to a lexically-matching English page:

    fee-schedule.md      vec 0.636  bm25 0.000  ->  score 0.445   (rank 9)
    perpetual-funding.md vec 0.582  bm25 1.000  ->  score 0.743   (rank 1)

That is how "if i am a superex vip6, what'd be my fee rate" retrieved no VIP
tier table at all while the Chinese wording of the same question retrieved it
four times over.

Appending the Chinese terms for the concepts an English query mentions restores
the lexical channel — and, because the embedding is computed on the expanded
string, lifts the vector score too (0.636 -> 0.797 for the case above).

This module is deliberately a static table rather than an LLM call: the router
runs on an 8B model whose term choice is not reliable enough to gate retrieval
quality on, and a table can be unit-tested offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CJK_RE = re.compile(r"[一-鿿]")

# Cap how much we bolt on. BM25 dilutes as the query grows: every added token
# that misses lowers the relative weight of the ones that hit.
MAX_ANCHORS = 8


@dataclass(frozen=True)
class Concept:
    """One support topic and how each language spells it.

    ``triggers`` are matched case-insensitively as substrings, so short entries
    must stay distinctive — "et" would fire inside "get", hence "et token".
    """

    triggers: tuple[str, ...]
    zh: tuple[str, ...]
    en: tuple[str, ...]


# Ordered by how often the topic shows up in community questions; when a query
# trips several concepts the earlier ones win the anchor budget.
CONCEPTS: tuple[Concept, ...] = (
    Concept(
        triggers=("fee rate", "fee", "commission", "maker", "taker", "vip",
                  "手续费", "费率", "费用", "等级"),
        zh=("手续费", "费率", "VIP 等级"),
        en=("fee", "rate", "maker", "taker"),
    ),
    Concept(
        triggers=("withdraw", "withdrawal", "提现", "提币"),
        zh=("提现", "提币"),
        en=("withdraw", "withdrawal"),
    ),
    Concept(
        triggers=("deposit", "recharge", "top up", "充值", "充币"),
        zh=("充值", "充币", "到账"),
        en=("deposit", "recharge"),
    ),
    Concept(
        triggers=("kyc", "verification", "identity", "实名", "认证", "身份"),
        zh=("实名认证", "身份认证", "KYC"),
        en=("kyc", "identity", "verification"),
    ),
    Concept(
        triggers=("margin mode", "cross margin", "isolated", "全仓", "逐仓", "保证金"),
        zh=("全仓", "逐仓", "保证金", "保证金模式"),
        en=("cross margin", "isolated margin"),
    ),
    Concept(
        triggers=("liquidat", "blown", "margin call", "爆仓", "强平", "强制平仓"),
        zh=("爆仓", "强平", "强制平仓"),
        en=("liquidation", "liquidated"),
    ),
    Concept(
        triggers=("funding rate", "funding fee", "资金费"),
        zh=("资金费率", "资金费用"),
        en=("funding rate", "funding fee"),
    ),
    Concept(
        triggers=("leverage", "杠杆"),
        zh=("杠杆", "杠杆倍数"),
        en=("leverage",),
    ),
    Concept(
        triggers=("futures", "perpetual", "contract", "合约", "永续"),
        zh=("合约", "永续合约"),
        en=("futures", "perpetual contract"),
    ),
    Concept(
        triggers=("spot", "现货"),
        zh=("现货", "现货交易"),
        en=("spot", "spot trading"),
    ),
    Concept(
        triggers=("2fa", "google auth", "authenticator", "two-factor",
                  "验证器", "谷歌验证"),
        zh=("谷歌验证器", "身份验证器", "两步验证"),
        en=("2fa", "authenticator", "google authenticator"),
    ),
    Concept(
        triggers=("referral", "rebate", "invite", "commission share",
                  "返佣", "邀请", "推荐"),
        zh=("返佣", "邀请", "推荐返佣"),
        en=("referral", "rebate", "invite"),
    ),
    Concept(
        triggers=("et token", "platform token", "平台币"),
        zh=("ET", "平台币", "抵扣"),
        en=("et token", "platform token", "deduction"),
    ),
    Concept(
        triggers=("freeze", "frozen", "locked", "冻结"),
        zh=("冻结", "解冻"),
        en=("frozen", "freeze"),
    ),
    Concept(
        triggers=("register", "sign up", "account", "注册", "账户", "帐户"),
        zh=("注册", "账户"),
        en=("register", "account"),
    ),
    Concept(
        triggers=("copy trad", "跟单"),
        zh=("跟单", "合约跟单"),
        en=("copy trading",),
    ),
    Concept(
        triggers=("grid", "网格"),
        zh=("网格", "网格策略"),
        en=("grid", "grid trading"),
    ),
    Concept(
        triggers=("free market", "amm", "liquidity", "自由市场", "流动性"),
        zh=("自由市场", "流动性", "做市"),
        en=("free market", "amm", "liquidity"),
    ),
    Concept(
        triggers=("earn", "staking", "apy", "理财", "年化"),
        zh=("理财", "年化"),
        en=("earn", "apy", "staking"),
    ),
    Concept(
        triggers=("order", "limit", "market order", "stop loss", "take profit",
                  "tp/sl", "reverse", "订单", "限价", "市价", "止盈", "止损",
                  "撤单", "撤回", "下单"),
        zh=("订单", "限价单", "市价单", "止盈止损", "下单"),
        en=("order", "limit order", "market order", "stop loss", "take profit"),
    ),
    Concept(
        triggers=("password", "密码"),
        zh=("密码", "登录密码", "资金密码", "找回密码"),
        en=("password", "login password", "fund password"),
    ),
    Concept(
        triggers=("phishing", "scam", "fraud", "fake support", "钓鱼", "诈骗",
                  "假客服", "举报"),
        zh=("钓鱼", "诈骗", "防钓鱼码", "假客服"),
        en=("phishing", "scam", "fraud", "anti-phishing code"),
    ),
    Concept(
        triggers=("human support", "contact support", "customer service",
                  "customer support", "complaint", "appeal", "ticket",
                  "客服", "工单", "申诉", "投诉"),
        zh=("客服", "人工客服", "工单", "申诉"),
        en=("customer support", "support ticket", "complaint"),
    ),
    Concept(
        triggers=("email", "phone", "bind", "binding", "bound", "邮箱",
                  "手机号", "绑定"),
        zh=("邮箱", "手机号", "绑定", "更改绑定"),
        en=("email", "phone number", "change binding"),
    ),
    Concept(
        triggers=("api key", "apikey", "api-key", "api 密钥", "api密钥", "密钥"),
        zh=("API 密钥", "API 管理"),
        en=("api key", "api management"),
    ),
    Concept(
        triggers=("login alert", "unusual login", "device", "异地登录",
                  "登录提醒", "设备"),
        zh=("异地登录", "登录提醒", "设备管理"),
        en=("login alert", "device management"),
    ),
    Concept(
        triggers=("wrong address", "wrong chain", "wrong network", "充错",
                  "提错", "地址错"),
        zh=("充错网络", "地址错误", "资产找回"),
        en=("wrong address", "wrong network", "recovery"),
    ),
    Concept(
        triggers=("buy", "sell", "买币", "卖币"),
        zh=("买币", "卖币", "下单"),
        en=("buy", "sell"),
    ),
    Concept(
        triggers=("trading pair", "pairs", "交易对"),
        zh=("交易对", "币种"),
        en=("trading pair", "listed coins"),
    ),
    Concept(
        triggers=("k-line", "kline", "candlestick", "chart", "indicator",
                  "k 线", "k线", "技术指标"),
        zh=("K 线", "技术指标", "行情"),
        en=("k-line", "chart", "indicator"),
    ),
    Concept(
        triggers=("mark price", "index price", "last price", "标记价", "指数价",
                  "最新价"),
        zh=("标记价格", "指数价格", "最新价格"),
        en=("mark price", "index price", "last price"),
    ),
    Concept(
        triggers=("maintenance margin", "维持保证金"),
        zh=("维持保证金", "维持保证金率"),
        en=("maintenance margin", "maintenance margin rate"),
    ),
    Concept(
        triggers=("campaign", "activity", "1usd", "活动", "体验金"),
        zh=("活动", "活动中心", "体验金"),
        en=("campaign", "activity", "trial bonus"),
    ),
    Concept(
        triggers=("web3", "wallet", "钱包"),
        zh=("钱包", "Web3 钱包"),
        en=("web3 wallet", "wallet"),
    ),
    Concept(
        triggers=("announcement", "listing", "new coin", "new token", "公告",
                  "上新", "上币"),
        zh=("公告", "上新币", "新币上线"),
        en=("announcement", "new listing"),
    ),
    Concept(
        triggers=("impermanent loss", "无常损失"),
        zh=("无常损失",),
        en=("impermanent loss",),
    ),
    Concept(
        triggers=("to stake", "staking", "质押"),
        zh=("质押", "理财"),
        en=("stake", "staking"),
    ),
)


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def anchors_for(query: str) -> list[str]:
    """Return the terms to append so BM25 can reach the other language.

    Chinese queries get English anchors, everything else gets Chinese ones —
    the corpus is Chinese and English, and non-Chinese scripts (fa/ru/vi) share
    BM25's Latin tokenisation path, so English is the useful bridge for them.
    """
    lowered = query.lower()
    want_zh = not _has_cjk(query)

    anchors: list[str] = []
    for concept in CONCEPTS:
        if not any(t in lowered for t in concept.triggers):
            continue
        for term in concept.zh if want_zh else concept.en:
            if term.lower() in lowered or term in anchors:
                continue
            anchors.append(term)
            if len(anchors) >= MAX_ANCHORS:
                return anchors
    return anchors


def expand_query(query: str) -> str:
    """Append cross-language anchors; return the query unchanged when none fit."""
    query = query.strip()
    if not query:
        return query
    anchors = anchors_for(query)
    if not anchors:
        return query
    return f"{query} {' '.join(anchors)}"
