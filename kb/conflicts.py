"""Detect factual conflicts between the crawled Help Center and the existing KB.

Strategy — cheap deterministic filter first, LLM only on survivors:

1. Pair each official Help Center chunk with lexically similar existing chunks
   (BM25, in-memory — deliberately avoids chroma so this can run while the
   vector index is rebuilding).
2. Keep only pairs where a *numeric claim about the same unit* differs. Prose
   rewording is not a conflict; "0.2%" vs "0.1%" is.
3. An LLM adjudicates only those survivors.

Resolution rule (from the operator): the official site / newest link wins. The
Help Center carries `updated_at`, so recency is checkable rather than assumed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from .document import Document
from .tokenize import tokenize

logger = logging.getLogger(__name__)

# Units whose values are worth comparing. Percentages, money, leverage, and
# durations cover the parameters support answers actually get wrong.
_NUMERIC_CLAIM_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*"
    r"(%|％|USDT|USD|U\b|倍|x\b|X\b|小时|小時|hours?|天|days?|分钟|分鐘|minutes?)",
    re.IGNORECASE,
)

_UNIT_ALIASES = {
    "％": "%",
    "usd": "USDT",
    "u": "USDT",
    "x": "倍",
    "小時": "小时",
    "hour": "小时",
    "hours": "小时",
    "day": "天",
    "days": "天",
    "分鐘": "分钟",
    "minute": "分钟",
    "minutes": "分钟",
}

# Topic keywords that make a numeric comparison meaningful. Without a shared
# topic, "0.1%" in two unrelated articles is not a conflict.
_TOPIC_KEYWORDS = (
    ("fee", ("手续费", "手續費", "费率", "費率", "maker", "taker", "fee")),
    ("withdraw", ("提现", "提現", "提幣", "提币", "withdraw")),
    ("deposit", ("充值", "充幣", "存入", "deposit")),
    ("leverage", ("杠杆", "槓桿", "leverage")),
    ("vip", ("vip", "等级", "等級", "tier")),
    ("funding", ("资金费", "資金費", "funding")),
    (
        "liquidation",
        # 強制平倉 is the official phrasing and does not contain 強平 as a substring.
        ("强平", "強平", "强制平仓", "強制平倉", "爆仓", "爆倉", "liquidation"),
    ),
    ("kyc", ("kyc", "实名", "實名", "认证", "認證")),
    ("earn", ("理财", "理財", "年化", "apy", "earn")),
    ("rebate", ("返佣", "返傭", "rebate", "commission")),
)


# SuperEx ships several futures products whose rules legitimately differ, so a
# numeric difference across products is not a conflict. Index Futures officially
# liquidates at margin ratio ≤100% while USDT-margined perpetuals liquidate at
# ≤0%; comparing them produced six confident false positives before this gate
# existed. Same for leverage: 150x is a perpetuals figure, not an index-futures one.
_PRODUCT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("index_futures", ("全币种合约", "全幣種合約", "指数合约", "指數合約", "index futures")),
    (
        "perpetual",
        ("永续合约", "永續合約", "u本位", "usdt-margined", "usdt margined", "perpetual"),
    ),
    ("bonus", ("体验金", "體驗金", "bonus", "trial fund")),
    ("copy_trading", ("跟单", "跟單", "copy trading", "副本")),
    ("free_market", ("free market", "自由区", "自由區", "amm")),
    ("spot_grid", ("网格", "網格", "grid")),
    ("earn", ("理财", "理財", "earn", "金融业务", "金融業務")),
    ("p2p", ("p2p", "c2c")),
    ("spot", ("现货", "現貨", "spot")),
)


def products_of(text: str) -> set[str]:
    """Which SuperEx product(s) a chunk is about; empty when undetermined."""
    lowered = text.lower()
    return {name for name, words in _PRODUCT_KEYWORDS if any(w in lowered for w in words)}


def product_of_document(doc: Document) -> set[str]:
    """Product attribution from metadata only — never from the body.

    Signals used: the file path, the document title, and (for crawled Help Center
    chunks) the owning Zendesk section. "Leverage Multiplier and Margin" reads as
    generic but lives under "USDT-Margined Perpetual Contracts", so its 150x
    figure does not apply to Index Futures.

    The body is deliberately excluded. A long Index Futures tutorial mentions
    perpetuals in passing, which made its product set overlap everything and let
    cross-product pairs through the gate — the exact false positives this
    function exists to stop.
    """
    signals = " ".join(
        str(part)
        for part in (doc.source, doc.section, doc.extras.get("section") or "")
        if part
    )
    return products_of(signals)


@dataclass(frozen=True)
class NumericClaim:
    value: float
    unit: str
    sentence: str

    def key(self) -> str:
        return self.unit


@dataclass(frozen=True)
class ConflictCandidate:
    topic: str
    unit: str
    old_doc: Document
    new_doc: Document
    old_claims: tuple[NumericClaim, ...]
    new_claims: tuple[NumericClaim, ...]

    def old_values(self) -> set[float]:
        return {c.value for c in self.old_claims}

    def new_values(self) -> set[float]:
        return {c.value for c in self.new_claims}


def _normalise_unit(unit: str) -> str:
    lowered = unit.strip().lower()
    return _UNIT_ALIASES.get(lowered, _UNIT_ALIASES.get(unit.strip(), unit.strip()))


# A bare "." must not end a sentence — "0.2%" would be split into "0." and "2%",
# silently turning a 0.2% claim into a 2% one. Latin periods only break a
# sentence when followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])\s*|(?<=\.)\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def extract_claims(text: str) -> list[NumericClaim]:
    """Every number-with-unit in the text, tagged with its sentence."""
    claims: list[NumericClaim] = []
    for sentence in _split_sentences(text):
        for m in _NUMERIC_CLAIM_RE.finditer(sentence):
            raw = m.group(1).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            claims.append(
                NumericClaim(
                    value=value,
                    unit=_normalise_unit(m.group(2)),
                    sentence=sentence[:300],
                )
            )
    return claims


def topics_of(text: str) -> set[str]:
    lowered = text.lower()
    return {name for name, words in _TOPIC_KEYWORDS if any(w in lowered for w in words)}


def _claims_by_unit(claims: Iterable[NumericClaim]) -> dict[str, list[NumericClaim]]:
    out: dict[str, list[NumericClaim]] = {}
    for c in claims:
        out.setdefault(c.key(), []).append(c)
    return out


# Worked examples state numbers that are outputs of their own premises, not
# platform parameters. "保證金合計 220 USDT" (案例三, 20 體驗金) and "保证金合计
# 300 USDT" (案例二, 100 体验金) are both arithmetically correct for their own
# case — flagging them as a parameter change would push someone to "fix" content
# that is already right. A numeric comparator cannot tell cases apart, so
# example blocks are excluded from comparison entirely.
_EXAMPLE_MARKERS = (
    "案例",
    "举例",
    "舉例",
    "例如",
    "示例",
    "假设",
    "假設",
    "for example",
    "example:",
    "case ",
    "scenario",
    "e.g.",
)


def is_worked_example(text: str) -> bool:
    """True when the text is an illustrative calculation rather than a rule."""
    lowered = text.lower()
    return any(marker in lowered for marker in _EXAMPLE_MARKERS)


def _overlap(a: list[str], b: list[str]) -> float:
    """Jaccard overlap on token sets.

    BM25 alone cannot gate relevance here: BM25Okapi yields negative scores for
    terms appearing in most of the corpus, so a score threshold is corpus-size
    dependent. Overlap is stable regardless of corpus size.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_candidates(
    old_docs: list[Document],
    new_docs: list[Document],
    *,
    top_n: int = 3,
    max_pairs_per_doc: int = 2,
    min_overlap: float = 0.12,
) -> list[ConflictCandidate]:
    """Pair official docs with existing ones and keep only numeric divergences."""
    from rank_bm25 import BM25Okapi

    # The marker usually sits in a heading ("案例二：…"), not in the sentence
    # carrying the number, so this has to be judged per chunk rather than per
    # sentence. Rules quoted inside an example block are unreliable to compare
    # anyway — precision matters more here, since every false positive invites
    # someone to "correct" content that is already right.
    indexable = [
        d for d in old_docs if extract_claims(d.text) and not is_worked_example(d.text)
    ]
    if not indexable:
        logger.warning("No numeric claims in the existing KB; nothing to compare")
        return []
    corpus = [tokenize(d.text) for d in indexable]
    bm25 = BM25Okapi(corpus)

    candidates: list[ConflictCandidate] = []
    for new in new_docs:
        new_claims = extract_claims(new.text)
        if not new_claims or is_worked_example(new.text):
            continue
        new_topics = topics_of(new.text)
        if not new_topics:
            continue
        new_products = product_of_document(new)
        new_tokens = tokenize(new.text)
        scores = bm25.get_scores(new_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_n]
        added = 0
        for idx, _score in ranked:
            if added >= max_pairs_per_doc:
                break
            old = indexable[idx]
            shared = new_topics & topics_of(old.text)
            if not shared:
                continue
            if _overlap(new_tokens, corpus[idx]) < min_overlap:
                continue
            # Different products may legitimately carry different values.
            old_products = product_of_document(old)
            if new_products and old_products and not (new_products & old_products):
                continue
            old_by_unit = _claims_by_unit(extract_claims(old.text))
            new_by_unit = _claims_by_unit(new_claims)
            matched = False
            for unit in sorted(set(old_by_unit) & set(new_by_unit)):
                # Document-level topic overlap is far too loose: a fee sentence
                # and a liquidation sentence both contain "%", and pairing them
                # invites the judge to invent a relationship. Require the two
                # *claim sentences themselves* to be about the same thing.
                for topic in sorted(shared):
                    old_same = [
                        c for c in old_by_unit[unit] if topic in topics_of(c.sentence)
                    ]
                    new_same = [
                        c for c in new_by_unit[unit] if topic in topics_of(c.sentence)
                    ]
                    if not old_same or not new_same:
                        continue
                    old_vals = {c.value for c in old_same}
                    new_vals = {c.value for c in new_same}
                    # Identical or overlapping value sets are agreement.
                    if old_vals & new_vals:
                        continue
                    candidates.append(
                        ConflictCandidate(
                            topic=topic,
                            unit=unit,
                            old_doc=old,
                            new_doc=new,
                            old_claims=tuple(old_same[:4]),
                            new_claims=tuple(new_same[:4]),
                        )
                    )
                    added += 1
                    matched = True
                    break
                if matched:
                    break
    logger.info("Numeric-divergence candidates: %d", len(candidates))
    return candidates
