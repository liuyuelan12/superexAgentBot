"""Real-time crypto spot prices for the support bot.

Why this exists: asked "现在 BTC 价格多少", the RAG bot retrieved a help-center
article whose worked example used 100000 as a placeholder price and reported it
as the live price — more than 50% above the real value. Price questions must
never be answered from the static index. This module intercepts them and returns
a real quote.

Primary source is SuperEx's OWN public market API
(``/spot/public/v3/summary``), so the bot quotes exactly what a user sees on the
platform, covers all ~710 listed pairs including SuperEx-native tokens (ET), and
needs no API key. CoinGecko and Binance are reliability fallbacks only, used if
SuperEx is unreachable; when a fallback answers, the reply says so, because a
fallback price may differ slightly from the platform's own.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import httpx

from config import (
    BINANCE_BASE,
    COINGECKO_BASE,
    PRICE_CACHE_TTL,
    PRICE_HTTP_TIMEOUT,
    SUPEREX_API_BASE,
    SUPEREX_MARKETS_URL,
    SUPEREX_SUMMARY_PATH,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Coin name resolution
#
# Coverage now comes from SuperEx's live map (~710 coins), so this table only
# resolves names to canonical tickers for intent extraction and gives the
# CoinGecko / Binance ids used by the fallbacks (mainstream coins only).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Coin:
    symbol: str
    coingecko_id: Optional[str]
    binance: Optional[str]


_FALLBACK_COINS: tuple[Coin, ...] = (
    Coin("BTC", "bitcoin", "BTCUSDT"),
    Coin("ETH", "ethereum", "ETHUSDT"),
    Coin("USDC", "usd-coin", "USDCUSDT"),
    Coin("BNB", "binancecoin", "BNBUSDT"),
    Coin("SOL", "solana", "SOLUSDT"),
    Coin("XRP", "ripple", "XRPUSDT"),
    Coin("DOGE", "dogecoin", "DOGEUSDT"),
    Coin("ADA", "cardano", "ADAUSDT"),
    Coin("TRX", "tron", "TRXUSDT"),
    Coin("TON", "the-open-network", "TONUSDT"),
    Coin("AVAX", "avalanche-2", "AVAXUSDT"),
    Coin("SHIB", "shiba-inu", "SHIBUSDT"),
    Coin("LINK", "chainlink", "LINKUSDT"),
    Coin("DOT", "polkadot", "DOTUSDT"),
    Coin("MATIC", "matic-network", "MATICUSDT"),
    Coin("LTC", "litecoin", "LTCUSDT"),
    Coin("BCH", "bitcoin-cash", "BCHUSDT"),
    Coin("UNI", "uniswap", "UNIUSDT"),
    Coin("ATOM", "cosmos", "ATOMUSDT"),
    Coin("NEAR", "near", "NEARUSDT"),
    Coin("APT", "aptos", "APTUSDT"),
    Coin("ARB", "arbitrum", "ARBUSDT"),
    Coin("OP", "optimism", "OPUSDT"),
    Coin("SUI", "sui", "SUIUSDT"),
    Coin("PEPE", "pepe", "PEPEUSDT"),
    Coin("USDT", "tether", None),
)

_FALLBACK_BY_SYMBOL: dict[str, Coin] = {c.symbol: c for c in _FALLBACK_COINS}

# alias (lowercased) -> canonical symbol. Names + the fallback tickers; any other
# ticker the user types is resolved against SuperEx's live map at query time.
_ALIASES: dict[str, str] = {c.symbol.lower(): c.symbol for c in _FALLBACK_COINS}
_ALIASES.update(
    {
        "bitcoin": "BTC", "比特币": "BTC", "比特幣": "BTC", "大饼": "BTC",
        "ethereum": "ETH", "以太坊": "ETH", "以太幣": "ETH", "以太币": "ETH", "以太": "ETH",
        "tether": "USDT", "泰达币": "USDT", "泰達幣": "USDT",
        "solana": "SOL", "索拉纳": "SOL",
        "ripple": "XRP", "瑞波": "XRP", "瑞波币": "XRP",
        "dogecoin": "DOGE", "狗狗币": "DOGE", "狗狗幣": "DOGE", "狗币": "DOGE",
        "cardano": "ADA", "艾达币": "ADA",
        "tron": "TRX", "波场": "TRX", "波場": "TRX",
        "polkadot": "DOT", "波卡": "DOT",
        "litecoin": "LTC", "莱特币": "LTC", "萊特幣": "LTC",
        "chainlink": "LINK",
        "avalanche": "AVAX", "雪崩": "AVAX",
        "shiba": "SHIB", "shibainu": "SHIB", "柴犬": "SHIB", "屎币": "SHIB",
        "pepe": "PEPE", "佩佩": "PEPE",
        "polygon": "MATIC",
        "cosmos": "ATOM",
        "arbitrum": "ARB",
        "optimism": "OP",
        "superex": "ET", "平台币": "ET", "平台幣": "ET", "外星人": "ET",
    }
)

# Uppercase words that are not coins — keeps bare-ticker extraction from firing on
# common acronyms. A user asking a genuine price question about one of these is
# vanishingly rare, and USDT/USDC still resolve via _ALIASES.
_TICKER_STOPWORDS = frozenset(
    {
        "USD", "USDT", "USDC", "NFT", "CEO", "CTO", "API", "KYC", "VIP", "FAQ",
        "ETF", "DAO", "P2P", "AMM", "APP", "APR", "APY", "URL", "FUD", "ATH",
        "DEX", "CEX", "PNL", "ROI", "OTC", "IEO", "ICO", "TVL", "GAS", "AI",
    }
)
_BARE_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,6}(?![A-Za-z0-9])")


# --------------------------------------------------------------------------- #
# Intent detection (pure, deterministic — the router's 8B model is not reliable
# enough to gate this, and a table can be unit-tested offline)
# --------------------------------------------------------------------------- #
_EXCLUDE_RE = re.compile(
    r"手续费|手續費|费率|費率|\bfee\b|commission"
    r"|标记价|標記價|指数价|指數價|最新价|最新價|mark\s*price|index\s*price|last\s*price"
    r"|资金费|資金費|funding"
    r"|强平|強平|爆仓|爆倉|清算|liquidat"
    r"|预测|預測|会涨|會漲|会跌|會跌|能涨|能漲|predict|target|走势|走勢|趋势|趨勢"
    r"|历史|歷史|昨天|去年|過去|过去|history|k\s*线|k\s*線|kline|candlestick|chart",
    re.IGNORECASE,
)
_PRICE_WORD_RE = re.compile(
    r"价格|價格|价钱|價錢|多少钱|多少錢|多少|现价|現價|币价|幣價|报价|報價|值多少|涨到|漲到|现在.*价|現在.*價"
    r"|price|worth|quote|how\s+much|what.*(cost|worth)|current.*(price|value)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PriceIntent:
    symbols: tuple[str, ...]  # canonical tickers, order preserved, deduped


def _extract_coins(text: str) -> list[str]:
    """Return candidate coin symbols mentioned in *text*, first-seen order.

    Known aliases (names + fallback tickers) plus any bare uppercase ticker, so a
    user can ask about any of SuperEx's ~710 listed coins, not just the curated
    ones. Unknown tickers are validated against the live map at fetch time.
    """
    lowered = text.lower()
    hits: dict[str, int] = {}

    for alias, symbol in _ALIASES.items():
        pos = _alias_pos(lowered, text, alias)
        if pos is not None:
            hits[symbol] = min(hits.get(symbol, 1 << 30), pos)

    for m in _BARE_TICKER_RE.finditer(text):
        tok = m.group()
        if tok in _TICKER_STOPWORDS or tok in hits:
            continue
        hits[tok] = min(hits.get(tok, 1 << 30), m.start())

    return sorted(hits, key=lambda s: hits[s])


def _alias_pos(lowered: str, original: str, alias: str) -> Optional[int]:
    if alias.isascii():
        m = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered)
        return m.start() if m else None
    idx = original.find(alias)
    return idx if idx >= 0 else None


def detect_price_intent(text: str) -> Optional[PriceIntent]:
    """Detect a real-time spot-price question and the coins it names.

    Returns None unless the text asks for a value AND names at least one coin AND
    is not a fee / concept / funding / liquidation / chart / prediction question.
    """
    if not text or not text.strip():
        return None
    if _EXCLUDE_RE.search(text):
        return None
    if not _PRICE_WORD_RE.search(text):
        return None
    coins = _extract_coins(text)
    if not coins:
        return None
    return PriceIntent(symbols=tuple(coins))


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PriceResult:
    symbol: str
    usd: Optional[float]  # None when unavailable
    change_24h: Optional[float]
    source: str  # "superex" | "coingecko" | "binance" | "unavailable"


SuperExFetch = Callable[[], Awaitable[dict[str, dict]]]
CoingeckoFetch = Callable[[list[str]], Awaitable[dict[str, dict]]]
BinanceFetch = Callable[[str], Awaitable[Optional[dict]]]


class PriceService:
    """SuperEx-own prices with a short cache and CoinGecko→Binance fallback."""

    def __init__(
        self,
        *,
        cache_ttl: float = PRICE_CACHE_TTL,
        timeout: float = PRICE_HTTP_TIMEOUT,
        superex_fetch: Optional[SuperExFetch] = None,
        coingecko_fetch: Optional[CoingeckoFetch] = None,
        binance_fetch: Optional[BinanceFetch] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._now = now
        self._sx = superex_fetch or self._default_superex
        self._cg = coingecko_fetch or self._default_coingecko
        self._bn = binance_fetch or self._default_binance
        # SuperEx returns every pair in one call, so the whole map is cached once.
        self._sx_map: dict[str, tuple[float, Optional[float]]] = {}
        self._sx_expiry: float = 0.0
        self._sx_lock = asyncio.Lock()

    async def get_prices(self, symbols: list[str]) -> list[PriceResult]:
        results: dict[str, PriceResult] = {}
        sx_map = await self._superex_map()
        for sym in symbols:
            row = sx_map.get(sym)
            if row is not None:
                results[sym] = PriceResult(sym, row[0], row[1], "superex")

        missing = [s for s in symbols if s not in results]
        if missing:
            await self._fill_fallback(missing, results)

        return [
            results.get(s, PriceResult(s, None, None, "unavailable")) for s in symbols
        ]

    async def _superex_map(self) -> dict[str, tuple[float, Optional[float]]]:
        if self._sx_map and self._sx_expiry > self._now():
            return self._sx_map
        async with self._sx_lock:
            if self._sx_map and self._sx_expiry > self._now():  # double-check
                return self._sx_map
            try:
                raw = await self._sx()
            except Exception:  # noqa: BLE001 - fall back rather than crash
                logger.warning("SuperEx price fetch failed", exc_info=True)
                return self._sx_map  # possibly stale; better than nothing
            parsed: dict[str, tuple[float, Optional[float]]] = {}
            for sym, row in raw.items():
                usd = _opt_float(row.get("usd"))
                if usd is not None:
                    parsed[sym] = (usd, _opt_float(row.get("change")))
            if parsed:
                self._sx_map = parsed
                self._sx_expiry = self._now() + self._ttl
            return self._sx_map

    async def _fill_fallback(
        self, symbols: list[str], results: dict[str, PriceResult]
    ) -> None:
        cg_ids = {
            _FALLBACK_BY_SYMBOL[s].coingecko_id: s
            for s in symbols
            if _FALLBACK_BY_SYMBOL.get(s) and _FALLBACK_BY_SYMBOL[s].coingecko_id
        }
        if cg_ids:
            try:
                data = await self._cg(list(cg_ids.keys()))
            except Exception:  # noqa: BLE001
                logger.warning("CoinGecko fallback failed", exc_info=True)
                data = {}
            for cid, sym in cg_ids.items():
                row = data.get(cid)
                if row and row.get("usd") is not None:
                    results[sym] = PriceResult(
                        sym, float(row["usd"]), _opt_float(row.get("usd_24h_change")), "coingecko"
                    )

        still = [
            s
            for s in symbols
            if s not in results and _FALLBACK_BY_SYMBOL.get(s) and _FALLBACK_BY_SYMBOL[s].binance
        ]
        if still:
            rows = await asyncio.gather(
                *(self._safe_binance(_FALLBACK_BY_SYMBOL[s].binance) for s in still)
            )
            for sym, row in zip(still, rows):
                if row and row.get("price") is not None:
                    results[sym] = PriceResult(
                        sym, float(row["price"]), _opt_float(row.get("change")), "binance"
                    )

    async def _safe_binance(self, symbol: str) -> Optional[dict]:
        try:
            return await self._bn(symbol)
        except Exception:  # noqa: BLE001
            logger.warning("Binance fallback failed for %s", symbol, exc_info=True)
            return None

    async def _default_superex(self) -> dict[str, dict]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{SUPEREX_API_BASE}{SUPEREX_SUMMARY_PATH}",
                headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
            )
            resp.raise_for_status()
            payload = resp.json()
        if payload.get("code") != 200:
            raise RuntimeError(f"SuperEx summary code={payload.get('code')}")
        out: dict[str, dict] = {}
        for item in payload.get("data", []):
            if item.get("quote_currency") != "USDT":
                continue  # quote everything in USDT
            base = item.get("base_currency")
            if base:
                out[base] = {
                    "usd": item.get("last_price"),
                    "change": item.get("price_change_percent_24h"),
                }
        return out

    async def _default_coingecko(self, ids: list[str]) -> dict[str, dict]:
        params = {
            "ids": ",".join(ids),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{COINGECKO_BASE}/simple/price", params=params)
            resp.raise_for_status()
            return resp.json()

    async def _default_binance(self, symbol: str) -> Optional[dict]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{BINANCE_BASE}/api/v3/ticker/24hr", params={"symbol": symbol}
            )
            resp.raise_for_status()
            data = resp.json()
            return {"price": data.get("lastPrice"), "change": data.get("priceChangePercent")}


def _opt_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Answer formatting (localised; mirrors the user's language like the rest of the bot)
# --------------------------------------------------------------------------- #
def _fmt_usd(value: float) -> str:
    if value >= 1:
        return f"${value:,.2f}"
    return f"${value:,.8f}".rstrip("0").rstrip(".")


def _fmt_change(pct: Optional[float], zh: bool) -> str:
    if pct is None:
        return ""
    sign = "+" if pct >= 0 else ""
    return f",24 小时 {sign}{pct:.2f}%" if zh else f", 24h {sign}{pct:.2f}%"


def format_price_answer(results: list[PriceResult], lang: str) -> str:
    zh = lang.startswith("zh")
    priced: list[PriceResult] = [r for r in results if r.usd is not None]
    unavailable: list[str] = [r.symbol for r in results if r.usd is None]

    parts: list[str] = []

    if priced:
        lines = [
            (
                f"**{r.symbol}** 现在 {_fmt_usd(r.usd)}{_fmt_change(r.change_24h, zh)}"
                if zh
                else f"**{r.symbol}** is {_fmt_usd(r.usd)}{_fmt_change(r.change_24h, zh)}"
            )
            for r in priced
        ]
        parts.append("\n".join(lines))

        all_superex = all(r.source == "superex" for r in priced)
        if all_superex:
            parts.append(
                "以上为 SuperEx 平台实时价格,可能有秒级波动。"
                if zh
                else "Live prices from SuperEx, may move by the second."
            )
        else:
            # a fallback source answered at least one coin
            parts.append(
                "价格为实时行情(部分来自第三方数据源,仅供参考);"
                "以 SuperEx 平台实际成交价为准。"
                if zh
                else "Real-time quotes (some from a third-party source, for reference); "
                "SuperEx's own price is authoritative."
            )

    if unavailable:
        coins = "、".join(unavailable) if zh else ", ".join(unavailable)
        parts.append(
            f"{coins} 暂时查不到实时价格,请在 SuperEx 官网查看: {SUPEREX_MARKETS_URL}"
            if zh
            else f"No live price available for {coins}. Check SuperEx: {SUPEREX_MARKETS_URL}"
        )

    return "\n\n".join(parts)
