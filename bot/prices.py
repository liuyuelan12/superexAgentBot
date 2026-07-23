"""Real-time crypto spot prices for the support bot.

Why this exists: asked "现在 BTC 价格多少", the RAG bot retrieved a help-center
article whose worked example used 100000 as a placeholder price and reported it
as the live price — more than 50% above the real value. Price questions must
never be answered from the static index. This module intercepts them and returns
a real quote, or, for coins no public source covers (ET and other SuperEx-native
tokens), points the user to the live SuperEx page instead of inventing a number.

Source: CoinGecko — a neutral aggregator, not a competitor exchange — with
Binance as a reliability fallback when CoinGecko is unreachable. The answer is
framed as third-party reference data: SuperEx's own matching-engine price is
authoritative, and the bot says so, because we deliberately are not quoting
SuperEx's own feed (its public price endpoint is auth-gated / WebSocket-only).
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
    SUPEREX_MARKETS_URL,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Coin table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Coin:
    symbol: str  # canonical ticker shown to the user, e.g. "BTC"
    coingecko_id: Optional[str]  # None for coins no aggregator lists
    binance: Optional[str]  # Binance spot symbol, e.g. "BTCUSDT"
    native: bool = False  # a SuperEx-native token -> always point to SuperEx


# Kept small and hand-curated on purpose. An unknown ticker is better sent to the
# SuperEx page than resolved by a fuzzy lookup that could quote the wrong asset.
_COINS: tuple[Coin, ...] = (
    Coin("BTC", "bitcoin", "BTCUSDT"),
    Coin("ETH", "ethereum", "ETHUSDT"),
    Coin("USDT", "tether", None),
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
    Coin("XLM", "stellar", "XLMUSDT"),
    Coin("NEAR", "near", "NEARUSDT"),
    Coin("APT", "aptos", "APTUSDT"),
    Coin("FIL", "filecoin", "FILUSDT"),
    Coin("ARB", "arbitrum", "ARBUSDT"),
    Coin("OP", "optimism", "OPUSDT"),
    Coin("SUI", "sui", "SUIUSDT"),
    Coin("PEPE", "pepe", "PEPEUSDT"),
    Coin("WLD", "worldcoin-wld", "WLDUSDT"),
    Coin("INJ", "injective-protocol", "INJUSDT"),
    Coin("SEI", "sei-network", "SEIUSDT"),
    Coin("TIA", "celestia", "TIAUSDT"),
    Coin("RUNE", "thorchain", "RUNEUSDT"),
    Coin("AAVE", "aave", "AAVEUSDT"),
    Coin("ETC", "ethereum-classic", "ETCUSDT"),
    Coin("XMR", "monero", None),
    Coin("BONK", "bonk", "BONKUSDT"),
    Coin("WIF", "dogwifcoin", "WIFUSDT"),
    Coin("FLOKI", "floki", "FLOKIUSDT"),
    Coin("ORDI", "ordinals", "ORDIUSDT"),
    # SuperEx-native: no aggregator lists these, so route to the live SuperEx page.
    Coin("ET", None, None, native=True),
)

_BY_SYMBOL: dict[str, Coin] = {c.symbol: c for c in _COINS}

# alias (lowercased) -> canonical symbol
_ALIASES: dict[str, str] = {}
for _c in _COINS:
    _ALIASES[_c.symbol.lower()] = _c.symbol
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
        "pepe": "PEPE", "佩佩": "PEPE", "青蛙": "PEPE",
        "polygon": "MATIC",
        "cosmos": "ATOM",
        "arbitrum": "ARB",
        "optimism": "OP",
        "monero": "XMR", "门罗": "XMR", "門羅": "XMR",
        "superex": "ET", "superextoken": "ET", "平台币": "ET", "平台幣": "ET", "外星人": "ET",
    }
)


# --------------------------------------------------------------------------- #
# Intent detection (pure, deterministic — the router's 8B model is not reliable
# enough to gate this, and a table can be unit-tested offline)
# --------------------------------------------------------------------------- #

# Turns a superficially price-shaped question into a NON-price one: fees, the
# futures mark/index/last-price concepts (which CoinGecko cannot answer anyway),
# funding, liquidation, charts/history, and price predictions (out of scope).
_EXCLUDE_RE = re.compile(
    r"手续费|手續費|费率|費率|\bfee\b|commission"
    r"|标记价|標記價|指数价|指數價|最新价|最新價|mark\s*price|index\s*price|last\s*price"
    r"|资金费|資金費|funding"
    r"|强平|強平|爆仓|爆倉|清算|liquidat"
    r"|预测|預測|会涨|會漲|会跌|會跌|能涨|能漲|predict|target|走势|走勢|趋势|趨勢"
    r"|历史|歷史|昨天|去年|過去|过去|history|k\s*线|k\s*線|kline|candlestick|chart",
    re.IGNORECASE,
)

# Something that signals "tell me the value", in any of the bot's languages.
_PRICE_WORD_RE = re.compile(
    r"价格|價格|价钱|價錢|多少钱|多少錢|多少|现价|現價|币价|幣價|报价|報價|值多少|涨到|漲到|现在.*价|現在.*價"
    r"|price|worth|quote|how\s+much|what.*(cost|worth)|current.*(price|value)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PriceIntent:
    symbols: tuple[str, ...]  # canonical tickers, order preserved, deduped


def _extract_coins(text: str) -> list[str]:
    """Return canonical symbols mentioned in *text*, in order, deduped."""
    found: list[str] = []
    lowered = text.lower()
    for alias, symbol in _ALIASES.items():
        if symbol in found:
            continue
        if _contains_alias(lowered, text, alias):
            found.append(symbol)
    # order by first appearance so "BTC 和 ETH" reads naturally
    found.sort(key=lambda s: _first_pos(lowered, text, s))
    return found


def _alias_forms(symbol: str) -> list[str]:
    return [a for a, s in _ALIASES.items() if s == symbol]


def _contains_alias(lowered: str, original: str, alias: str) -> bool:
    if alias.isascii():
        # word-bounded so "eth" doesn't fire inside "ethernet" nor "et" inside "get"
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered) is not None
    return alias in original


def _first_pos(lowered: str, original: str, symbol: str) -> int:
    positions = []
    for alias in _alias_forms(symbol):
        hay = lowered if alias.isascii() else original
        idx = hay.find(alias)
        if idx >= 0:
            positions.append(idx)
    return min(positions) if positions else 1 << 30


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
    usd: Optional[float]  # None when native / unavailable
    change_24h: Optional[float]  # percent, may be None even with a price
    source: str  # "coingecko" | "binance" | "native" | "unavailable"


CoingeckoFetch = Callable[[list[str]], Awaitable[dict[str, dict]]]
BinanceFetch = Callable[[str], Awaitable[Optional[dict]]]


class PriceService:
    """Fetch spot prices with a short cache and a CoinGecko→Binance fallback."""

    def __init__(
        self,
        *,
        cache_ttl: float = PRICE_CACHE_TTL,
        timeout: float = PRICE_HTTP_TIMEOUT,
        coingecko_fetch: Optional[CoingeckoFetch] = None,
        binance_fetch: Optional[BinanceFetch] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._now = now
        self._cache: dict[str, tuple[float, PriceResult]] = {}
        self._cg = coingecko_fetch or self._default_coingecko
        self._bn = binance_fetch or self._default_binance

    async def get_prices(self, symbols: list[str]) -> list[PriceResult]:
        out: list[PriceResult] = []
        need: list[str] = []
        for sym in symbols:
            cached = self._cached(sym)
            if cached is not None:
                out.append(cached)
            else:
                need.append(sym)

        fetched = await self._fetch(need) if need else {}
        for sym in need:
            out.append(fetched.get(sym, PriceResult(sym, None, None, "unavailable")))

        # restore original order
        order = {s: i for i, s in enumerate(symbols)}
        out.sort(key=lambda r: order.get(r.symbol, 1 << 30))
        return out

    def _cached(self, symbol: str) -> Optional[PriceResult]:
        entry = self._cache.get(symbol)
        if entry and entry[0] > self._now():
            return entry[1]
        return None

    def _store(self, result: PriceResult) -> None:
        # never cache a transient failure — retry it next time
        if result.source in ("coingecko", "binance", "native"):
            self._cache[result.symbol] = (self._now() + self._ttl, result)

    async def _fetch(self, symbols: list[str]) -> dict[str, PriceResult]:
        results: dict[str, PriceResult] = {}

        native = [s for s in symbols if _BY_SYMBOL.get(s) and _BY_SYMBOL[s].native]
        for s in native:
            results[s] = PriceResult(s, None, None, "native")

        cg_ids = {
            _BY_SYMBOL[s].coingecko_id: s
            for s in symbols
            if _BY_SYMBOL.get(s) and _BY_SYMBOL[s].coingecko_id
        }
        if cg_ids:
            try:
                data = await self._cg(list(cg_ids.keys()))
            except Exception:  # noqa: BLE001 - price lookup must degrade, not crash
                logger.warning("CoinGecko price fetch failed", exc_info=True)
                data = {}
            for cid, sym in cg_ids.items():
                row = data.get(cid)
                if row and row.get("usd") is not None:
                    results[sym] = PriceResult(
                        sym, float(row["usd"]), _opt_float(row.get("usd_24h_change")), "coingecko"
                    )

        # Binance reliability fallback for anything CoinGecko didn't return.
        missing = [
            s
            for s in symbols
            if s not in results and _BY_SYMBOL.get(s) and _BY_SYMBOL[s].binance
        ]
        if missing:
            binance_rows = await asyncio.gather(
                *(self._safe_binance(_BY_SYMBOL[s].binance) for s in missing),
                return_exceptions=False,
            )
            for sym, row in zip(missing, binance_rows):
                if row and row.get("price") is not None:
                    results[sym] = PriceResult(
                        sym, float(row["price"]), _opt_float(row.get("change")), "binance"
                    )

        for r in results.values():
            self._store(r)
        return results

    async def _safe_binance(self, symbol: str) -> Optional[dict]:
        try:
            return await self._bn(symbol)
        except Exception:  # noqa: BLE001
            logger.warning("Binance fallback failed for %s", symbol, exc_info=True)
            return None

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
            return {
                "price": data.get("lastPrice"),
                "change": data.get("priceChangePercent"),
            }


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
    # sub-dollar coins (SHIB, PEPE…) need more precision
    return f"${value:,.8f}".rstrip("0").rstrip(".")


def _fmt_change(pct: Optional[float], lang: str) -> str:
    if pct is None:
        return ""
    sign = "+" if pct >= 0 else ""
    if lang.startswith("zh"):
        return f",24 小时 {sign}{pct:.2f}%"
    return f", 24h {sign}{pct:.2f}%"


def format_price_answer(results: list[PriceResult], lang: str) -> str:
    zh = lang.startswith("zh")
    lines: list[str] = []
    native: list[str] = []

    for r in results:
        if r.source == "native":
            native.append(r.symbol)
        elif r.usd is not None:
            if zh:
                lines.append(f"**{r.symbol}** 现在约 {_fmt_usd(r.usd)}{_fmt_change(r.change_24h, lang)}")
            else:
                lines.append(f"**{r.symbol}** is about {_fmt_usd(r.usd)}{_fmt_change(r.change_24h, lang)}")
        else:
            native.append(r.symbol)  # unavailable -> treat like native: send to SuperEx

    parts: list[str] = []
    if lines:
        parts.append("\n".join(lines))
        if zh:
            parts.append(
                "以上为第三方聚合的实时市场行情,仅供参考;"
                "SuperEx 平台的实际成交价请以官网为准。"
            )
        else:
            parts.append(
                "These are real-time market quotes from a third-party aggregator, "
                "for reference only. SuperEx's own trading price is authoritative — "
                "check the platform for the exact figure."
            )
    if native:
        coins = "、".join(native) if zh else ", ".join(native)
        if zh:
            parts.append(
                f"{coins} 的实时价格第三方数据源查不到,"
                f"请直接在 SuperEx 官网查看: {SUPEREX_MARKETS_URL}"
            )
        else:
            parts.append(
                f"No third-party feed covers {coins}. "
                f"Please check the live price on SuperEx: {SUPEREX_MARKETS_URL}"
            )
    return "\n\n".join(parts)
