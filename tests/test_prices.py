"""Tests for the real-time price tool.

Pins the behaviour that motivated it: "现在 BTC 价格多少" must be recognised as a
price question and never answered from the static index, while fee / concept /
prediction questions that merely mention a coin or the word 价格 must NOT be
hijacked by the price path.
"""

from __future__ import annotations

import asyncio

import pytest

from bot.prices import (
    PriceResult,
    PriceService,
    detect_price_intent,
    format_price_answer,
)


class TestDetectPositive:
    @pytest.mark.parametrize(
        ("text", "symbol"),
        [
            ("现在btc价格多少", "BTC"),
            ("BTC 现在多少钱", "BTC"),
            ("以太坊多少钱", "ETH"),
            ("what is the btc price", "BTC"),
            ("how much is bitcoin right now", "BTC"),
            ("price of ETH", "ETH"),
            ("狗狗币现在价格", "DOGE"),
            ("SOL 报价", "SOL"),
        ],
    )
    def test_real_price_questions_are_detected(self, text, symbol):
        intent = detect_price_intent(text)
        assert intent is not None
        assert symbol in intent.symbols

    def test_multiple_coins_in_order(self):
        intent = detect_price_intent("BTC 和 ETH 现在价格多少")
        assert intent is not None
        assert intent.symbols == ("BTC", "ETH")

    def test_native_token_is_detected(self):
        intent = detect_price_intent("ET 现在价格多少")
        assert intent is not None
        assert "ET" in intent.symbols


class TestDetectNegative:
    @pytest.mark.parametrize(
        "text",
        [
            "现货交易手续费是多少",          # fee, not price
            "VIP6 手续费多少",               # fee
            "标记价格和指数价格的区别",       # futures concept, not a spot quote
            "资金费率怎么算",                # funding
            "BTC 会涨到多少",                # prediction
            "btc 手续费多少",                # coin + fee -> still a fee question
            "充值最低多少",                  # 多少 but no coin
            "how do I withdraw BTC",         # coin but no price word
            "什么是比特币",                  # concept, no price word
            "BTC 的 K 线怎么看",             # chart
        ],
    )
    def test_non_price_questions_are_ignored(self, text):
        assert detect_price_intent(text) is None

    def test_bare_coin_without_price_word_is_ignored(self):
        assert detect_price_intent("BTC") is None

    def test_coin_substring_does_not_false_trigger(self):
        # "et" inside "get"/"target", "eth" inside "ethernet"
        assert detect_price_intent("how do I get the price down") is None
        assert detect_price_intent("ethernet cable price") is None


def _fake_service(coingecko=None, binance=None):
    async def cg(ids):
        return coingecko or {}

    async def bn(symbol):
        return (binance or {}).get(symbol)

    clock = [1000.0]
    return PriceService(coingecko_fetch=cg, binance_fetch=bn, now=lambda: clock[0]), clock


class TestFetch:
    def test_coingecko_price_is_returned(self):
        svc, _ = _fake_service(
            coingecko={"bitcoin": {"usd": 65000.0, "usd_24h_change": -1.5}}
        )
        [r] = asyncio.run(svc.get_prices(["BTC"]))
        assert r.symbol == "BTC"
        assert r.usd == 65000.0
        assert r.change_24h == -1.5
        assert r.source == "coingecko"

    def test_binance_fallback_when_coingecko_empty(self):
        svc, _ = _fake_service(
            coingecko={}, binance={"BTCUSDT": {"price": "64000", "change": "2.1"}}
        )
        [r] = asyncio.run(svc.get_prices(["BTC"]))
        assert r.usd == 64000.0
        assert r.source == "binance"

    def test_native_token_never_hits_the_network(self):
        called = {"cg": False}

        async def cg(ids):
            called["cg"] = True
            return {}

        svc = PriceService(coingecko_fetch=cg, binance_fetch=lambda s: None)
        [r] = asyncio.run(svc.get_prices(["ET"]))
        assert r.source == "native"
        assert r.usd is None

    def test_unavailable_when_both_sources_fail(self):
        svc, _ = _fake_service(coingecko={}, binance={})
        [r] = asyncio.run(svc.get_prices(["BTC"]))
        assert r.source == "unavailable"
        assert r.usd is None

    def test_cache_hit_avoids_refetch(self):
        calls = {"n": 0}

        async def cg(ids):
            calls["n"] += 1
            return {"bitcoin": {"usd": 65000.0, "usd_24h_change": 0.0}}

        svc = PriceService(coingecko_fetch=cg, binance_fetch=lambda s: None, cache_ttl=45)
        asyncio.run(svc.get_prices(["BTC"]))
        asyncio.run(svc.get_prices(["BTC"]))
        assert calls["n"] == 1  # second call served from cache

    def test_failure_is_not_cached(self):
        state = {"n": 0}

        async def cg(ids):
            state["n"] += 1
            return {} if state["n"] == 1 else {"bitcoin": {"usd": 65000.0}}

        svc = PriceService(coingecko_fetch=cg, binance_fetch=lambda s: None)
        r1 = asyncio.run(svc.get_prices(["BTC"]))[0]
        r2 = asyncio.run(svc.get_prices(["BTC"]))[0]
        assert r1.source == "unavailable"
        assert r2.source == "coingecko"  # retried, not stuck on the cached failure


class TestFormat:
    def test_zh_answer_has_price_and_caveat(self):
        out = format_price_answer(
            [PriceResult("BTC", 65000.0, -1.5, "coingecko")], "zh"
        )
        assert "BTC" in out
        assert "$65,000.00" in out
        assert "以官网为准" in out
        assert "100000" not in out  # the old hallucinated number must be nowhere

    def test_en_answer(self):
        out = format_price_answer(
            [PriceResult("ETH", 1900.0, 2.0, "coingecko")], "en"
        )
        assert "ETH" in out
        assert "$1,900.00" in out
        assert "authoritative" in out

    def test_native_points_to_superex(self):
        out = format_price_answer([PriceResult("ET", None, None, "native")], "zh")
        assert "superex.com" in out
        assert "ET" in out

    def test_sub_dollar_precision(self):
        out = format_price_answer(
            [PriceResult("PEPE", 0.00000812, None, "coingecko")], "en"
        )
        assert "0.00000812" in out

    def test_unavailable_falls_back_to_superex_link(self):
        out = format_price_answer([PriceResult("BTC", None, None, "unavailable")], "zh")
        assert "superex.com" in out
