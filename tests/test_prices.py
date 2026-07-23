"""Tests for the real-time price tool.

Pins the behaviour that motivated it: "现在 BTC 价格多少" must be recognised as a
price question and answered from a live feed (SuperEx's own, primarily), never
from the static index; fee / concept / prediction questions that merely mention
a coin or the word 价格 must NOT be hijacked by the price path.
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
            ("ET 现在价格多少", "ET"),        # SuperEx-native, via bare-ticker path
            ("ONDO 现在多少钱", "ONDO"),      # long-tail coin not in the alias table
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


class TestDetectNegative:
    @pytest.mark.parametrize(
        "text",
        [
            "现货交易手续费是多少",
            "VIP6 手续费多少",
            "标记价格和指数价格的区别",
            "资金费率怎么算",
            "BTC 会涨到多少",
            "btc 手续费多少",
            "充值最低多少",
            "how do I withdraw BTC",
            "什么是比特币",
            "BTC 的 K 线怎么看",
        ],
    )
    def test_non_price_questions_are_ignored(self, text):
        assert detect_price_intent(text) is None

    def test_bare_coin_without_price_word_is_ignored(self):
        assert detect_price_intent("BTC") is None

    def test_coin_substring_does_not_false_trigger(self):
        assert detect_price_intent("how do I get the price down") is None
        assert detect_price_intent("ethernet cable price") is None

    def test_acronyms_are_not_treated_as_coins(self):
        # "KYC" / "VIP" are uppercase but not coins; with no other coin, no intent
        assert detect_price_intent("KYC 要多少钱") is None
        assert detect_price_intent("VIP 等级价格") is None


def _fake_service(superex=None, coingecko=None, binance=None):
    async def sx():
        return superex or {}

    async def cg(ids):
        return coingecko or {}

    async def bn(symbol):
        return (binance or {}).get(symbol)

    clock = [1000.0]
    svc = PriceService(
        superex_fetch=sx,
        coingecko_fetch=cg,
        binance_fetch=bn,
        now=lambda: clock[0],
    )
    return svc, clock


class TestFetch:
    def test_superex_is_the_primary_source(self):
        svc, _ = _fake_service(
            superex={"BTC": {"usd": "65160.90", "change": "-1.22"}}
        )
        [r] = asyncio.run(svc.get_prices(["BTC"]))
        assert r.source == "superex"
        assert r.usd == 65160.90
        assert r.change_24h == -1.22

    def test_native_token_comes_from_superex(self):
        # ET has no aggregator coverage — SuperEx's own feed is exactly why it works
        svc, _ = _fake_service(superex={"ET": {"usd": "2.0273", "change": "-1.22"}})
        [r] = asyncio.run(svc.get_prices(["ET"]))
        assert r.source == "superex"
        assert r.usd == 2.0273

    def test_coingecko_fallback_when_superex_lacks_coin(self):
        svc, _ = _fake_service(
            superex={},  # SuperEx returned nothing for BTC
            coingecko={"bitcoin": {"usd": 64000.0, "usd_24h_change": 2.1}},
        )
        [r] = asyncio.run(svc.get_prices(["BTC"]))
        assert r.source == "coingecko"
        assert r.usd == 64000.0

    def test_binance_fallback_when_superex_and_coingecko_empty(self):
        svc, _ = _fake_service(
            superex={}, coingecko={}, binance={"BTCUSDT": {"price": "63000", "change": "1.0"}}
        )
        [r] = asyncio.run(svc.get_prices(["BTC"]))
        assert r.source == "binance"
        assert r.usd == 63000.0

    def test_unavailable_when_all_sources_fail(self):
        svc, _ = _fake_service(superex={}, coingecko={}, binance={})
        [r] = asyncio.run(svc.get_prices(["XYZ"]))
        assert r.source == "unavailable"
        assert r.usd is None

    def test_superex_map_is_fetched_once_and_cached(self):
        calls = {"n": 0}

        async def sx():
            calls["n"] += 1
            return {"BTC": {"usd": "65000", "change": "0"}, "ETH": {"usd": "1900", "change": "0"}}

        svc = PriceService(superex_fetch=sx, now=lambda: 1000.0)
        asyncio.run(svc.get_prices(["BTC"]))
        asyncio.run(svc.get_prices(["ETH"]))  # different coin, same cached map
        assert calls["n"] == 1

    def test_stale_superex_map_is_refetched_after_ttl(self):
        calls = {"n": 0}
        clock = [1000.0]

        async def sx():
            calls["n"] += 1
            return {"BTC": {"usd": "65000", "change": "0"}}

        svc = PriceService(superex_fetch=sx, cache_ttl=45, now=lambda: clock[0])
        asyncio.run(svc.get_prices(["BTC"]))
        clock[0] += 100  # past the TTL
        asyncio.run(svc.get_prices(["BTC"]))
        assert calls["n"] == 2

    def test_order_is_preserved(self):
        svc, _ = _fake_service(
            superex={"BTC": {"usd": "65000", "change": "0"}, "ETH": {"usd": "1900", "change": "0"}}
        )
        results = asyncio.run(svc.get_prices(["ETH", "BTC"]))
        assert [r.symbol for r in results] == ["ETH", "BTC"]


class TestFormat:
    def test_superex_answer_has_price_and_platform_caveat(self):
        out = format_price_answer(
            [PriceResult("BTC", 65160.90, -1.22, "superex")], "zh"
        )
        assert "BTC" in out
        assert "$65,160.90" in out
        assert "SuperEx 平台实时价格" in out
        assert "100000" not in out  # the old hallucinated number must be nowhere

    def test_fallback_source_carries_third_party_caveat(self):
        out = format_price_answer(
            [PriceResult("BTC", 64000.0, None, "coingecko")], "zh"
        )
        assert "第三方" in out
        assert "以 SuperEx 平台实际成交价为准" in out

    def test_en_answer(self):
        out = format_price_answer([PriceResult("ETH", 1900.0, 2.0, "superex")], "en")
        assert "ETH" in out
        assert "$1,900.00" in out
        assert "SuperEx" in out

    def test_sub_dollar_precision(self):
        out = format_price_answer(
            [PriceResult("PEPE", 0.00000812, None, "superex")], "en"
        )
        assert "0.00000812" in out

    def test_unavailable_points_to_superex(self):
        out = format_price_answer([PriceResult("XYZ", None, None, "unavailable")], "zh")
        assert "superex.com" in out
        assert "XYZ" in out

    def test_mixed_priced_and_unavailable(self):
        out = format_price_answer(
            [
                PriceResult("BTC", 65000.0, None, "superex"),
                PriceResult("XYZ", None, None, "unavailable"),
            ],
            "zh",
        )
        assert "$65,000.00" in out
        assert "XYZ" in out
        assert "superex.com" in out
