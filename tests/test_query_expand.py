"""Tests for bilingual query expansion.

The anchor list is what lets an English question reach a Chinese-only page, so
these tests pin the behaviour that regressed in production: "if i am a superex
vip6, what'd be my fee rate" retrieved no VIP tier table and the model invented
rates that appear nowhere in the corpus.
"""

from __future__ import annotations

import pytest

from kb.query_expand import MAX_ANCHORS, anchors_for, expand_query


class TestDirection:
    def test_english_query_gets_chinese_anchors(self):
        anchors = anchors_for("superex vip6 fee rate")
        assert "手续费" in anchors
        assert "费率" in anchors

    def test_chinese_query_gets_english_anchors(self):
        anchors = anchors_for("vip6 的手续费是多少")
        assert any(a == "fee" for a in anchors)
        assert all(not _is_cjk_word(a) for a in anchors)

    def test_the_production_failure_case(self):
        """The exact message that made the bot hallucinate 0.07% / 0.015%."""
        expanded = expand_query("if i am a superex vip6, what'd be my fee rate")
        assert "手续费" in expanded
        assert "费率" in expanded
        # the original wording must survive — it carries the tier the user asked for
        assert "vip6" in expanded


class TestCoverage:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("how do I withdraw USDT", "提现"),
            ("my deposit has not arrived", "充值"),
            ("what is cross margin", "全仓"),
            ("will I get liquidated", "爆仓"),
            ("how is the funding rate calculated", "资金费率"),
            ("my assets are frozen", "冻结"),
            ("how to pass KYC", "实名认证"),
            ("lost my 2fa authenticator", "谷歌验证器"),
            ("what is the referral rebate", "返佣"),
            ("how does copy trading work", "跟单"),
            # gaps found by auditing the 83-question standard set
            ("I forgot my login password", "密码"),
            ("what is anti-phishing code", "钓鱼"),
            ("how do I create and manage API keys", "API 密钥"),
            ("I withdrew to the wrong address", "充错网络"),
            ("how to buy or sell coins", "买币"),
            ("difference between mark price and index price", "标记价格"),
            ("what is the maintenance margin rate", "维持保证金"),
            ("how do I contact human support", "客服"),
            ("where can I see announcements and new listings", "公告"),
            ("what is impermanent loss", "无常损失"),
            ("how to connect my web3 wallet", "钱包"),
            ("how do I change my bound email", "邮箱"),
        ],
    )
    def test_common_english_questions_reach_chinese_pages(self, query, expected):
        assert expected in expand_query(query)


class TestSafety:
    def test_no_anchors_when_topic_is_unknown(self):
        query = "hello there"
        assert expand_query(query) == query

    def test_empty_query_is_untouched(self):
        assert expand_query("") == ""
        assert expand_query("   ") == ""

    def test_anchor_count_is_capped(self):
        # a query tripping many concepts at once must not balloon: every added
        # token that misses dilutes BM25 for the ones that hit
        query = "vip fee withdraw deposit kyc leverage spot futures grid earn order"
        assert len(anchors_for(query)) <= MAX_ANCHORS

    def test_already_present_terms_are_not_duplicated(self):
        expanded = expand_query("手续费 费率 fee rate")
        assert expanded.count("手续费") == 1
        assert expanded.count("费率") == 1

    def test_short_trigger_does_not_fire_inside_another_word(self):
        # "et token" rather than "et", so "get"/"market"/"budget" stay quiet
        assert "平台币" not in expand_query("how do I get started")

    @pytest.mark.parametrize(
        "phrase",
        [
            "how do I get started",
            "thanks a lot",
            "good morning everyone",
            "can you help me please",
        ],
    )
    def test_chit_chat_is_not_expanded(self, phrase):
        assert expand_query(phrase) == phrase

    def test_triggers_avoid_the_known_substring_traps(self):
        # bare "event" would fire inside "prevent"; bare "stake" inside "mistake";
        # bare "api" inside "rapid" — each concept uses a longer form instead.
        assert expand_query("prevent this") == "prevent this"
        assert expand_query("that was a mistake") == "that was a mistake"
        assert expand_query("rapid growth") == "rapid growth"

    def test_expansion_is_pure(self):
        query = "superex vip6 fee rate"
        first, second = expand_query(query), expand_query(query)
        assert first == second
        assert query in first


def _is_cjk_word(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)
