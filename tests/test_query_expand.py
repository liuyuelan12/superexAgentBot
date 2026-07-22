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

    def test_expansion_is_pure(self):
        query = "superex vip6 fee rate"
        first, second = expand_query(query), expand_query(query)
        assert first == second
        assert query in first


def _is_cjk_word(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)
