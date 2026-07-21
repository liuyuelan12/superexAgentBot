"""Unit tests for numeric-claim extraction and conflict candidate pairing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.conflicts import extract_claims, find_candidates, topics_of  # noqa: E402
from kb.document import Document  # noqa: E402


def doc(text, *, type="cs_faq", source="x.md", lang="zh-CN", **extras):
    return Document(
        text=text, source=source, lang=lang, type=type, section="", extras=extras
    )


# ── claim extraction ───────────────────────────────────────────────────────────


def test_extracts_percentage():
    claims = extract_claims("现货交易费率 0.2%")
    assert (claims[0].value, claims[0].unit) == (0.2, "%")


def test_extracts_usdt_amount_with_thousands_separator():
    claims = extract_claims("每日限额 1,000,000 USDT")
    assert claims[0].value == 1000000.0
    assert claims[0].unit == "USDT"


def test_normalises_equivalent_units():
    assert extract_claims("最高 150 倍")[0].unit == "倍"
    assert extract_claims("up to 150x")[0].unit == "倍"
    assert extract_claims("费率 0.1％")[0].unit == "%"


def test_normalises_duration_units():
    assert extract_claims("24 小時內到賬")[0].unit == "小时"
    assert extract_claims("within 24 hours")[0].unit == "小时"


def test_number_without_unit_is_ignored():
    assert extract_claims("第 3 步点击确认") == []


def test_claim_keeps_its_sentence_for_evidence():
    claims = extract_claims("充值免费。提现手续费 1 USDT 起。")
    assert "提现手续费" in claims[0].sentence


def test_no_claims_in_prose():
    assert extract_claims("SuperEx 是一家 Web3 交易所") == []


# ── topic tagging ──────────────────────────────────────────────────────────────


def test_topic_detection_is_bilingual():
    assert "fee" in topics_of("手续费说明")
    assert "fee" in topics_of("Maker and Taker fee")
    assert "withdraw" in topics_of("提現流程")


def test_unrelated_text_has_no_topic():
    assert topics_of("今天天气不错") == set()


# ── candidate pairing ──────────────────────────────────────────────────────────


def test_same_value_is_not_a_conflict():
    old = [doc("现货手续费 0.1%")]
    new = [doc("现货手续费 0.1%", type="help_center", lang="zh-TW")]
    assert find_candidates(old, new) == []


def test_differing_value_on_shared_topic_is_a_candidate():
    old = [doc("现货交易手续费 0.2%")]
    new = [doc("现货交易手续费 0.1%", type="help_center", lang="zh-TW")]
    cands = find_candidates(old, new)
    assert len(cands) == 1
    assert cands[0].unit == "%"
    assert cands[0].old_values() == {0.2}
    assert cands[0].new_values() == {0.1}


def test_shared_unit_without_shared_topic_is_not_a_candidate():
    """0.2% about fees and 0.2% about slippage are unrelated."""
    old = [doc("提现手续费 0.2%")]
    new = [doc("理财年化 0.5%", type="help_center", lang="zh-TW")]
    assert find_candidates(old, new) == []


def test_docs_without_numbers_are_skipped():
    old = [doc("手续费说明请看官网")]
    new = [doc("手续费说明请看官网", type="help_center", lang="zh-TW")]
    assert find_candidates(old, new) == []


def test_empty_inputs_do_not_raise():
    assert find_candidates([], []) == []
    assert find_candidates([doc("费率 0.1%")], []) == []


def test_same_unit_in_unrelated_sentences_is_not_a_candidate():
    """Regression: a fee sentence and a liquidation sentence both contain '%'.

    Pairing them let the judge invent a relationship and fabricate a conflict.
    """
    old = [doc("強制平倉，當保證金率≤100%時觸發。交易手續費 Maker 0.02%。")]
    new = [
        doc(
            "交易手續費：Maker 0.02%，Taker 0.06%。強平說明另見他處。",
            type="help_center",
            lang="zh-TW",
        )
    ]
    cands = find_candidates(old, new)
    # The only shared-topic claim pair (fee 0.02% vs 0.02%) agrees, so nothing
    # should be reported — and liquidation must not be matched against fees.
    assert [c.topic for c in cands if c.topic == "liquidation"] == []


def test_claim_sentences_must_share_the_topic_not_just_the_document():
    old = [doc("提現手續費 1 USDT。杠杆最高 10 倍。")]
    new = [doc("提現手續費 2 USDT。杠杆最高 150 倍。", type="help_center", lang="zh-TW")]
    cands = find_candidates(old, new)
    assert cands, "a genuine same-topic divergence should still be found"
    for c in cands:
        old_s = " ".join(x.sentence for x in c.old_claims)
        new_s = " ".join(x.sentence for x in c.new_claims)
        assert topics_of(old_s) & topics_of(new_s), "paired sentences must share a topic"


def test_overlapping_value_sets_are_treated_as_agreement():
    """New doc lists several tiers including the old value -> not a conflict."""
    old = [doc("现货手续费 0.1%")]
    new = [doc("现货手续费 VIP0 0.1% VIP1 0.09%", type="help_center", lang="zh-TW")]
    assert find_candidates(old, new) == []
