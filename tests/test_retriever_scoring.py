"""Unit tests for retrieval ranking: dedup, link-only penalty, doc-type weights.

These exercise the pure `score_pool` function, so no chroma index or embedding
model is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.chunker import is_link_only  # noqa: E402
from kb.retriever import score_pool  # noqa: E402


def candidate(doc_id, text, *, vec=0.6, bm25=0.0, lang="zh-CN", type="cs_faq", **meta):
    return {
        "doc_id": doc_id,
        "text": text,
        "vec_sim": vec,
        "bm25_norm": bm25,
        "meta": {"lang": lang, "type": type, "source": f"{doc_id}.md", **meta},
    }


def pool(*candidates):
    return {c["doc_id"]: c for c in candidates}


# ── dedup ──────────────────────────────────────────────────────────────────────


def test_identical_text_across_languages_collapses_to_one_hit():
    """The CSV loader emits zh-TW and zh-CN rows with identical text."""
    hits = score_pool(
        pool(
            candidate("tw", "SuperEx交易费率", lang="zh-TW", vec=0.8),
            candidate("cn", "SuperEx交易费率", lang="zh-CN", vec=0.8),
            candidate("other", "完全不同的内容", vec=0.7),
        ),
        lang_boost="zh",
        top_k=5,
    )
    assert len(hits) == 2
    assert [h.text for h in hits] == ["SuperEx交易费率", "完全不同的内容"]


def test_dedup_keeps_the_higher_scoring_duplicate():
    hits = score_pool(
        pool(
            candidate("low", "同样的话", vec=0.4),
            candidate("high", "同样的话", vec=0.9),
        ),
        lang_boost=None,
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0].doc_id == "high"


def test_dedup_ignores_whitespace_differences():
    hits = score_pool(
        pool(
            candidate("a", "费率 说明"),
            candidate("b", "费率说明"),
        ),
        lang_boost=None,
        top_k=5,
    )
    assert len(hits) == 1


def test_dedup_frees_slots_for_real_content():
    """Regression: duplicates used to eat TOP_K slots, starving the answer."""
    hits = score_pool(
        pool(
            candidate("d1", "重复文案", vec=0.9),
            candidate("d2", "重复文案", vec=0.9),
            candidate("d3", "重复文案", vec=0.9),
            candidate("real1", "VIP6 现货 Maker 0.06%", vec=0.5),
            candidate("real2", "VIP6 合约 Maker 0.014%", vec=0.5),
        ),
        lang_boost=None,
        top_k=3,
    )
    texts = [h.text for h in hits]
    assert "VIP6 现货 Maker 0.06%" in texts
    assert "VIP6 合约 Maker 0.014%" in texts


# ── link-only penalty ──────────────────────────────────────────────────────────


def test_link_only_chunk_is_demoted_below_substantive_content():
    hits = score_pool(
        pool(
            candidate("stub", "请参考以下链接", vec=0.70, link_only=True),
            candidate("real", "VIP6 现货 Maker 0.06% / Taker 0.08%", vec=0.60),
        ),
        lang_boost=None,
        top_k=5,
    )
    assert hits[0].doc_id == "real"
    assert hits[1].doc_id == "stub"


def test_link_only_chunk_still_surfaces_when_nothing_else_matches():
    """Demote, don't hide — a link beats no answer at all."""
    hits = score_pool(
        pool(candidate("stub", "请参考以下链接", vec=0.7, link_only=True)),
        lang_boost=None,
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0].doc_id == "stub"


def test_penalties_do_not_touch_vec_sim():
    """SIM_THRESHOLD gates refusal on vec_sim, so penalties must leave it alone."""
    hits = score_pool(
        pool(candidate("stub", "请参考以下链接", vec=0.72, link_only=True, type="cs-script")),
        lang_boost=None,
        top_k=5,
    )
    assert hits[0].vec_sim == 0.72
    assert hits[0].score < 0.72


# ── doc-type weights ───────────────────────────────────────────────────────────


def test_marketing_copy_is_damped_below_support_faq():
    hits = score_pool(
        pool(
            candidate("promo", "SuperEx 交易费率最低", vec=0.70, type="cs-script"),
            candidate("faq", "现货挂单 0.2%", vec=0.66, type="cs_faq"),
        ),
        lang_boost=None,
        top_k=5,
    )
    assert hits[0].doc_id == "faq"


def test_unknown_doc_type_is_not_penalised():
    hits = score_pool(
        pool(candidate("w", "wiki 内容", vec=0.5, bm25=0.0, type="wiki")),
        lang_boost=None,
        top_k=5,
    )
    assert hits[0].score == 0.7 * 0.5  # HYBRID_VECTOR_WEIGHT, no multipliers


# ── lang boost ─────────────────────────────────────────────────────────────────


def test_lang_boost_prefers_the_users_language():
    hits = score_pool(
        pool(
            candidate("en", "fee info", lang="en", vec=0.6),
            candidate("zh", "费率信息", lang="zh-CN", vec=0.6),
        ),
        lang_boost="zh",
        top_k=5,
    )
    assert hits[0].doc_id == "zh"


def test_rows_without_text_are_skipped():
    hits = score_pool(
        {"x": {"doc_id": "x", "text": None, "vec_sim": 0.9, "bm25_norm": 0.9, "meta": {}}},
        lang_boost=None,
        top_k=5,
    )
    assert hits == []


# ── is_link_only heuristic ─────────────────────────────────────────────────────


def test_detects_the_vip_fee_stub():
    text = (
        "## 费用\n\n### VIP 等级和手续费折扣\n\n"
        "请参考以下链接：https://www.superex.com/userCenter/fee/trading\n"
    )
    assert is_link_only(text) is True


def test_detects_button_only_marketing_copy():
    assert is_link_only("【开始注册】 【APP下载】 【加入代理商】") is True


def test_real_tutorial_step_is_not_flagged():
    """43-char residue — the closest true negative in the corpus."""
    text = (
        "# 如何注册帳戶（網頁端）\n\n1. 點擊頁面右上方【註冊/登錄】。"
        "您可以選擇使用手機號或郵箱進行注册。按要求輸入【郵箱/手機號碼】【密碼】，"
        "勾選【我已閱讀並同意《SuperEx服務條款》】—輸入【郵箱/手機驗證碼】並提交，完成註冊即可。"
    )
    assert is_link_only(text) is False


def test_text_without_url_or_button_is_never_flagged():
    assert is_link_only("现货交易费率：挂单 0.2% / 吃单 0.2%") is False
