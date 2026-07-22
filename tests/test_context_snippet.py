"""Retrieved chunks must reach the model intact.

The context builder used to clip every chunk at 1200 characters, which hit 51%
of the corpus. The VIP fee tables run past that, so VIP 9 was cut off and the
bot answered that it only had "VIP levels 0 through 8" — a refusal caused by
truncation, not by missing data.
"""

from __future__ import annotations

from config import CONTEXT_SNIPPET_CHARS, TOP_K
from bot.handlers import _format_hits
from kb.retriever import Hit


def _hit(text: str, name: str = "fee-schedule.md") -> Hit:
    source = f"raw/客服/补充/{name}"
    return Hit(
        doc_id="d1",
        text=text,
        source=source,
        section="VIP 各等级现货交易手续费",
        lang="zh-CN",
        score=0.9,
        vec_sim=0.8,
        bm25_norm=0.7,
        metadata={"source": source, "type": "cs_faq"},
    )


class TestSnippetLength:
    def test_limit_clears_the_longest_chunk_in_the_corpus(self):
        # measured maximum was 3454 characters; the limit must sit above it or
        # chunks are silently clipped again
        assert CONTEXT_SNIPPET_CHARS >= 3500

    def test_a_full_vip_table_survives_intact(self):
        table = "\n".join(
            f"- VIP {n} 现货手续费:基础费率 Maker 0.0{n}00% / Taker 0.0{n}50%;"
            f"ET 抵扣后 Maker 0.0{n}00% / Taker 0.0{n}40%。达标:30 天现货成交量 "
            f"≥ {n}0,000,000 USDT 且 ET 持仓 ≥ {n}0,000。"
            for n in range(10)
        )
        text = "## VIP 各等级现货交易手续费\n\n" + table
        assert len(text) > 1200, "fixture must exceed the old limit to be meaningful"

        out = _format_hits([_hit(text)])
        assert "VIP 9 现货手续费" in out, "the last tier must not be cut off"
        assert "…" not in out

    def test_worst_case_context_stays_reasonable(self):
        hits = [_hit("字" * CONTEXT_SNIPPET_CHARS) for _ in range(TOP_K)]
        # a bound worth keeping visible: every model in use handles this, but a
        # future TOP_K bump should have to look at this number
        assert len(_format_hits(hits)) < 20_000
