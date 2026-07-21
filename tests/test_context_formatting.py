"""Tests for how retrieved chunks are presented to the answering model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.handlers import _format_hits  # noqa: E402
from kb.retriever import Hit  # noqa: E402


def hit(text="内容", *, source="a.md", lang="zh-CN", **meta):
    return Hit(
        doc_id="d",
        text=text,
        source=source,
        section="",
        lang=lang,
        score=0.5,
        vec_sim=0.5,
        bm25_norm=0.0,
        metadata={"source": source, "lang": lang, **meta},
    )


def test_official_chunks_are_tagged_with_update_date():
    out = _format_hits(
        [hit(authority="official", updated_at="2025-03-13T09:00:00Z", type="help_center")]
    )
    assert "[OFFICIAL, updated 2025-03-13]" in out


def test_official_without_date_still_marked_official():
    out = _format_hits([hit(authority="official", type="help_center")])
    assert "[OFFICIAL]" in out


def test_internal_sources_are_marked_as_notes():
    out = _format_hits([hit(type="cs_faq")])
    assert "[internal note]" in out
    assert "OFFICIAL" not in out


def test_conflicting_sources_are_distinguishable_by_tag():
    """The answer prompt resolves conflicts using these tags."""
    out = _format_hits(
        [
            hit("保證金率≤100%", type="tutorial"),
            hit(
                "保證金率≤0%",
                authority="official",
                updated_at="2025-03-13T00:00:00Z",
                type="help_center",
            ),
        ]
    )
    lines = [b for b in out.split("\n\n") if b.strip()]
    assert "[internal note]" in lines[0] and "≤100%" in lines[0]
    assert "[OFFICIAL, updated 2025-03-13]" in lines[1] and "≤0%" in lines[1]


def test_wiki_sources_keep_wikilink_citation():
    out = _format_hits([hit(source="wiki/entities/platform-superex.md", type="wiki")])
    assert "[[platform-superex]]" in out


def test_long_chunks_are_truncated():
    out = _format_hits([hit("x" * 5000)])
    assert "…" in out
    assert len(out) < 1500


def test_numbering_is_sequential():
    out = _format_hits([hit("a"), hit("b"), hit("c")])
    assert out.startswith("[1] ")
    assert "[2] " in out and "[3] " in out
