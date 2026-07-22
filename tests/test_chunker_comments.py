"""Tests for authoring-note stripping in the chunker.

Curated raw pages carry maintenance notes for whoever edits them. One such
note took up nearly half of the cross-margin chunk once it was indexed,
eating context on every retrieval and diluting the chunk's embedding.
"""

from __future__ import annotations

from kb.chunker import split_markdown_header_aware, strip_html_comments

class TestHtmlCommentStripping:
    """Authoring notes in curated raw pages must not reach the model."""

    def test_comment_is_removed(self):
        text = "## Title\n\n<!-- note for maintainers -->\n\nReal content."
        out = strip_html_comments(text)
        assert "maintainers" not in out
        assert "Real content." in out

    def test_multiline_comment_is_removed(self):
        text = "## T\n\n<!--\n  line one\n  line two\n-->\n\nBody."
        out = strip_html_comments(text)
        assert "line one" not in out
        assert "Body." in out

    def test_content_without_comments_is_unchanged(self):
        text = "## T\n\nJust content, 5 > 3 and a < b."
        assert strip_html_comments(text) == text

    def test_chunker_drops_comments_from_indexed_text(self):
        text = (
            "## Cross margin\n\n"
            "<!-- keep both languages in sync; Chinese is source of truth -->\n\n"
            "Cross margin shares one pool across every open position, "
            "so one loss can raise the liquidation price of the others.\n"
        )
        docs = split_markdown_header_aware(text, source="t.md", lang="en")
        assert docs, "expected at least one chunk"
        assert all("source of truth" not in d.text for d in docs)
        assert any("shares one pool" in d.text for d in docs)
