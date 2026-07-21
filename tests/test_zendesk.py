"""Unit tests for the Help Center HTML → markdown conversion (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.zendesk import (  # noqa: E402
    ANNOUNCEMENT_CATEGORY_IDS,
    Article,
    html_to_markdown,
    to_lang,
)


def test_headings_become_markdown_headings():
    assert html_to_markdown("<h2>Fee Schedule</h2>") == "## Fee Schedule"


def test_paragraphs_are_separated():
    out = html_to_markdown("<p>First line.</p><p>Second line.</p>")
    assert out == "First line.\n\nSecond line."


def test_unordered_list():
    out = html_to_markdown("<ul><li>Maker</li><li>Taker</li></ul>")
    assert out.splitlines() == ["- Maker", "- Taker"]


def test_ordered_list_is_numbered_sequentially():
    out = html_to_markdown("<ol><li>Open app</li><li>Tap deposit</li><li>Pick chain</li></ol>")
    assert out.splitlines() == ["1. Open app", "2. Tap deposit", "3. Pick chain"]


def test_links_become_markdown_links():
    out = html_to_markdown('<p>See <a href="https://superex.com/fee">fees</a>.</p>')
    assert "[fees](https://superex.com/fee)" in out


def test_bare_link_without_text_keeps_url():
    out = html_to_markdown('<a href="https://superex.com"></a>')
    assert out == "https://superex.com"


def test_emphasis():
    assert html_to_markdown("<p><strong>Bold</strong> and <em>italic</em></p>") == (
        "**Bold** and *italic*"
    )


def test_table_becomes_markdown_table():
    html = (
        "<table><tr><th>VIP</th><th>Maker</th></tr>"
        "<tr><td>VIP 6</td><td>0.0250%</td></tr></table>"
    )
    lines = html_to_markdown(html).splitlines()
    assert lines[0] == "| VIP | Maker |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| VIP 6 | 0.0250% |"


def test_html_entities_are_decoded():
    assert html_to_markdown("<p>Fee &lt; 0.1% &amp; falling</p>") == "Fee < 0.1% & falling"


def test_scripts_and_styles_are_dropped():
    out = html_to_markdown("<p>Keep</p><script>alert(1)</script><style>.a{}</style>")
    assert out == "Keep"
    assert "alert" not in out


def test_line_breaks():
    assert html_to_markdown("Line A<br/>Line B") == "Line A\nLine B"


def test_excess_blank_lines_collapse():
    out = html_to_markdown("<div><p>A</p><div></div><div></div><p>B</p></div>")
    assert "\n\n\n" not in out


def test_empty_input():
    assert html_to_markdown("") == ""


def test_malformed_html_does_not_raise():
    assert "text" in html_to_markdown("<p>text<unclosed<<<")


def test_cjk_content_survives():
    out = html_to_markdown("<h3>提現手續費</h3><p>依網絡動態調整</p>")
    assert "提現手續費" in out and "依網絡動態調整" in out


def test_image_alt_text_is_kept_as_placeholder():
    assert "[图片: 充值页面]" in html_to_markdown('<img alt="充值页面" src="x.png">')


# ── locale mapping ─────────────────────────────────────────────────────────────


def test_locale_maps_onto_repo_lang_vocabulary():
    assert to_lang("en-001") == "en"
    assert to_lang("zh-hk") == "zh-TW"
    assert to_lang("pt-br") == "pt"
    assert to_lang("uk-ua") == "uk"


def test_unmapped_locale_passes_through():
    assert to_lang("ru") == "ru"
    assert to_lang("fa") == "fa"


# ── announcement filter ────────────────────────────────────────────────────────


def test_announcement_categories_are_excluded_by_id():
    """IDs, not names — names differ per locale, IDs are identical."""
    assert 4410470420249 in ANNOUNCEMENT_CATEGORY_IDS  # Announcements
    assert 22895674590233 in ANNOUNCEMENT_CATEGORY_IDS  # User Notice
    assert 4412787796889 not in ANNOUNCEMENT_CATEGORY_IDS  # Beginner's Center


# ── article model ──────────────────────────────────────────────────────────────


def article(**kw):
    base = dict(
        id=1,
        title="How to Deposit",
        body_markdown="body",
        url="https://support.superex.com/hc/en-001/articles/1",
        locale="en-001",
        category="Beginner's Center",
        section="Deposit",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
    )
    base.update(kw)
    return Article(**base)


def test_slug_is_filesystem_safe():
    assert article(title="How to Deposit / Withdraw?").slug() == "how-to-deposit-withdraw"


def test_slug_keeps_cjk():
    assert "充值教程" in article(title="充值教程").slug()


def test_slug_never_empty():
    assert article(title="???").slug() == "article"


def test_article_lang_follows_locale():
    assert article(locale="zh-hk").lang == "zh-TW"
