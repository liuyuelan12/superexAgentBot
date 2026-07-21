"""Unit tests for the markdown → Telegram HTML converter."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.telegram_format import (  # noqa: E402
    md_to_telegram_html,
    split_for_telegram,
    strip_markdown,
)

# Exactly the tags Telegram documents for parse_mode=HTML.
TELEGRAM_TAGS = {"b", "i", "u", "s", "a", "code", "pre", "blockquote"}
_TAG_RE = re.compile(r"</?([a-zA-Z][\w-]*)")


def assert_telegram_safe(html: str) -> None:
    """No unsupported tags, and every tag properly nested and closed."""
    stack: list[str] = []
    for m in re.finditer(r"</?([a-zA-Z][\w-]*)(?:\s[^>]*)?>", html):
        tag = m.group(1).lower()
        assert tag in TELEGRAM_TAGS, f"unsupported tag <{tag}> in {html!r}"
        if m.group(0).startswith("</"):
            assert stack and stack[-1] == tag, f"bad nesting at {m.group(0)} in {html!r}"
            stack.pop()
        else:
            stack.append(tag)
    assert not stack, f"unclosed tags {stack} in {html!r}"


# ── the bug from the screenshot ────────────────────────────────────────────────


def test_bold_headings_from_the_reported_screenshot():
    src = (
        "好的，以下是 SuperEx 的具体费率标准：\n\n"
        "**现货交易费率：**\n"
        "- 挂单 (Maker)：0.2%\n"
        "- 吃单 (Taker)：0.2%\n"
    )
    out = md_to_telegram_html(src)
    assert "**" not in out
    assert "<b>现货交易费率：</b>" in out
    assert "• 挂单 (Maker)：0.2%" in out
    assert_telegram_safe(out)


def test_no_raw_markdown_markers_survive():
    src = "**粗** *斜* __粗2__ _斜2_ ~~删~~ ### 标题\n- a\n1. b\n---\n"
    out = md_to_telegram_html(src)
    assert "**" not in out and "~~" not in out and "__" not in out
    assert not re.search(r"^#{1,6}\s", out, re.MULTILINE)
    assert_telegram_safe(out)


# ── inline formatting ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("**bold**", "<b>bold</b>"),
        ("__bold__", "<b>bold</b>"),
        ("*italic*", "<i>italic</i>"),
        ("_italic_", "<i>italic</i>"),
        ("~~gone~~", "<s>gone</s>"),
        ("`code`", "<code>code</code>"),
        ("[SuperEx](https://superex.com)", '<a href="https://superex.com">SuperEx</a>'),
    ],
)
def test_inline_conversions(src, expected):
    assert md_to_telegram_html(src) == expected


def test_headings_become_bold():
    assert md_to_telegram_html("## 费用") == "<b>费用</b>"
    assert md_to_telegram_html("###### deep") == "<b>deep</b>"


def test_bullets_and_ordered_lists():
    out = md_to_telegram_html("- one\n* two\n+ three\n1. first\n2) second")
    assert out.splitlines() == ["• one", "• two", "• three", "1. first", "2. second"]


def test_horizontal_rule_is_dropped():
    assert md_to_telegram_html("a\n---\nb") == "a\nb"
    assert md_to_telegram_html("a\n***\nb") == "a\nb"


def test_blockquote():
    out = md_to_telegram_html("> line one\n> line two")
    assert out == "<blockquote>line one\nline two</blockquote>"
    assert_telegram_safe(out)


# ── escaping ───────────────────────────────────────────────────────────────────


def test_angle_brackets_and_ampersand_are_escaped():
    out = md_to_telegram_html("5 < 6 & 7 > 3")
    assert out == "5 &lt; 6 &amp; 7 &gt; 3"
    assert_telegram_safe(out)


def test_model_emitted_html_is_neutralised_not_executed():
    out = md_to_telegram_html("<script>alert(1)</script>")
    assert "<script" not in out
    assert "&lt;script&gt;" in out
    assert_telegram_safe(out)


def test_ampersand_inside_link_url_is_escaped():
    out = md_to_telegram_html("[fee](https://superex.com/a?x=1&y=2)")
    assert out == '<a href="https://superex.com/a?x=1&amp;y=2">fee</a>'
    assert_telegram_safe(out)


# ── the underscore landmine ────────────────────────────────────────────────────


def test_underscores_in_bot_handle_are_left_alone():
    """The refusal template contains @SuperEx_Zendesk_Bot — it must survive."""
    src = "暂无相关资料，请私信 @SuperEx_Zendesk_Bot 开工单联系客服 ✉️"
    out = md_to_telegram_html(src)
    assert "@SuperEx_Zendesk_Bot" in out
    assert "<i>" not in out
    assert_telegram_safe(out)


def test_underscores_in_urls_are_left_alone():
    src = "https://support.superex.com/hc/en-001/articles/123-How_to_deposit_funds"
    assert md_to_telegram_html(src) == src


def test_snake_case_identifier_is_not_italicised():
    assert md_to_telegram_html("use max_leverage_ratio here") == "use max_leverage_ratio here"


# ── code ───────────────────────────────────────────────────────────────────────


def test_fenced_code_block_with_language():
    out = md_to_telegram_html("```python\nx = 1\n```")
    assert out == '<pre><code class="language-python">x = 1</code></pre>'
    assert_telegram_safe(out)


def test_markdown_inside_code_stays_literal():
    out = md_to_telegram_html("`**not bold**`")
    assert out == "<code>**not bold**</code>"


def test_code_content_is_escaped():
    out = md_to_telegram_html("`a < b`")
    assert out == "<code>a &lt; b</code>"
    assert_telegram_safe(out)


# ── tables ─────────────────────────────────────────────────────────────────────


def test_table_is_flattened_with_bold_header():
    src = "| VIP | Maker | Taker |\n|---|---|---|\n| VIP6 | 0.06% | 0.08% |"
    out = md_to_telegram_html(src)
    assert "|" not in out
    assert out.splitlines() == ["<b>VIP — Maker — Taker</b>", "VIP6 — 0.06% — 0.08%"]
    assert_telegram_safe(out)


# ── robustness ─────────────────────────────────────────────────────────────────


def test_unbalanced_bold_marker_is_dropped():
    out = md_to_telegram_html("**开头没有收尾")
    assert "*" not in out
    assert out == "开头没有收尾"


def test_unclosed_tag_from_split_is_auto_closed():
    """A part boundary can leave <b> open; the balancer must close it."""
    out = md_to_telegram_html("## 标题\n**粗体被截断")
    assert_telegram_safe(out)


def test_empty_and_whitespace_input():
    assert md_to_telegram_html("") == ""
    assert md_to_telegram_html("   \n\n ") == ""


def test_cjk_and_emoji_mixed():
    out = md_to_telegram_html("**费率** 说明 🚀 *note*")
    assert out == "<b>费率</b> 说明 🚀 <i>note</i>"
    assert_telegram_safe(out)


def test_excess_blank_lines_collapse():
    assert md_to_telegram_html("a\n\n\n\n\nb") == "a\n\nb"


# ── splitting ──────────────────────────────────────────────────────────────────


def test_short_text_is_not_split():
    assert split_for_telegram("hello", 100) == ["hello"]


def test_split_prefers_paragraph_boundaries():
    parts = split_for_telegram("A" * 60 + "\n\n" + "B" * 60, 100)
    assert parts == ["A" * 60, "B" * 60]


def test_every_split_part_converts_to_valid_html():
    src = ("**段落标题**\n- 条目一\n- 条目二\n\n" * 40).strip()
    parts = split_for_telegram(src, 300)
    assert len(parts) > 1
    for part in parts:
        assert len(part) <= 300
        assert_telegram_safe(md_to_telegram_html(part))


def test_split_of_empty_text():
    assert split_for_telegram("   ", 100) == []


# ── plain-text fallback ────────────────────────────────────────────────────────


def test_strip_markdown_removes_all_syntax():
    src = "**粗** `code` [x](https://a.com)\n## 标题\n- item\n---\n| a | b |\n|---|---|"
    out = strip_markdown(src)
    assert "**" not in out and "`" not in out and "#" not in out
    assert "<" not in out
    assert "粗" in out and "code" in out and "标题" in out
    assert "• item" in out


def test_strip_markdown_keeps_link_target_visible():
    assert strip_markdown("[fee page](https://superex.com/fee)") == (
        "fee page https://superex.com/fee"
    )


def test_strip_markdown_preserves_bot_handle():
    src = "请私信 @SuperEx_Zendesk_Bot 开工单"
    assert strip_markdown(src) == src
