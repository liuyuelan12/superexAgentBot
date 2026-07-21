"""Markdown → Telegram-safe HTML.

Telegram's HTML parse mode supports ONLY these tags:
    b/strong, i/em, u/ins, s/strike/del, a, code, pre, blockquote,
    span class="tg-spoiler", tg-emoji, tg-time

There is no support for headings, bullet/numbered lists, tables or horizontal
rules in either HTML or MarkdownV2 mode — those have to be flattened into plain
text. Only ``<``, ``>`` and ``&`` need escaping.

Reference: https://core.telegram.org/bots/api#formatting-options

Everything here is a pure function so it can be unit-tested without a bot token.
"""

from __future__ import annotations

import re

__all__ = ["md_to_telegram_html", "split_for_telegram", "strip_markdown"]

# Tags we are willing to emit. Anything else the model invents gets escaped.
_ALLOWED_TAGS = ("b", "i", "u", "s", "a", "code", "pre", "blockquote")

_BULLET = "• "
_COL_SEP = " — "

# ── code extraction ────────────────────────────────────────────────────────────
_FENCE_RE = re.compile(r"```[ \t]*(\w*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_PLACEHOLDER = "\x00C{}\x00"
_PLACEHOLDER_RE = re.compile(r"\x00C(\d+)\x00")

# ── line-level structure ───────────────────────────────────────────────────────
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(?=\S)")
_ORDERED_RE = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(?=\S)")
_QUOTE_RE = re.compile(r"^\s{0,3}&gt;\s?")

# ── inline formatting ──────────────────────────────────────────────────────────
_LINK_RE = re.compile(r"\[([^\]\n]*?)\]\(\s*(\S+?)\s*\)")
_BOLD_RE = re.compile(r"(?<!\\)\*\*(?!\s)(.+?)(?<!\s)\*\*", re.DOTALL)
_BOLD_US_RE = re.compile(r"(?<![\w\\])__(?!\s)(.+?)(?<!\s)__(?!\w)", re.DOTALL)
_STRIKE_RE = re.compile(r"(?<!\\)~~(?!\s)(.+?)(?<!\s)~~", re.DOTALL)
# Single-marker emphasis is deliberately conservative: the opening marker must
# not follow a word character and the closing marker must not precede one, so
# identifiers like @SuperEx_Zendesk_Bot and URLs with underscores survive intact.
_ITALIC_STAR_RE = re.compile(r"(?<![\w*\\])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_ITALIC_US_RE = re.compile(r"(?<![\w_\\])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")

_LEFTOVER_MARKER_RE = re.compile(r"(?<!\\)(\*{2,}|_{2,}|~{2,})")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

_TAG_RE = re.compile(r"</?(%s)(?:\s[^>]*)?>" % "|".join(_ALLOWED_TAGS))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(url: str) -> str:
    """Escape a URL taken from *unescaped* source (used for fenced-code languages)."""
    return _escape(url).replace('"', "&quot;")


def _quote_attr(url: str) -> str:
    """Escape a URL that has already been through :func:`_escape`.

    Running the full escape again would turn ``&amp;`` into ``&amp;amp;``.
    """
    return url.replace('"', "&quot;")


def _extract_code(text: str) -> tuple[str, list[str]]:
    """Pull code out before any other rewriting so its contents stay literal."""
    blocks: list[str] = []

    def _take_fence(m: re.Match[str]) -> str:
        lang, body = m.group(1), m.group(2)
        attr = f' class="language-{_escape_attr(lang)}"' if lang else ""
        blocks.append(f"<pre><code{attr}>{_escape(body.strip(chr(10)))}</code></pre>")
        return _PLACEHOLDER.format(len(blocks) - 1)

    def _take_inline(m: re.Match[str]) -> str:
        blocks.append(f"<code>{_escape(m.group(1))}</code>")
        return _PLACEHOLDER.format(len(blocks) - 1)

    text = _FENCE_RE.sub(_take_fence, text)
    text = _INLINE_CODE_RE.sub(_take_inline, text)
    return text, blocks


def _flatten_table(rows: list[str]) -> list[str]:
    """Telegram has no tables — render each row as a single line, header in bold."""
    out: list[str] = []
    header_done = False
    for row in rows:
        if _TABLE_SEP_RE.match(row):
            continue
        m = _TABLE_ROW_RE.match(row)
        inner = m.group(1) if m else row.strip().strip("|")
        cells = [c.strip() for c in inner.split("|") if c.strip()]
        if not cells:
            continue
        line = _COL_SEP.join(cells)
        if not header_done:
            out.append(f"<b>{line}</b>")
            header_done = True
        else:
            out.append(line)
    return out


def _convert_lines(text: str) -> str:
    """Flatten headings, tables, lists, rules and quotes into Telegram-legal text."""
    lines = text.split("\n")
    out: list[str] = []
    table: list[str] = []
    quote: list[str] = []

    def _flush_table() -> None:
        if table:
            out.extend(_flatten_table(table))
            table.clear()

    def _flush_quote() -> None:
        if quote:
            out.append("<blockquote>" + "\n".join(quote) + "</blockquote>")
            quote.clear()

    for line in lines:
        if _TABLE_ROW_RE.match(line) or (table and _TABLE_SEP_RE.match(line)):
            _flush_quote()
            table.append(line)
            continue
        _flush_table()

        if _QUOTE_RE.match(line):
            quote.append(_QUOTE_RE.sub("", line).strip())
            continue
        _flush_quote()

        if _HR_RE.match(line):
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            out.append(f"<b>{heading.group(1)}</b>")
            continue
        line = _BULLET_RE.sub(lambda m: m.group(1) + _BULLET, line)
        line = _ORDERED_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}. ", line)
        out.append(line)

    _flush_table()
    _flush_quote()
    return "\n".join(out)


def _convert_inline(text: str) -> str:
    text = _LINK_RE.sub(
        lambda m: f'<a href="{_quote_attr(m.group(2))}">{m.group(1) or m.group(2)}</a>',
        text,
    )
    text = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BOLD_US_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", text)
    text = _ITALIC_STAR_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    text = _ITALIC_US_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    # Whatever emphasis markers survived were unbalanced — drop them rather than
    # letting the user see raw syntax.
    return _LEFTOVER_MARKER_RE.sub("", text)


def _balance_tags(html: str) -> str:
    """Close tags left open and drop stray closers, so Telegram never rejects us."""
    stack: list[str] = []
    out: list[str] = []
    pos = 0
    for m in _TAG_RE.finditer(html):
        out.append(html[pos : m.start()])
        pos = m.end()
        tag = m.group(1)
        if m.group(0).startswith("</"):
            if tag not in stack:
                continue
            while stack[-1] != tag:
                out.append(f"</{stack.pop()}>")
            stack.pop()
            out.append(f"</{tag}>")
        else:
            stack.append(tag)
            out.append(m.group(0))
    out.append(html[pos:])
    while stack:
        out.append(f"</{stack.pop()}>")
    return "".join(out)


def md_to_telegram_html(text: str) -> str:
    """Convert loosely-formatted markdown into the HTML subset Telegram accepts."""
    if not text or not text.strip():
        return ""
    body, code_blocks = _extract_code(text)
    body = _escape(body)
    body = _convert_lines(body)
    body = _convert_inline(body)
    body = _PLACEHOLDER_RE.sub(lambda m: code_blocks[int(m.group(1))], body)
    body = _balance_tags(body)
    return _MULTI_BLANK_RE.sub("\n\n", body).strip()


def strip_markdown(text: str) -> str:
    """Plain-text fallback used when Telegram rejects the generated HTML."""
    body = _FENCE_RE.sub(lambda m: m.group(2).strip("\n"), text)
    body = _INLINE_CODE_RE.sub(lambda m: m.group(1), body)
    lines: list[str] = []
    for line in body.split("\n"):
        if _HR_RE.match(line) or _TABLE_SEP_RE.match(line):
            continue
        row = _TABLE_ROW_RE.match(line)
        if row:
            cells = [c.strip() for c in row.group(1).split("|") if c.strip()]
            line = _COL_SEP.join(cells)
        heading = _HEADING_RE.match(line)
        if heading:
            line = heading.group(1)
        line = _BULLET_RE.sub(lambda m: m.group(1) + _BULLET, line)
        lines.append(line)
    body = "\n".join(lines)
    body = _LINK_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}".strip(), body)
    body = _LEFTOVER_MARKER_RE.sub("", body)
    body = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", body)
    return _MULTI_BLANK_RE.sub("\n\n", body).strip()


def split_for_telegram(text: str, limit: int) -> list[str]:
    """Split on paragraph/line/word boundaries, before any HTML is generated.

    Splitting the markdown source rather than the rendered HTML guarantees a tag
    is never cut in half; each part is converted (and tag-balanced) on its own.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break
        cut = remaining[:limit]
        split_at = cut.rfind("\n\n")
        if split_at < limit // 2:
            split_at = cut.rfind("\n")
        if split_at < limit // 2:
            split_at = cut.rfind(" ")
        if split_at > 0:
            cut = remaining[:split_at]
        parts.append(cut.rstrip())
        remaining = remaining[len(cut) :].lstrip()
    return [p for p in parts if p]
