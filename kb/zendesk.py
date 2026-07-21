"""SuperEx Help Center (Zendesk) client.

support.superex.com exposes a public Help Center API, which is far more reliable
than scraping the JS-rendered www.superex.com SPA. Article bodies come back as
HTML and are converted to markdown here.

Scale note (measured 2026-07-22): the Help Center holds ~13k articles across 11
locales, but **95% of them are announcements** (token listings, delistings,
maintenance, campaigns). Only ~1,279 articles across all locales are rules and
tutorials. ANNOUNCEMENT_CATEGORY_IDS filters those out — the category IDs are
identical across every locale, so filtering by ID (not name) is language-safe.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Any, Iterator

logger = logging.getLogger(__name__)

HELP_CENTER_HOST = "https://support.superex.com"
API_BASE = f"{HELP_CENTER_HOST}/api/v2/help_center"

# Verified identical across all 11 locales, so filtering by id is language-safe.
ANNOUNCEMENT_CATEGORY_IDS: frozenset[int] = frozenset(
    {
        4410470420249,  # Announcements — listings, delistings, maintenance, events
        22895674590233,  # User Notice — protocol lists
    }
)

# Help Center locale -> the lang codes already used across this repo's KB.
LOCALE_TO_LANG: dict[str, str] = {
    "en-001": "en",
    "zh-hk": "zh-TW",
    "pt-br": "pt",
    "uk-ua": "uk",
}

_REQUEST_TIMEOUT = 40
_RETRY_SLEEPS = (1, 3, 8)


def to_lang(locale: str) -> str:
    """Map a Help Center locale onto the repo's existing lang vocabulary."""
    return LOCALE_TO_LANG.get(locale, locale)


# ── HTML → markdown ────────────────────────────────────────────────────────────

_BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "tr"}
_SKIP_TAGS = {"script", "style", "head", "meta", "link"}


class _MarkdownExtractor(HTMLParser):
    """Convert Zendesk article HTML into markdown that the chunker can split.

    Deliberately narrow: headings, lists, links, emphasis, code and tables are
    the only structures Zendesk articles actually use.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip_depth = 0
        self._list_stack: list[str] = []
        self._ol_counters: list[int] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._in_cell = False
        self._row: list[str] = []
        self._table_rows: list[list[str]] = []
        self._in_table = False

    # -- helpers
    def _emit(self, text: str) -> None:
        if self._href is not None:
            self._link_text.append(text)
        elif self._in_cell:
            self._row.append(text)
        else:
            self._out.append(text)

    def _newline(self, count: int = 1) -> None:
        if self._href is not None or self._in_cell:
            return
        self._out.append("\n" * count)

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        attrd = dict(attrs)
        if tag == "br":
            self._newline()
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._newline(2)
            self._out.append("#" * int(tag[1]) + " ")
        elif tag in _BLOCK_TAGS:
            self._newline(2)
        elif tag in ("ul", "ol"):
            self._list_stack.append(tag)
            if tag == "ol":
                self._ol_counters.append(0)
            self._newline(2)
        elif tag == "li":
            self._newline()
            indent = "  " * max(0, len(self._list_stack) - 1)
            if self._list_stack and self._list_stack[-1] == "ol":
                self._ol_counters[-1] += 1
                self._out.append(f"{indent}{self._ol_counters[-1]}. ")
            else:
                self._out.append(f"{indent}- ")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "code":
            self._emit("`")
        elif tag == "a":
            href = (attrd.get("href") or "").strip()
            self._href = href or None
            self._link_text = []
            if not href:
                self._href = None
        elif tag == "table":
            self._in_table = True
            self._table_rows = []
            self._newline(2)
        elif tag == "tr" and self._in_table:
            self._row = []
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._row = self._row or []
        elif tag == "img":
            alt = (attrd.get("alt") or "").strip()
            if alt:
                self._emit(f"[图片: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._newline(2)
        elif tag in _BLOCK_TAGS and tag != "tr":
            self._newline(2)
        elif tag in ("ul", "ol"):
            if self._list_stack:
                popped = self._list_stack.pop()
                if popped == "ol" and self._ol_counters:
                    self._ol_counters.pop()
            self._newline(2)
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "code":
            self._emit("`")
        elif tag == "a" and self._href is not None:
            text = "".join(self._link_text).strip()
            href, self._href, self._link_text = self._href, None, []
            self._emit(f"[{text}]({href})" if text else href)
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = False
        elif tag == "tr" and self._in_table:
            cells = [c.strip() for c in ("".join(self._row) or "").split("\x00") if c.strip()]
            joined = [c.strip() for c in self._row if c.strip()]
            self._table_rows.append(joined or cells)
            self._row = []
        elif tag == "table":
            self._flush_table()
            self._in_table = False

    def _flush_table(self) -> None:
        rows = [r for r in self._table_rows if r]
        if not rows:
            return
        width = max(len(r) for r in rows)
        for i, row in enumerate(rows):
            padded = row + [""] * (width - len(row))
            self._out.append("| " + " | ".join(padded) + " |\n")
            if i == 0:
                self._out.append("|" + "---|" * width + "\n")
        self._out.append("\n")
        self._table_rows = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        if self._in_cell:
            self._row.append(data.strip())
            return
        text = re.sub(r"[ \t]+", " ", data)
        if not text.strip():
            if self._out and not self._out[-1].endswith((" ", "\n")):
                self._emit(" ")
            return
        self._emit(text)

    def result(self) -> str:
        text = "".join(self._out)
        text = unescape(text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


def html_to_markdown(html: str) -> str:
    """Best-effort HTML → markdown for Zendesk article bodies."""
    if not html:
        return ""
    parser = _MarkdownExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not abort a crawl
        logger.warning("HTML parse failed; falling back to tag stripping")
        return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()
    return parser.result()


# ── API models ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Article:
    id: int
    title: str
    body_markdown: str
    url: str
    locale: str
    category: str
    section: str
    created_at: str
    updated_at: str
    edited_at: str = ""
    labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def lang(self) -> str:
        return to_lang(self.locale)

    def slug(self) -> str:
        base = re.sub(r"[^\w一-鿿-]+", "-", self.title).strip("-").lower()
        return (base[:60] or "article").rstrip("-")


def _get(url: str) -> dict[str, Any]:
    """GET with retries; Zendesk rate-limits with 429 + Retry-After."""
    last: Exception | None = None
    for attempt, sleep_for in enumerate((*_RETRY_SLEEPS, None)):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "SuperExAgentBot/kb-crawler"}
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After", "10"))
                logger.warning("Rate limited; sleeping %ss", wait)
                time.sleep(wait)
                continue
            if exc.code >= 500 and sleep_for is not None:
                time.sleep(sleep_for)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if sleep_for is None:
                break
            logger.warning("Request failed (%s); retry %d", exc, attempt + 1)
            time.sleep(sleep_for)
    raise RuntimeError(f"GET failed after retries: {url}") from last


def _paginate(url: str, key: str) -> Iterator[dict[str, Any]]:
    page: str | None = url
    while page:
        data = _get(page)
        yield from data.get(key, [])
        page = data.get("next_page")


def fetch_locales() -> list[str]:
    return list(_get(f"{API_BASE}/locales.json").get("locales", []))


def fetch_taxonomy(locale: str) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    """Return {category_id: name} and {section_id: {...}} for one locale."""
    categories = {
        int(c["id"]): str(c["name"]).strip()
        for c in _paginate(f"{API_BASE}/{locale}/categories.json?per_page=100", "categories")
    }
    sections = {
        int(s["id"]): {
            "name": str(s["name"]).strip(),
            "category_id": int(s["category_id"]),
        }
        for s in _paginate(f"{API_BASE}/{locale}/sections.json?per_page=100", "sections")
    }
    return categories, sections


def fetch_rule_articles(locale: str) -> list[Article]:
    """Every non-announcement article for a locale, as markdown."""
    categories, sections = fetch_taxonomy(locale)
    wanted = {
        sid
        for sid, s in sections.items()
        if s["category_id"] not in ANNOUNCEMENT_CATEGORY_IDS
    }
    if not wanted:
        logger.warning("No rule sections found for locale %s", locale)
        return []

    out: list[Article] = []
    for raw in _paginate(f"{API_BASE}/{locale}/articles.json?per_page=100", "articles"):
        sid = int(raw.get("section_id") or 0)
        if sid not in wanted:
            continue
        if raw.get("draft"):
            continue
        section = sections[sid]
        body = html_to_markdown(raw.get("body") or "")
        if not body.strip():
            continue
        out.append(
            Article(
                id=int(raw["id"]),
                title=str(raw.get("title") or "").strip(),
                body_markdown=body,
                url=str(raw.get("html_url") or ""),
                locale=locale,
                category=categories.get(section["category_id"], "Uncategorized"),
                section=section["name"],
                created_at=str(raw.get("created_at") or ""),
                updated_at=str(raw.get("updated_at") or ""),
                edited_at=str(raw.get("edited_at") or ""),
                labels=tuple(raw.get("label_names") or ()),
            )
        )
    logger.info("locale=%s -> %d rule articles", locale, len(out))
    return out
