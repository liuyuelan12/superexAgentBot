"""Top-level loader that dispatches each known source to the right parser."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    KB_SOURCES,
    REPO_ROOT,
    WIKI_DIR,
)

from .chunker import (
    chunk_plain_pages,
    relpath,
    split_bitget_faq,
    split_markdown_header_aware,
)
from .csv_loader import load_customer_service_csv
from .document import Document
from .pdf_extract import extract_pdf_pages

logger = logging.getLogger(__name__)

# Heuristic language tags by path component.
_LANG_HINTS: tuple[tuple[str, str], ...] = (
    ("/英文/", "en"),
    ("/中文/", "zh-TW"),
    ("/EN/", "en"),
    ("English", "en"),
    ("english", "en"),
)


def _detect_lang(path: Path) -> str:
    s = str(path)
    for needle, lang in _LANG_HINTS:
        if needle in s:
            return lang
    return "zh-TW"  # repo default


def _load_customer_service(root: Path) -> list[Document]:
    docs: list[Document] = []
    bitget = root / "bitget" / "喂AI文档.md"
    if bitget.exists():
        text = bitget.read_text(encoding="utf-8", errors="ignore")
        docs.extend(split_bitget_faq(text, source=relpath(bitget, REPO_ROOT), lang="en"))
    superex_dir = root / "superex"
    if superex_dir.exists():
        for csv_path in superex_dir.glob("*.csv"):
            docs.extend(
                load_customer_service_csv(csv_path, relpath(csv_path, REPO_ROOT))
            )
    return docs


def _load_official_tutorial(root: Path) -> list[Document]:
    docs: list[Document] = []
    if not root.exists():
        return docs

    for md in root.rglob("*.md"):
        text = md.read_text(encoding="utf-8", errors="ignore")
        docs.extend(
            split_markdown_header_aware(
                text=text,
                source=relpath(md, REPO_ROOT),
                lang=_detect_lang(md),
                doc_type="tutorial",
                max_tokens=CHUNK_SIZE_TOKENS,
                overlap_tokens=CHUNK_OVERLAP_TOKENS,
            )
        )

    for html in root.rglob("*.html"):
        try:
            from html import unescape as _unescape
        except Exception:  # pragma: no cover
            _unescape = lambda x: x  # type: ignore
        raw = html.read_text(encoding="utf-8", errors="ignore")
        stripped = re.sub(r"<[^>]+>", " ", raw)
        text = _unescape(re.sub(r"\s+", " ", stripped)).strip()
        if len(text) < 50:
            continue
        docs.extend(
            split_markdown_header_aware(
                text=text,
                source=relpath(html, REPO_ROOT),
                lang=_detect_lang(html),
                doc_type="tutorial",
                max_tokens=CHUNK_SIZE_TOKENS,
                overlap_tokens=CHUNK_OVERLAP_TOKENS,
            )
        )

    for pdf in root.rglob("*.pdf"):
        pages = extract_pdf_pages(pdf)
        if not pages:
            continue
        docs.extend(
            chunk_plain_pages(
                pages=pages,
                source=relpath(pdf, REPO_ROOT),
                lang=_detect_lang(pdf),
                doc_type="pdf",
                max_tokens=CHUNK_SIZE_TOKENS,
                overlap_tokens=CHUNK_OVERLAP_TOKENS,
            )
        )

    logger.info("Official tutorial -> %d docs", len(docs))
    return docs


_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    return data, text[m.end():]


_WIKI_STUB_MARKERS = ("占位页", "TODO Phase", "Phase 2 时填充", "Phase 1 时填充")


def _load_wiki(root: Path) -> list[Document]:
    docs: list[Document] = []
    if not root.exists():
        return docs
    for md in root.rglob("*.md"):
        if md.name.startswith("lint-report"):
            continue
        text = md.read_text(encoding="utf-8", errors="ignore")
        frontmatter, body = _parse_frontmatter(text)
        if not body.strip():
            continue
        if frontmatter.get("status") == "stub":
            continue
        if any(m in body for m in _WIKI_STUB_MARKERS):
            continue
        extras: dict[str, str | int | float | bool] = {}
        for k in ("type", "status", "audience", "last_verified"):
            v = frontmatter.get(k)
            if isinstance(v, (str, int, float, bool)):
                extras[k] = v
        aliases = frontmatter.get("aliases")
        if isinstance(aliases, list):
            extras["aliases"] = " | ".join(str(a) for a in aliases)
        sub = split_markdown_header_aware(
            text=body,
            source=relpath(md, REPO_ROOT),
            lang="zh-CN",
            doc_type="wiki",
            max_tokens=CHUNK_SIZE_TOKENS,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )
        if extras and sub:
            sub = [
                Document(
                    text=d.text,
                    source=d.source,
                    lang=d.lang,
                    type=d.type,
                    section=d.section,
                    extras={**d.extras, **extras},
                )
                for d in sub
            ]
        docs.extend(sub)
    logger.info("Wiki -> %d docs", len(docs))
    return docs


def load_all(sources: list[str] | None = None) -> list[Document]:
    """Load documents from every configured source (or a filtered subset)."""
    selected = sources or list(KB_SOURCES.keys())
    docs: list[Document] = []
    for key in selected:
        root = KB_SOURCES.get(key)
        if root is None:
            logger.warning("Unknown source key: %s", key)
            continue
        if not root.exists():
            logger.info("Source path missing, skipping: %s", root)
            continue
        if key == "raw_customer_service":
            docs.extend(_load_customer_service(root))
        elif key == "raw_official_tutorial":
            docs.extend(_load_official_tutorial(root))
        elif key == "wiki":
            docs.extend(_load_wiki(root))
    logger.info("Total documents loaded: %d", len(docs))
    return docs
