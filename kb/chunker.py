"""Chunkers for different document formats."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .document import Document

logger = logging.getLogger(__name__)

NUMBERED_QA = re.compile(r"^(\d+)\\?\.\s+", re.MULTILINE)
H2_SPLIT = re.compile(r"\n(?=##\s+)")
CJK_RANGE = re.compile(r"[一-鿿㐀-䶿]")

# ── "see this link" stub detection ─────────────────────────────────────────────
# Some KB sections are just a heading plus a URL ("VIP 等级和手续费折扣 / 请参考以下
# 链接：…"). They match support questions lexically but answer nothing, so they
# crowd real content out of TOP_K. Flagging them lets the retriever demote them.
_URL_RE = re.compile(r"https?://\S+")
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_BUTTON_RE = re.compile(r"【[^】]*】")
_REFERRAL_PHRASES = (
    "请参考以下链接",
    "详情请查看以下链接",
    "请查看以下链接",
    "请参考",
    "详情请查看",
    "详见",
    "立即了解",
    "点击以下链接",
    "如下链接",
    "see the following link",
    "please refer to",
)
# Calibrated against the real 472-chunk corpus by sweeping the threshold:
#   25 chars -> "## 费用 / ### VIP 等级和手续费折扣 / 请参考以下链接：…"   (stub, want to catch)
#   32 chars -> "### 返佣 / 推荐返佣怎么算 / 请参考以下链接：…"              (stub, want to catch)
#   43 chars -> "如何注册帳戶（網頁端）" full tutorial step                  (real, must NOT catch)
# 40 sits in that gap with a small margin. Raising it past 43 starts flagging
# sections that genuinely answer something — re-run the sweep before changing.
_LINK_ONLY_RESIDUE_CHARS = 40


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_html_comments(text: str) -> str:
    """Drop authoring notes so they never reach the model.

    Curated raw pages carry maintenance notes — why a section exists, which
    language is authoritative, what to keep in sync. They are for whoever edits
    the file, not for the answering model: left in, they eat context on every
    retrieval and dilute the chunk's embedding. One such note took up nearly
    half of the cross-margin chunk.
    """
    return _HTML_COMMENT_RE.sub("", text)


def is_link_only(text: str) -> bool:
    """True when a chunk carries a URL but essentially no answer of its own."""
    if not _URL_RE.search(text) and "【" not in text:
        return False
    residue = _HEADING_LINE_RE.sub("", text)
    residue = _URL_RE.sub("", residue)
    residue = _BUTTON_RE.sub("", residue)
    for phrase in _REFERRAL_PHRASES:
        residue = residue.replace(phrase, "")
    residue = re.sub(r"[\s:：，。,.、\-—_|*#>()（）]+", "", residue)
    return len(residue) < _LINK_ONLY_RESIDUE_CHARS


def _estimate_tokens(text: str) -> int:
    """Rough token estimator that works across languages without tiktoken's vocab bias."""
    cjk = len(CJK_RANGE.findall(text))
    rest = len(text) - cjk
    return cjk + max(1, rest // 4)


def split_bitget_faq(text: str, source: str, lang: str = "en") -> list[Document]:
    """Split bitget FAQ markdown where each Q is prefixed by `<n>\\.`."""
    matches = list(NUMBERED_QA.finditer(text))
    if not matches:
        return [Document(text=text.strip(), source=source, lang=lang, type="faq")]

    docs: list[Document] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if not lines:
            continue
        question = lines[0]
        answer = " ".join(lines[1:]) if len(lines) > 1 else ""
        clean = (
            f"Q: {question}\nA: {answer}"
            .replace("\\!", "!")
            .replace("\\'", "'")
            .replace("\\.", ".")
        )
        docs.append(
            Document(
                text=clean,
                source=source,
                lang=lang,
                type="faq",
                section=f"#{m.group(1)} {question[:60]}",
            )
        )
    return docs


def split_markdown_header_aware(
    text: str,
    source: str,
    lang: str,
    doc_type: str = "tutorial",
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Document]:
    """Split by ## headers; long sections get a sliding window."""
    sections = H2_SPLIT.split(strip_html_comments(text))
    docs: list[Document] = []
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 30:
            continue
        header_match = re.match(r"##\s+(.+)", sec)
        section_name = header_match.group(1).strip() if header_match else ""
        if _estimate_tokens(sec) <= int(max_tokens * 1.5):
            docs.append(
                Document(
                    text=sec,
                    source=source,
                    lang=lang,
                    type=doc_type,
                    section=section_name,
                    extras={"link_only": is_link_only(sec)},
                )
            )
        else:
            for piece in _sliding_window(sec, max_tokens, overlap_tokens):
                docs.append(
                    Document(
                        text=piece,
                        source=source,
                        lang=lang,
                        type=doc_type,
                        section=section_name,
                        extras={"link_only": is_link_only(piece)},
                    )
                )
    if not docs and text.strip():
        for piece in _sliding_window(text.strip(), max_tokens, overlap_tokens):
            docs.append(
                Document(
                    text=piece,
                    source=source,
                    lang=lang,
                    type=doc_type,
                    section="",
                )
            )
    return docs


def _sliding_window(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Approximate token-based sliding window that splits on paragraph then sentence boundary."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[。！？!?\.])\s+", para)
            for sent in sentences:
                stok = _estimate_tokens(sent)
                if buf_tokens + stok > max_tokens and buf:
                    chunks.append("\n\n".join(buf))
                    buf, buf_tokens = _carry_overlap(buf, overlap_tokens)
                buf.append(sent)
                buf_tokens += stok
            continue
        if buf_tokens + para_tokens > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = _carry_overlap(buf, overlap_tokens)
        buf.append(para)
        buf_tokens += para_tokens
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _carry_overlap(buf: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Keep the tail of the previous chunk to seed the next one."""
    if not buf or overlap_tokens <= 0:
        return [], 0
    tail = buf[-1]
    return [tail], _estimate_tokens(tail)


def chunk_plain_pages(
    pages: list[str],
    source: str,
    lang: str,
    doc_type: str = "pdf",
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Document]:
    """Treat each page as a logical block and apply sliding window if needed."""
    docs: list[Document] = []
    for page_idx, page in enumerate(pages):
        page = page.strip()
        if len(page) < 30:
            continue
        for piece in _sliding_window(page, max_tokens, overlap_tokens):
            docs.append(
                Document(
                    text=piece,
                    source=source,
                    lang=lang,
                    type=doc_type,
                    section=f"page-{page_idx + 1}",
                )
            )
    return docs


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
