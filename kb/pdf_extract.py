"""PDF -> per-page text using pymupdf (fitz)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_pdf_pages(path: Path) -> list[str]:
    """Return list of page-text strings; empty pages are kept as ''."""
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.warning("pymupdf not installed; skipping %s", path)
        return []

    pages: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page in doc:
                text = page.get_text("text") or ""
                pages.append(text.strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract %s: %s", path, exc)
        return []
    non_empty = sum(1 for p in pages if p)
    if non_empty == 0:
        logger.info("PDF %s has no extractable text (likely scanned); skipping", path.name)
    return pages
