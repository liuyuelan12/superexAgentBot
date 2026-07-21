"""Loader for the SuperEx multilingual customer-service CSV."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .chunker import is_link_only
from .document import Document

logger = logging.getLogger(__name__)

# Column index -> ISO language code. Column 2 has an empty header but holds 繁體中文.
LANG_COLS: dict[int, str] = {
    2: "zh-TW",
    3: "zh-CN",
    4: "en",
    5: "ru",
    6: "fa",
    7: "uk",
    8: "vi",
    9: "es",
    10: "fr",
}


def load_customer_service_csv(path: Path, source_label: str) -> list[Document]:
    """Each row -> up to 9 language-specific Documents sharing a group_id."""
    docs: list[Document] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return docs
        for row_idx, row in enumerate(reader):
            if len(row) < 11:
                continue
            category = (row[0] or "").strip()
            topic = (row[1] or "").strip()
            if not topic and not category:
                continue
            group_id = f"cs-csv-{row_idx}"
            section = " / ".join(p for p in (category, topic) if p)
            for col_idx, lang in LANG_COLS.items():
                text = (row[col_idx] or "").strip()
                if not text or len(text) < 8:
                    continue
                docs.append(
                    Document(
                        text=text,
                        source=source_label,
                        lang=lang,
                        type="cs-script",
                        section=section,
                        extras={
                            "group_id": group_id,
                            "category": category,
                            "topic": topic,
                            "link_only": is_link_only(text),
                        },
                    )
                )
    logger.info("CSV %s -> %d docs", path.name, len(docs))
    return docs
