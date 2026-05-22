"""Document data model shared by loaders, chunker, and indexer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    lang: str
    type: str
    section: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        key = f"{self.source}|{self.section}|{self.text[:200]}|{self.lang}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": self.source,
            "lang": self.lang,
            "type": self.type,
            "section": self.section,
        }
        for k, v in self.extras.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        return meta

    def basename(self) -> str:
        return self.source.rsplit("/", 1)[-1]
