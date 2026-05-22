"""Build / refresh the chroma + BM25 hybrid index."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Iterable

from config import (
    BM25_PATH,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DATA_DIR,
    EMBED_MODEL_NAME,
    INDEX_META_PATH,
    KB_SOURCES,
)

from .document import Document
from .loader import load_all
from .tokenize import tokenize

logger = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s (first run downloads ~2.3GB)", EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_chroma_collection(reset: bool = False):
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:  # noqa: BLE001
            pass
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _scan_source_hashes(source_keys: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in source_keys:
        root = KB_SOURCES.get(key)
        if root is None or not root.exists():
            continue
        if root.is_file():
            out[str(root)] = _sha1_of(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".docx"}:
                continue
            out[str(p)] = _sha1_of(p)
    return out


def build_index(sources: list[str] | None = None, *, force: bool = True) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selected = sources or list(KB_SOURCES.keys())

    logger.info("Loading documents from sources: %s", selected)
    docs = load_all(selected)
    if not docs:
        raise RuntimeError("No documents loaded; aborting index build")

    deduped: dict[str, Document] = {}
    for d in docs:
        deduped.setdefault(d.doc_id, d)
    docs = list(deduped.values())
    logger.info("Embedding %d unique chunks", len(docs))

    model = _get_embed_model()
    texts = [d.text for d in docs]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    collection = _get_chroma_collection(reset=force)
    chunk = 256
    for i in range(0, len(docs), chunk):
        batch = docs[i : i + chunk]
        collection.upsert(
            ids=[d.doc_id for d in batch],
            embeddings=[embeddings[j].tolist() for j in range(i, i + len(batch))],
            metadatas=[d.metadata() for d in batch],
            documents=[d.text for d in batch],
        )

    logger.info("Fitting BM25 over %d chunks", len(docs))
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(d.text) for d in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_payload = {
        "bm25": bm25,
        "doc_ids": [d.doc_id for d in docs],
    }
    with BM25_PATH.open("wb") as f:
        pickle.dump(bm25_payload, f)

    file_hashes = _scan_source_hashes(selected)
    meta = {
        "built_at": int(time.time()),
        "sources": selected,
        "doc_count": len(docs),
        "files": file_hashes,
    }
    INDEX_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("Index built: %d chunks across %d files", len(docs), len(file_hashes))
    return meta


def needs_rebuild(sources: list[str] | None = None, threshold_pct: float = 0.05) -> bool:
    """Return True if the on-disk index is missing or the source files drifted."""
    if not INDEX_META_PATH.exists() or not BM25_PATH.exists():
        return True
    try:
        meta = json.loads(INDEX_META_PATH.read_text())
    except json.JSONDecodeError:
        return True
    selected = sources or list(KB_SOURCES.keys())
    if set(meta.get("sources", [])) != set(selected):
        return True
    current = _scan_source_hashes(selected)
    old = meta.get("files", {})
    if not old:
        return True
    keys = set(current) | set(old)
    diffs = sum(1 for k in keys if current.get(k) != old.get(k))
    return diffs / max(len(old), 1) >= threshold_pct
