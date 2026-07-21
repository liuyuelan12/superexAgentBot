"""Hybrid vector + BM25 retriever."""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    BM25_PATH,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DOC_TYPE_WEIGHTS,
    EMBED_MODEL_NAME,
    HYBRID_VECTOR_WEIGHT,
    LINK_ONLY_PENALTY,
    TOP_K,
)

from .tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    doc_id: str
    text: str
    source: str
    section: str
    lang: str
    score: float
    vec_sim: float
    bm25_norm: float
    metadata: dict[str, Any]

    def basename(self) -> str:
        return self.source.rsplit("/", 1)[-1]


_WS_RE = re.compile(r"\s+")


def _dedup_key(text: str) -> str:
    """Collapse whitespace/case so translations that are byte-identical fold together."""
    return _WS_RE.sub("", text).lower()


def score_pool(
    pool: dict[str, dict[str, Any]],
    *,
    lang_boost: str | None,
    top_k: int,
) -> list[Hit]:
    """Blend vector + BM25 scores, apply quality weights, dedup, and truncate.

    Pure function over an already-fetched candidate pool so the ranking rules can
    be unit-tested without chroma or the embedding model.

    Penalties multiply ``score`` only — never ``vec_sim`` — because SIM_THRESHOLD
    gates the refusal path on raw vector similarity.
    """
    alpha = HYBRID_VECTOR_WEIGHT
    boost = (lang_boost or "").split("-")[0].lower()
    hits: list[Hit] = []
    for v in pool.values():
        if v["text"] is None:
            continue
        meta = v["meta"] or {}
        score = alpha * v["vec_sim"] + (1 - alpha) * v["bm25_norm"]
        lang = str(meta.get("lang", ""))
        if boost and lang and lang.split("-")[0].lower() == boost:
            score *= 1.05
        if meta.get("link_only"):
            score *= LINK_ONLY_PENALTY
        score *= DOC_TYPE_WEIGHTS.get(str(meta.get("type", "")), 1.0)
        hits.append(
            Hit(
                doc_id=v["doc_id"],
                text=v["text"],
                source=str(meta.get("source", "")),
                section=str(meta.get("section", "")),
                lang=lang,
                score=score,
                vec_sim=v["vec_sim"],
                bm25_norm=v["bm25_norm"],
                metadata=meta,
            )
        )

    hits.sort(key=lambda h: -h.score)
    # The CSV loader emits one Document per language column, so zh-TW and zh-CN
    # rows with identical text become distinct doc_ids and used to occupy two
    # TOP_K slots each. Sorting first means the survivor is the best-scoring one
    # (and the lang_boost multiplier already favours the user's language).
    deduped: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        key = _dedup_key(hit.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped[:top_k]


class Retriever:
    def __init__(self) -> None:
        self._collection = None
        self._embed_model = None
        self._bm25 = None
        self._bm25_doc_ids: list[str] = []
        self._bm25_pos_by_id: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        if not Path(CHROMA_DIR).exists() or not BM25_PATH.exists():
            raise RuntimeError(
                "Index not built. Run: python -m scripts.rebuild_index --force"
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(CHROMA_COLLECTION)
        self._embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        with BM25_PATH.open("rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_doc_ids = payload["doc_ids"]
        self._bm25_pos_by_id = {d: i for i, d in enumerate(self._bm25_doc_ids)}
        logger.info(
            "Retriever ready: chroma=%d, bm25=%d",
            self._collection.count(),
            len(self._bm25_doc_ids),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = TOP_K,
        lang_boost: str | None = None,
    ) -> list[Hit]:
        if not query.strip():
            return []
        pool: dict[str, dict[str, Any]] = {}
        widen = max(top_k * 3, 10)

        # vector search
        q_emb = self._embed_model.encode([query], normalize_embeddings=True)
        vec = self._collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=widen,
        )
        v_ids = vec.get("ids", [[]])[0]
        v_docs = vec.get("documents", [[]])[0]
        v_metas = vec.get("metadatas", [[]])[0]
        v_dists = vec.get("distances", [[]])[0]
        for did, doc, meta, dist in zip(v_ids, v_docs, v_metas, v_dists):
            pool[did] = {
                "doc_id": did,
                "text": doc,
                "meta": meta or {},
                "vec_sim": max(0.0, 1.0 - float(dist)),
                "bm25_norm": 0.0,
            }

        # bm25 search
        if self._bm25 is not None:
            q_tokens = tokenize(query)
            if q_tokens:
                scores = self._bm25.get_scores(q_tokens)
                bm25_top = sorted(
                    enumerate(scores), key=lambda x: x[1], reverse=True
                )[:widen]
                max_score = bm25_top[0][1] if bm25_top else 0.0
                missing_ids: list[str] = []
                for pos, score in bm25_top:
                    if score <= 0 or max_score <= 0:
                        continue
                    did = self._bm25_doc_ids[pos]
                    norm = score / max_score
                    if did in pool:
                        pool[did]["bm25_norm"] = norm
                    else:
                        missing_ids.append(did)
                        pool[did] = {
                            "doc_id": did,
                            "text": None,
                            "meta": None,
                            "vec_sim": 0.0,
                            "bm25_norm": norm,
                        }
                if missing_ids:
                    extra = self._collection.get(ids=missing_ids)
                    for did, doc, meta in zip(
                        extra.get("ids", []),
                        extra.get("documents", []),
                        extra.get("metadatas", []),
                    ):
                        if did in pool:
                            pool[did]["text"] = doc
                            pool[did]["meta"] = meta or {}

        return score_pool(pool, lang_boost=lang_boost, top_k=top_k)
