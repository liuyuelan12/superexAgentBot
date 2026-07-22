"""Hybrid vector + BM25 retriever."""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    BM25_PATH,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DOC_TYPE_WEIGHTS,
    EMBED_MODEL_NAME,
    HELP_CENTER_LANGS,
    HYBRID_VECTOR_WEIGHT,
    LANG_MISMATCH_PENALTY,
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
        doc_type = str(meta.get("type", ""))
        score = alpha * v["vec_sim"] + (1 - alpha) * v["bm25_norm"]
        lang = str(meta.get("lang", ""))
        primary = lang.split("-")[0].lower()
        if boost and lang and primary == boost:
            score *= 1.05
        elif (
            doc_type == "help_center"
            and boost in HELP_CENTER_LANGS
            and primary
            and primary != boost
        ):
            # A same-language version of this article is known to exist.
            score *= LANG_MISMATCH_PENALTY
        if meta.get("link_only"):
            score *= LINK_ONLY_PENALTY
        score *= DOC_TYPE_WEIGHTS.get(doc_type, 1.0)
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
                max_score = float(max(scores)) if len(scores) else 0.0
                if max_score > 0:
                    # Give every pooled document its *true* score in both channels.
                    #
                    # Previously a document that only one channel surfaced kept 0.0 in
                    # the other, which reads as "irrelevant" when it actually means
                    # "outside that channel's top-N" — a very different claim. The
                    # blend then buried it: the Chinese cross-margin rules rank 13th
                    # in BM25 for an English question but 90th by vector, so they
                    # scored 0.7*0.0 + 0.3*0.551 = 0.165 and never reached the model.
                    #
                    # Both true values are cheap: get_scores() already covers the whole
                    # corpus, and the stored embeddings are normalised, so cosine
                    # similarity is a dot product against the query vector.
                    for did, entry in pool.items():
                        pos = self._bm25_pos_by_id.get(did)
                        if pos is not None:
                            entry["bm25_norm"] = max(0.0, float(scores[pos]) / max_score)

                    bm25_top = sorted(
                        enumerate(scores), key=lambda x: x[1], reverse=True
                    )[:widen]
                    missing_ids: list[str] = []
                    for pos, score in bm25_top:
                        if score <= 0:
                            continue
                        did = self._bm25_doc_ids[pos]
                        if did in pool:
                            continue
                        missing_ids.append(did)
                        pool[did] = {
                            "doc_id": did,
                            "text": None,
                            "meta": None,
                            "vec_sim": 0.0,
                            "bm25_norm": max(0.0, float(score) / max_score),
                        }
                    if missing_ids:
                        self._fill_from_store(pool, missing_ids, q_emb[0])

        return score_pool(pool, lang_boost=lang_boost, top_k=top_k)

    def _fill_from_store(
        self,
        pool: dict[str, dict[str, Any]],
        doc_ids: list[str],
        query_vector: Any,
    ) -> None:
        """Load text, metadata and true vector similarity for BM25-only hits.

        Without the embeddings these documents would keep ``vec_sim = 0.0``, which
        both sinks their blended score and hides them from the SIM_THRESHOLD gate
        that decides whether the bot answers at all.
        """
        try:
            extra = self._collection.get(
                ids=doc_ids, include=["documents", "metadatas", "embeddings"]
            )
        except Exception:  # noqa: BLE001 - retrieval must degrade, not fail
            logger.warning("Could not load BM25-only hits from the store", exc_info=True)
            return

        embeddings = extra.get("embeddings")
        for idx, did in enumerate(extra.get("ids", [])):
            entry = pool.get(did)
            if entry is None:
                continue
            documents = extra.get("documents") or []
            metadatas = extra.get("metadatas") or []
            if idx < len(documents):
                entry["text"] = documents[idx]
            if idx < len(metadatas):
                entry["meta"] = metadatas[idx] or {}
            if embeddings is None or idx >= len(embeddings):
                continue
            # Both sides are L2-normalised at index time, and the collection uses
            # cosine space, so this matches the 1 - distance the vector path yields.
            similarity = float(np.dot(query_vector, embeddings[idx]))
            entry["vec_sim"] = max(0.0, min(1.0, similarity))
