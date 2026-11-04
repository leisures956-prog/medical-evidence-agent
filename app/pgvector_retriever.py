"""Persistent Embedding, vector retrieval and hybrid retrieval.

The existing ``vector`` and ``hybrid`` modes remain in-memory and unchanged.
This module provides new ``vector_pg`` and ``hybrid_pg`` modes backed by the
PostgreSQL + pgvector adapter.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any, Iterable

from app.medical_models import MedicalDocument, RetrievalHit
from app.pgvector_store import (
    PgVectorError,
    PgVectorStore,
    VectorRecord,
    content_sha256,
)


class PersistentEmbeddingError(PgVectorError):
    """Raised when the local embedding model cannot be loaded or used."""


def split_text(text: str, *, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    """Split text deterministically while retaining a small context overlap."""

    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0。")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars。")
    value = (text or "").strip()
    if not value:
        return []
    chunks: list[str] = []
    start = 0
    step = max_chars - overlap
    while start < len(value):
        end = min(len(value), start + max_chars)
        chunks.append(value[start:end])
        if end == len(value):
            break
        start += step
    return chunks


def _rows(value: Any) -> list[list[float]]:
    """Convert numpy/list model output to plain Python rows."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise PersistentEmbeddingError("Embedding 模型返回了无法识别的结果。")
    if not value:
        return []
    first = value[0]
    if not isinstance(first, (list, tuple)):
        value = [value]
    rows: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise PersistentEmbeddingError("Embedding 输出不是二维向量数组。")
        rows.append([float(item) for item in row])
    return rows


class SentenceTransformerEmbedder:
    """Lazy wrapper so importing the API does not load model weights."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise PersistentEmbeddingError(
                "缺少 sentence-transformers，请安装 requirements-rag.txt。"
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise PersistentEmbeddingError(
                f"Embedding 模型加载失败：{self.model_name}。"
            ) from exc
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            encoded = self._load().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except PersistentEmbeddingError:
            raise
        except Exception as exc:
            raise PersistentEmbeddingError("Embedding 编码失败。") from exc
        return _rows(encoded)


class PersistentVectorRetriever:
    """Build and query a versioned persistent vector index."""

    def __init__(
        self,
        store: PgVectorStore,
        *,
        embedder: Any | None = None,
        model_name: str | None = None,
        corpus_version: str | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
    ):
        self.store = store
        self.embedder = embedder or SentenceTransformerEmbedder(model_name)
        self.model_name = model_name or getattr(
            self.embedder, "model_name", os.getenv("EMBEDDING_MODEL", "local-embedding")
        )
        self.corpus_version = corpus_version or os.getenv(
            "VECTOR_CORPUS_VERSION", "pubmed-snapshot-v1"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_env(
        cls,
        *,
        database_url: str | None = None,
        model_name: str | None = None,
        corpus_version: str | None = None,
    ) -> "PersistentVectorRetriever":
        store = PgVectorStore(database_url)
        return cls(
            store,
            model_name=model_name,
            corpus_version=corpus_version,
        )

    def _chunk_records(self, documents: Iterable[MedicalDocument]) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        for document in documents:
            text = document.retrieval_text()
            chunks = split_text(
                text,
                max_chars=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for index, chunk in enumerate(chunks):
                chunk_id = f"{self.corpus_version}:{document.id}:{index}"
                records.append(
                    VectorRecord(
                        document=document,
                        chunk_id=chunk_id,
                        chunk_index=index,
                        chunk_text=chunk,
                        embedding=(),
                        content_sha256=content_sha256(chunk),
                    )
                )
        return records

    def index_documents(
        self,
        documents: Iterable[MedicalDocument],
        *,
        corpus_version: str | None = None,
    ) -> int:
        if corpus_version:
            self.corpus_version = corpus_version
        records = self._chunk_records(list(documents))
        if not records:
            return 0
        vectors = _rows(self.embedder.encode([record.chunk_text for record in records]))
        if len(vectors) != len(records):
            raise PersistentEmbeddingError(
                f"Embedding 行数不一致：收到 {len(vectors)}，预期 {len(records)}。"
            )
        dimension = len(vectors[0])
        if dimension < 1 or any(len(vector) != dimension for vector in vectors):
            raise PersistentEmbeddingError("Embedding 维度不一致。")
        embedded = [
            replace(record, embedding=tuple(vector))
            for record, vector in zip(records, vectors, strict=True)
        ]
        return self.store.upsert(
            embedded,
            corpus_version=self.corpus_version,
            model_name=self.model_name,
        )

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or top_k < 1:
            return []
        vectors = _rows(self.embedder.encode([query]))
        if len(vectors) != 1:
            raise PersistentEmbeddingError("查询 Embedding 输出行数异常。")
        raw_hits = self.store.search(
            vectors[0],
            corpus_version=self.corpus_version,
            model_name=self.model_name,
            # Multiple chunks may belong to one document; deduplicate below.
            limit=max(top_k * 4, 20),
        )
        best_by_document: dict[str, RetrievalHit] = {}
        for hit in raw_hits:
            previous = best_by_document.get(hit.document.id)
            if previous is None or hit.score > previous.score:
                best_by_document[hit.document.id] = hit
        results = sorted(
            best_by_document.values(),
            key=lambda hit: (-hit.score, hit.document.id),
        )
        return [
            hit.model_copy(update={"retrieval_method": "vector_pg"})
            for hit in results[:top_k]
        ]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


class PersistentHybridRetriever:
    """Combine persistent semantic hits with deterministic keyword overlap."""

    def __init__(
        self,
        vector_retriever: PersistentVectorRetriever,
        *,
        documents: list[MedicalDocument] | None = None,
        semantic_weight: float = 0.7,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight 必须在 0 到 1 之间。")
        self.vector_retriever = vector_retriever
        self.documents = documents
        self.semantic_weight = semantic_weight

    def _documents(self) -> list[MedicalDocument]:
        if self.documents is None:
            self.documents = self.vector_retriever.store.list_documents(
                corpus_version=self.vector_retriever.corpus_version
            )
        return self.documents

    @staticmethod
    def _keyword_score(query: str, document: MedicalDocument) -> float:
        query_terms = _tokens(query)
        if not query_terms:
            return 0.0
        document_terms = _tokens(document.retrieval_text())
        return len(query_terms & document_terms) / len(query_terms)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or top_k < 1:
            return []
        documents = self._documents()
        semantic_hits = self.vector_retriever.search(query, top_k=max(top_k * 3, 20))
        semantic_by_id = {hit.document.id: hit for hit in semantic_hits}
        document_by_id = {document.id: document for document in documents}
        document_by_id.update({hit.document.id: hit.document for hit in semantic_hits})

        keyword_scores = {
            document.id: self._keyword_score(query, document)
            for document in documents
        }
        candidate_ids = set(semantic_by_id) | {
            document_id for document_id, score in keyword_scores.items() if score > 0
        }
        if not candidate_ids:
            return []

        max_keyword = max(keyword_scores.values(), default=1.0) or 1.0
        combined: list[RetrievalHit] = []
        for document_id in candidate_ids:
            semantic_hit = semantic_by_id.get(document_id)
            raw_semantic = (
                semantic_hit.semantic_score or semantic_hit.score
                if semantic_hit is not None
                else 0.0
            )
            normalized_semantic = (
                max(0.0, min(1.0, (raw_semantic + 1.0) / 2.0))
                if semantic_hit is not None
                else 0.0
            )
            keyword_score = keyword_scores.get(document_id, 0.0)
            normalized_keyword = keyword_score / max_keyword
            score = (
                self.semantic_weight * normalized_semantic
                + (1 - self.semantic_weight) * normalized_keyword
            )
            combined.append(
                RetrievalHit(
                    document=document_by_id[document_id],
                    score=score,
                    semantic_score=raw_semantic,
                    keyword_score=keyword_score,
                    retrieval_method="hybrid_pg",
                )
            )

        combined.sort(key=lambda hit: (-hit.score, hit.document.id))
        return combined[:top_k]
