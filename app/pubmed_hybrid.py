from __future__ import annotations

from collections.abc import Callable
import logging
import os
import time
from threading import Lock

from app.hybrid_retriever import HybridRetriever
from app.medical_models import MedicalDocument, RetrievalHit
from app.pubmed_client_v2 import PubMedClientError, PubMedClientV2
from app.request_context import get_request_id
from app.semantic_retriever import EmbeddingError, SemanticRetriever


class PubMedHybridError(RuntimeError):
    """Raised when the PubMed candidate retrieval stage fails."""


class PubMedHybridEmbeddingError(RuntimeError):
    """Raised when the PubMed hybrid stage cannot use local embeddings."""


logger = logging.getLogger(__name__)


class PubMedHybridRetriever:
    """Retrieve PubMed candidates, then rank them with the existing hybrid retriever.

    The existing ``hybrid`` source remains the local-document mode. This adapter
    only creates a temporary PubMed corpus for the new ``pubmed_hybrid`` mode.
    """

    def __init__(
        self,
        pubmed_client: PubMedClientV2 | None = None,
        model_name: str | None = None,
        semantic_weight: float = 0.7,
        semantic_factory: Callable[[list[MedicalDocument], str | None, object | None], SemanticRetriever]
        | None = None,
        cache_enabled: bool | None = None,
        cache_ttl_seconds: float | None = None,
        cache_max_entries: int | None = None,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight 必须在 0 到 1 之间。")
        self.pubmed_client = pubmed_client or PubMedClientV2()
        self.model_name = model_name
        self.semantic_weight = semantic_weight
        self._embedder = None
        self._semantic_factory = semantic_factory or self._default_semantic_factory
        self.cache_enabled = (
            os.getenv("PUBMED_HYBRID_CACHE_ENABLED", "1").strip().lower()
            not in {"0", "false", "no"}
            if cache_enabled is None
            else bool(cache_enabled)
        )
        self.cache_ttl_seconds = (
            float(os.getenv("PUBMED_HYBRID_CACHE_TTL_SECONDS", "300"))
            if cache_ttl_seconds is None
            else float(cache_ttl_seconds)
        )
        self.cache_max_entries = (
            int(os.getenv("PUBMED_HYBRID_CACHE_MAX_ENTRIES", "128"))
            if cache_max_entries is None
            else int(cache_max_entries)
        )
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds 不能小于 0。")
        if self.cache_max_entries < 1:
            raise ValueError("cache_max_entries 必须大于 0。")
        self._cache: dict[str, tuple[float, list[RetrievalHit]]] = {}
        self._cache_lock = Lock()

    @staticmethod
    def _default_semantic_factory(
        documents: list[MedicalDocument],
        model_name: str | None,
        embedder: object | None,
    ) -> SemanticRetriever:
        return SemanticRetriever(documents, model_name=model_name, embedder=embedder)

    @staticmethod
    def _deduplicate(documents: list[MedicalDocument]) -> list[MedicalDocument]:
        seen: set[str] = set()
        unique: list[MedicalDocument] = []
        for document in documents:
            if document.id in seen:
                continue
            seen.add(document.id)
            unique.append(document)
        return unique

    def _cache_key(self, query: str, candidate_limit: int, top_k: int) -> str:
        model = self.model_name or os.getenv("EMBEDDING_MODEL", "")
        return "\x00".join(
            (
                query.strip().casefold(),
                str(candidate_limit),
                str(top_k),
                str(self.semantic_weight),
                model,
            )
        )

    @staticmethod
    def _copy_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return [hit.model_copy(deep=True) for hit in hits]

    def _get_cached(self, key: str) -> list[RetrievalHit] | None:
        if not self.cache_enabled or self.cache_ttl_seconds == 0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            item = self._cache.get(key)
            if item is None:
                return None
            created, hits = item
            if now - created > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            return self._copy_hits(hits)

    def _set_cached(self, key: str, hits: list[RetrievalHit]) -> None:
        if not self.cache_enabled or self.cache_ttl_seconds == 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), self._copy_hits(hits))
            while len(self._cache) > self.cache_max_entries:
                oldest_key = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest_key, None)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def search(
        self,
        query: str,
        candidate_limit: int = 30,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []
        if not 20 <= candidate_limit <= 50:
            raise ValueError("candidate_limit 必须在 20 到 50 之间。")
        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1。")

        cache_key = self._cache_key(query, candidate_limit, top_k)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info(
                "pubmed_hybrid_cache_hit",
                extra={
                    "request_id": get_request_id(),
                    "candidate_limit": candidate_limit,
                    "top_k": top_k,
                    "result_count": len(cached),
                },
            )
            return cached

        try:
            documents = self.pubmed_client.search(query, limit=candidate_limit)
        except PubMedClientError as exc:
            raise PubMedHybridError(str(exc)) from exc

        documents = self._deduplicate(documents)
        if not documents:
            self._set_cached(cache_key, [])
            return []

        semantic = self._semantic_factory(documents, self.model_name, self._embedder)
        try:
            ranked = HybridRetriever(
                documents=documents,
                semantic_retriever=semantic,
                semantic_weight=self.semantic_weight,
            ).search(query, top_k=min(top_k, len(documents)))
        except EmbeddingError as exc:
            raise PubMedHybridEmbeddingError(str(exc)) from exc

        self._embedder = getattr(semantic, "embedder", self._embedder)
        self._set_cached(cache_key, ranked)
        return self._copy_hits(ranked)
