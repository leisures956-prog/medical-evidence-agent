from __future__ import annotations

from collections.abc import Callable

from app.hybrid_retriever import HybridRetriever
from app.medical_models import MedicalDocument, RetrievalHit
from app.pubmed_client_v2 import PubMedClientError, PubMedClientV2
from app.semantic_retriever import EmbeddingError, SemanticRetriever


class PubMedHybridError(RuntimeError):
    """Raised when the PubMed candidate retrieval stage fails."""


class PubMedHybridEmbeddingError(RuntimeError):
    """Raised when the PubMed hybrid stage cannot use local embeddings."""


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
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight 必须在 0 到 1 之间。")
        self.pubmed_client = pubmed_client or PubMedClientV2()
        self.model_name = model_name
        self.semantic_weight = semantic_weight
        self._embedder = None
        self._semantic_factory = semantic_factory or self._default_semantic_factory

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

        try:
            documents = self.pubmed_client.search(query, limit=candidate_limit)
        except PubMedClientError as exc:
            raise PubMedHybridError(str(exc)) from exc

        documents = self._deduplicate(documents)
        if not documents:
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
        return ranked
