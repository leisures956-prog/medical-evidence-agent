from __future__ import annotations

import numpy as np
import pytest

from app.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from app.medical_models import MedicalDocument, RetrievalHit
from app.search import SearchService


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        del batch_size, show_progress_bar
        return np.asarray(
            [
                5.0 if "checkpoint" in passage.lower() else -2.0
                for _, passage in pairs
            ],
            dtype=np.float32,
        )


def _hit(document_id: str, title: str, abstract: str, score: float):
    return RetrievalHit(
        document=MedicalDocument(
            id=document_id,
            source="pubmed",
            title=title,
            abstract=abstract,
            metadata={"pmid": document_id},
        ),
        score=score,
        semantic_score=score,
        retrieval_method="hybrid_persisted",
    )


def test_cross_encoder_reranks_by_query_document_score():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    hits = [
        _hit("generic", "Lung cancer overview", "General background", 0.95),
        _hit(
            "specific",
            "Lung cancer immunotherapy",
            "Immune checkpoint inhibitor outcomes",
            0.80,
        ),
    ]

    result = reranker.rerank("lung cancer immunotherapy", hits, top_k=2)

    assert [hit.document.id for hit in result] == ["specific", "generic"]
    assert result[0].score == 5.0
    assert result[0].retrieval_method == "hybrid_persisted_cross_reranked"


def test_cross_encoder_quality_penalties_downrank_editorial_without_abstract():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    hits = [
        _hit(
            "editorial",
            "Editorial: lung cancer immune checkpoint update",
            "",
            0.95,
        ),
        _hit(
            "study",
            "Lung cancer immune checkpoint trial",
            "Prospective patient outcomes",
            0.80,
        ),
    ]

    result = reranker.rerank_with_quality(
        "lung cancer immunotherapy",
        hits,
        top_k=2,
    )

    assert [hit.document.id for hit in result] == ["study", "editorial"]
    assert result[0].retrieval_method == (
        "hybrid_persisted_cross_quality_reranked"
    )


def test_cross_encoder_requires_explicit_local_model(monkeypatch):
    monkeypatch.delenv("CROSS_ENCODER_MODEL", raising=False)
    reranker = CrossEncoderReranker()

    with pytest.raises(CrossEncoderRerankerError, match="CROSS_ENCODER_MODEL"):
        reranker.load_model()


class FakeHybridRetriever:
    def search(self, query, *, top_k):
        del query
        return [
            _hit("generic", "Lung cancer overview", "General background", 0.95),
            _hit(
                "specific",
                "Lung cancer immunotherapy",
                "Immune checkpoint inhibitor outcomes",
                0.80,
            ),
        ][:top_k]


def test_search_service_exposes_cross_encoder_mode():
    service = SearchService(
        local_persistent_hybrid_retriever=FakeHybridRetriever(),
        cross_encoder_reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
    )

    results, total = service.search(
        "lung cancer immunotherapy",
        limit=2,
        source="hybrid_persisted_cross_reranked",
    )

    assert total == 2
    assert results[0].id == "specific"


def test_search_service_exposes_cross_encoder_quality_mode():
    service = SearchService(
        local_persistent_hybrid_retriever=FakeHybridRetriever(),
        cross_encoder_reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
    )

    results, total = service.search(
        "lung cancer immunotherapy",
        limit=2,
        source="hybrid_persisted_cross_quality_reranked",
    )

    assert total == 2
    assert results[0].id == "specific"
