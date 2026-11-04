from __future__ import annotations

import numpy as np

from app.cross_encoder_reranker import CrossEncoderReranker
from app.medical_models import MedicalDocument, RetrievalHit
from app.multi_query_retriever import MultiQueryRetriever, document_identity
from app.metadata_filter import RetrievalFilters
from app.search import SearchService


def _hit(pmid: str, title: str, abstract: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        document=MedicalDocument(
            id=f"pubmed:{pmid}",
            source="pubmed",
            title=title,
            abstract=abstract,
            doi=f"10.1/{pmid}",
            metadata={"pmid": pmid},
        ),
        score=score,
        retrieval_method="hybrid_persisted",
    )


class FakeRetriever:
    def search(self, query, *, top_k=5):
        common = _hit("1", "Lung cancer immunotherapy trial", "Results", 0.9)
        if "checkpoint" in query.lower():
            rows = [common, _hit("2", "Checkpoint inhibitor study", "Data", 0.8)]
        else:
            rows = [common, _hit("3", "Editorial: lung cancer", "", 0.95)]
        return rows[:top_k]


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        del batch_size, show_progress_bar
        return np.asarray(
            [5.0 if "trial" in passage.lower() else 4.0 for _, passage in pairs],
            dtype=np.float32,
        )


def test_document_identity_prefers_pmid():
    assert document_identity(_hit("123", "Title", "Abstract", 1).document) == (
        "pmid:123"
    )


def test_multi_query_fuses_deduplicates_and_quality_reranks():
    retriever = MultiQueryRetriever(
        FakeRetriever(),
        CrossEncoderReranker(model=FakeCrossEncoder()),
    )

    result = retriever.search_with_diagnostics(
        "lung cancer immunotherapy",
        top_k=3,
        candidate_limit=5,
    )

    assert result.hits[0].document.metadata["pmid"] == "1"
    assert len({document_identity(hit.document) for hit in result.hits}) == 3
    assert len(result.diagnostics.queries) > 1
    assert result.diagnostics.raw_hit_count > result.diagnostics.unique_hit_count
    assert result.diagnostics.duplicates_removed > 0
    assert result.diagnostics.filtered_out_count == 0


def test_multi_query_filters_before_final_top_k():
    retriever = MultiQueryRetriever(
        FakeRetriever(),
        CrossEncoderReranker(model=FakeCrossEncoder()),
    )

    result = retriever.search_with_diagnostics(
        "lung cancer immunotherapy",
        top_k=3,
        candidate_limit=5,
        filters=RetrievalFilters(
            require_abstract=True,
            exclude_article_types=("editorial",),
        ),
    )

    assert all(hit.document.abstract for hit in result.hits)
    assert result.diagnostics.excluded_by_abstract == 1
    assert result.diagnostics.filtered_out_count == 1


class FakeAdvancedRetriever:
    def search(self, query, *, top_k=5, candidate_limit=30, filters=None):
        del query, candidate_limit, filters
        return [_hit("1", "Lung cancer immunotherapy trial", "Results", 5.0)][
            :top_k
        ]


def test_search_service_exposes_advanced_persisted_mode():
    service = SearchService(multi_query_retriever=FakeAdvancedRetriever())

    results, total = service.search(
        "lung cancer immunotherapy",
        limit=5,
        source="advanced_persisted",
        candidate_limit=30,
    )

    assert total == 1
    assert results[0].id == "pubmed:1"


def test_search_service_rejects_filters_for_non_advanced_mode():
    service = SearchService(multi_query_retriever=FakeAdvancedRetriever())

    try:
        service.search(
            "lung cancer",
            limit=5,
            source="local",
            filters=RetrievalFilters(require_abstract=True),
        )
    except ValueError as exc:
        assert "advanced_persisted" in str(exc)
    else:
        raise AssertionError("非 advanced_persisted 模式应拒绝 filters")
