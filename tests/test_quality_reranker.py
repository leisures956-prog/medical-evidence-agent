from app.medical_models import MedicalDocument, RetrievalHit
from app.quality_reranker import QualityReranker, quality_flags
from app.search import SearchService


def _document(document_id: str, title: str, abstract: str) -> MedicalDocument:
    return MedicalDocument(
        id=document_id,
        source="pubmed",
        title=title,
        abstract=abstract,
        metadata={"pmid": document_id},
    )


def test_quality_flags_are_transparent():
    document = _document(
        "1",
        "Editorial: Correction and systematic review of lung cancer therapy",
        "",
    )

    assert quality_flags(document) == (
        "abstract_missing",
        "correction",
        "editorial",
        "review",
    )


def test_reranker_downranks_missing_abstract_editorial():
    ordinary = _document("ordinary", "Lung cancer trial", "Methods and results")
    editorial = _document("editorial", "Editorial: lung cancer update", "")
    hits = [
        RetrievalHit(
            document=editorial,
            score=0.95,
            retrieval_method="hybrid_persisted",
        ),
        RetrievalHit(
            document=ordinary,
            score=0.86,
            retrieval_method="hybrid_persisted",
        ),
    ]

    reranker = QualityReranker()
    result = reranker.rerank(hits, top_k=2)

    assert result[0].document.id == "ordinary"
    assert result[0].score == 0.86
    assert reranker.explain(editorial)["quality_penalty"] == 0.2


class FakeHybridRetriever:
    def search(self, query, *, top_k):
        return [
            RetrievalHit(
                document=_document("editorial", "Editorial: lung cancer", ""),
                score=0.95,
                retrieval_method="hybrid_persisted",
            ),
            RetrievalHit(
                document=_document("ordinary", "Lung cancer trial", "Results"),
                score=0.86,
                retrieval_method="hybrid_persisted",
            ),
        ][:top_k]


def test_search_service_exposes_quality_reranked_mode_without_test_client():
    service = SearchService(local_persistent_hybrid_retriever=FakeHybridRetriever())
    results, total = service.search(
        "lung cancer",
        limit=2,
        source="hybrid_persisted_reranked",
    )

    assert total == 2
    assert results[0].id == "ordinary"
