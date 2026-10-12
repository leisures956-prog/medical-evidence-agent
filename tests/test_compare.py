from fastapi.testclient import TestClient

from app.comparison import classify_article
from app.main import app, get_search_service
from app.medical_models import MedicalDocument, RetrievalHit
from app.search import SearchService


def make_document(pmid: str, title: str, abstract: str) -> MedicalDocument:
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title=title,
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        metadata={"pmid": pmid},
    )


def test_classify_article_marks_correction_review_and_missing_abstract():
    correction = classify_article(
        make_document("1", "Correction: Immunotherapy study", "A correction notice.")
    )
    review = classify_article(
        make_document("2", "Lung cancer systematic review", "Review abstract.")
    )
    missing = classify_article(make_document("3", "Clinical trial study", ""))

    assert correction.flags == ["correction"]
    assert review.flags == ["review"]
    assert missing.flags == ["abstract_missing"]
    assert missing.abstract_available is False


class FakePubMedHybrid:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, query, candidate_limit, top_k):
        self.calls.append((query, candidate_limit, top_k))
        return [
            RetrievalHit(
                document=document,
                score=0.95 - index * 0.05,
                retrieval_method="pubmed_hybrid",
            )
            for index, document in enumerate(self.documents[:top_k])
        ]


def test_compare_api_returns_table_and_article_markers():
    documents = [
        make_document(
            "1",
            "Correction: Immunotherapy study",
            "This corrects a previously published article.",
        ),
        make_document(
            "2",
            "Lung cancer immunotherapy: a systematic review",
            "We review available evidence.",
        ),
        make_document("3", "Clinical trial study", ""),
    ]
    retriever = FakePubMedHybrid(documents)
    service = SearchService(pubmed_hybrid_retriever=retriever)
    app.dependency_overrides[get_search_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/compare",
            json={
                "question": "lung cancer immunotherapy",
                "top_k": 3,
                "candidate_limit": 20,
                "mode": "metadata_only",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_method"] == "pubmed_hybrid"
    assert payload["extraction_mode"] == "metadata_only"
    assert payload["compared_count"] == 3
    assert [row["pmid"] for row in payload["comparison_table"]] == ["1", "2", "3"]
    assert payload["comparison_table"][0]["markers"]["flags"] == ["correction"]
    assert payload["comparison_table"][1]["markers"]["flags"] == ["review"]
    assert payload["comparison_table"][2]["markers"]["flags"] == ["abstract_missing"]
    assert payload["comparison_table"][2]["evidence"]["method"]["status"] == "abstract_unavailable"
    assert "校正" in payload["summary"]
    assert "综述" in payload["summary"]
    assert "没有摘要" in payload["summary"]
    assert retriever.calls == [("lung cancer immunotherapy", 20, 3)]


def test_compare_requires_at_least_two_documents():
    response = TestClient(app).post(
        "/compare",
        json={
            "question": "lung cancer",
            "top_k": 1,
            "candidate_limit": 20,
        },
    )

    assert response.status_code == 422
