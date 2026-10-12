import httpx

from app.main import app, get_search_service
from app.medical_models import MedicalDocument, RetrievalHit
from app.pubmed_hybrid import PubMedHybridRetriever
from app.pubmed_client_v2 import PubMedClientV2
from app.search import SearchService
from fastapi.testclient import TestClient


class FakePubMedClient:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, query, limit):
        self.calls.append((query, limit))
        return self.documents


class FakeSemanticRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.embedder = object()

    def search(self, query, top_k=5):
        return [
            RetrievalHit(
                document=document,
                score=0.9 - index * 0.1,
                semantic_score=0.9 - index * 0.1,
                retrieval_method="semantic",
            )
            for index, document in enumerate(self.documents[:top_k])
        ]


def fake_factory(documents, model_name, embedder):
    return FakeSemanticRetriever(documents)


def make_document(pmid, title):
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title=title,
        abstract=f"Abstract for {title}.",
        metadata={"pmid": pmid},
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def test_pubmed_hybrid_fetches_candidates_and_returns_top_k():
    documents = [
        make_document("1", "alpha method"),
        make_document("2", "beta method"),
        make_document("1", "duplicate alpha method"),
    ]
    client = FakePubMedClient(documents)
    retriever = PubMedHybridRetriever(
        pubmed_client=client,
        semantic_factory=fake_factory,
    )

    hits = retriever.search("alpha", candidate_limit=20, top_k=2)

    assert client.calls == [("alpha", 20)]
    assert len(hits) == 2
    assert len({hit.document.id for hit in hits}) == 2


def test_pubmed_hybrid_requires_candidate_range():
    retriever = PubMedHybridRetriever(
        pubmed_client=FakePubMedClient([]),
        semantic_factory=fake_factory,
    )

    try:
        retriever.search("query", candidate_limit=10)
    except ValueError as exc:
        assert "20 到 50" in str(exc)
    else:
        raise AssertionError("candidate_limit outside the contract should fail")


def test_pubmed_hybrid_api_uses_new_source_without_changing_local_hybrid():
    documents = [make_document("123", "PubMed alpha evidence")]
    retriever = PubMedHybridRetriever(
        pubmed_client=FakePubMedClient(documents),
        semantic_factory=fake_factory,
    )
    service = SearchService(pubmed_hybrid_retriever=retriever)
    app.dependency_overrides[get_search_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/search",
            json={
                "query": "alpha",
                "limit": 1,
                "candidate_limit": 20,
                "source": "pubmed_hybrid",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["source"] == "pubmed_hybrid"
    assert response.json()["results"][0]["id"] == "pubmed:123"


def test_pubmed_hybrid_pubmed_timeout_is_wrapped():
    def timeout_handler(request: httpx.Request):
        raise httpx.ReadTimeout("timed out", request=request)

    service = SearchService(
        pubmed_hybrid_retriever=PubMedHybridRetriever(
            pubmed_client=PubMedClientV2(transport=httpx.MockTransport(timeout_handler)),
            semantic_factory=fake_factory,
        )
    )
    app.dependency_overrides[get_search_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/search",
            json={
                "query": "alpha",
                "limit": 1,
                "candidate_limit": 20,
                "source": "pubmed_hybrid",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "超时" in response.json()["detail"]
