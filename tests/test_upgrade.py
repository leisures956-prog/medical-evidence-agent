import httpx
import pytest
from fastapi.testclient import TestClient
from app.main import app, get_search_service
from app.search import SearchService, RichPubMedAdapter
from app.pubmed_client_v2 import PubMedClientV2
from app.grounded_llm import GroundedAnswerer, LLMProviderError
from app.medical_models import MedicalDocument, RetrievalHit
from test_next_stage import PUBMED_XML


def test_pubmed_metadata_reaches_api():
    def handler(request):
        if request.url.path.endswith('esearch.fcgi'):
            return httpx.Response(200, json={'esearchresult': {'idlist': ['123456']}})
        return httpx.Response(200, text=PUBMED_XML)
    adapter = RichPubMedAdapter()
    adapter.client = PubMedClientV2(transport=httpx.MockTransport(handler))
    service = SearchService(pubmed_client=adapter)
    app.dependency_overrides[get_search_service] = lambda: service
    try:
        response = TestClient(app).post('/search', json={'query': 'evidence', 'source': 'pubmed'})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    hit = response.json()['results'][0]
    assert hit['doi'] == '10.1000/example'
    assert hit['authors'] == ['Wei Zhang']
    assert 'Result text.' in hit['content']


def test_llm_rejects_mixed_valid_invalid_citations():
    class Provider:
        def complete(self, messages):
            return 'Claim [1] and [99]'
    hit = RetrievalHit(document=MedicalDocument(id='one', title='Evidence'),
                       score=1, retrieval_method='local')
    with pytest.raises(LLMProviderError, match='越界'):
        GroundedAnswerer(Provider()).answer('question', [hit])


def test_llm_missing_configuration_returns_error(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    monkeypatch.delenv('LLM_MODEL', raising=False)
    response = TestClient(app).post('/answer', json={'question': 'Python', 'mode': 'llm'})
    assert response.status_code == 502


def test_empty_semantic_corpus_does_not_load_model():
    from app.semantic_retriever import SemanticRetriever
    assert SemanticRetriever([]).search('question') == []


def test_hybrid_api_with_fake_embeddings(monkeypatch):
    from app.semantic_retriever import SemanticRetriever
    def fake_search(self, query, top_k=5):
        return [RetrievalHit(document=doc, score=0.8, semantic_score=0.8,
                             retrieval_method='semantic') for doc in self.documents]
    monkeypatch.setattr(SemanticRetriever, 'search', fake_search)
    response = TestClient(app).post('/search', json={'query': 'Python', 'source': 'hybrid'})
    assert response.status_code == 200
    assert response.json()['results']
    assert response.json()['source'] == 'hybrid'


def test_hybrid_blank_query_is_rejected():
    response = TestClient(app).post('/search', json={'query': ' ', 'source': 'hybrid'})
    assert response.status_code == 400
