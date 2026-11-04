import pytest
from fastapi.testclient import TestClient

from app.grounded_llm import LLMCompletion
from app.main import app, get_llm_provider, get_search_service
from app.medical_models import LLMUsage
from app.search import SearchHit


pytestmark = pytest.mark.api_runtime


class FakeSearchService:
    def __init__(self, abstract: str):
        self.abstract = abstract

    def search(self, query, limit, source, candidate_limit=30, filters=None):
        hits = [
            SearchHit(
                id="pubmed:123456",
                title="Example lung cancer study",
                content=self.abstract,
                score=0.9,
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/123456/",
                year=2025,
            )
        ]
        return hits[:limit], len(hits)


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def complete_with_metadata(self, messages):
        self.calls += 1
        return LLMCompletion(
            text="摘要报告了相关研究结果 [1:abstract-1]。",
            model="fake-model",
            usage=LLMUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            estimated_cost_usd=0,
            latency_ms=12.5,
        )


def test_answer_llm_returns_traceable_evidence_and_runtime_metadata():
    provider = FakeProvider()
    app.dependency_overrides[get_search_service] = lambda: FakeSearchService(
        "This abstract reports an immunotherapy outcome in lung cancer patients."
    )
    app.dependency_overrides[get_llm_provider] = lambda: provider
    try:
        response = TestClient(app).post(
            "/answer",
            json={
                "question": "What does the study report?",
                "source": "pubmed_hybrid",
                "mode": "llm",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["model"] == "fake-model"
    assert body["usage"]["total_tokens"] == 120
    assert body["citation_bindings"][0]["pmid"] == "123456"
    assert body["citation_bindings"][0]["source_section"] == "abstract"
    assert body["quality"]["valid"] is True
    assert body["quality"]["citation_coverage"] == 1.0


def test_answer_refuses_missing_abstract_without_calling_provider():
    provider = FakeProvider()
    app.dependency_overrides[get_search_service] = lambda: FakeSearchService("")
    app.dependency_overrides[get_llm_provider] = lambda: provider
    try:
        response = TestClient(app).post(
            "/answer",
            json={"question": "What does it report?", "source": "pubmed", "mode": "llm"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["abstention_reason"] == "abstract_unavailable"
    assert provider.calls == 0
