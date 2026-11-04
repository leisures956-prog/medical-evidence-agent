from app.grounded_llm import LLMCompletion
from app.main import AgentAnswerRequest, agent_answer, app
from app.medical_models import LLMUsage
from app.search import SearchHit


class FakeAgentSearchService:
    def __init__(self, abstract: str, *, empty: bool = False):
        self.abstract = abstract
        self.empty = empty
        self.calls = []

    def search(self, query, limit, source, candidate_limit=30, filters=None):
        self.calls.append((query, limit, source, candidate_limit))
        if self.empty:
            return [], 0
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


class FakeAgentProvider:
    def __init__(self):
        self.calls = 0

    def complete_with_metadata(self, messages):
        self.calls += 1
        return LLMCompletion(
            text="该研究报告了肺癌免疫治疗相关结果 [1:abstract-1]。",
            model="fake-agent-model",
            usage=LLMUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            estimated_cost_usd=0,
            latency_ms=12.5,
        )


def test_agent_answer_endpoint_returns_traceable_result():
    provider = FakeAgentProvider()
    service = FakeAgentSearchService(
        "This abstract reports an immunotherapy outcome in lung cancer patients."
    )
    response = agent_answer(
        AgentAnswerRequest(
            question="What does the study report?",
            retrieval_method="pubmed_hybrid",
            top_k=1,
            candidate_limit=20,
        ),
        service,
        provider,
    )
    body = response.model_dump()
    assert body["status"] == "completed"
    assert body["abstained"] is False
    assert body["citations"] == [1]
    assert body["citation_bindings"][0]["pmid"] == "123456"
    assert body["trace"][-1]["step"] == "citation_check"
    assert body["trace"][-1]["status"] == "completed"
    assert service.calls == [("What does the study report?", 1, "pubmed_hybrid", 20)]
    assert provider.calls == 1


def test_agent_answer_endpoint_abstains_without_retrieval_results():
    provider = FakeAgentProvider()
    service = FakeAgentSearchService("", empty=True)
    response = agent_answer(
        AgentAnswerRequest(
            question="unavailable evidence",
            retrieval_method="local",
            top_k=1,
        ),
        service,
        provider,
    )
    body = response.model_dump()
    assert body["status"] == "abstained"
    assert body["abstained"] is True
    assert body["abstention_reason"] == "query_rewrite_exhausted"
    assert body["attempt_count"] == 1
    assert provider.calls == 0


def test_agent_answer_route_is_registered():
    assert any(route.path == "/agent-answer" for route in app.routes)
