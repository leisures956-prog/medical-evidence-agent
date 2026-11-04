from app.agent_graph import create_agent_graph, run_agent_graph
from app.agent_nodes import answer_agent_node
from app.agent_state import AgentRequest
from app.medical_models import CitationBinding, GroundedAnswer, MedicalDocument, RetrievalHit


def make_hit() -> RetrievalHit:
    return RetrievalHit(
        document=MedicalDocument(
            id="pubmed:123456",
            source="pubmed",
            title="Example study",
            abstract="The intervention improved survival.",
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            metadata={"pmid": "123456"},
        ),
        score=0.9,
        retrieval_method="pubmed_hybrid",
    )


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


class RewriteRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k):
        self.calls.append(query)
        if query == "lung cancer immunotherapy":
            return []
        return [make_hit()]


def valid_answer():
    binding = CitationBinding(
        citation_number=1,
        pmid="123456",
        evidence_id="abstract-1",
        source_section="abstract",
        exact_text="The intervention improved survival.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    )
    return GroundedAnswer(
        answer="The intervention improved survival [1:abstract-1].",
        sources=[make_hit().document],
        citations=[1],
        citation_bindings=[binding],
        model="fake",
        prompt_version="test",
    )


class FakeAnswerer:
    def __init__(self, answers=None):
        self.answers = list(answers or [valid_answer()])
        self.calls = 0

    def answer(self, question, hits, evidence_records):
        self.calls += 1
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]


def invalid_answer():
    result = valid_answer()
    result.answer = "The intervention improved survival [9:abstract-1]."
    return result


def test_graph_completes_happy_path_with_fake_components():
    retriever = FakeRetriever([make_hit()])
    answerer = FakeAnswerer()

    state = run_agent_graph(
        AgentRequest(question="survival", top_k=1), retriever, answerer
    )

    assert state.status == "completed"
    assert state.citations == [1]
    assert state.attempt_count == 1
    assert answerer.calls == 1
    assert [event.step for event in state.trace] == [
        "query_analysis",
        "retrieval",
        "evidence_gate",
        "answer",
        "citation_check",
    ]


def test_graph_rewrites_query_once_after_empty_evidence():
    retriever = RewriteRetriever()
    answerer = FakeAnswerer()

    state = run_agent_graph(
        AgentRequest(question="lung cancer immunotherapy", top_k=1),
        retriever,
        answerer,
    )

    assert state.status == "completed"
    assert state.query_rewrite_count == 1
    assert len(retriever.calls) == 2
    assert retriever.calls[0] == "lung cancer immunotherapy"
    assert retriever.calls[1] != retriever.calls[0]
    assert state.metadata["original_question"] == "lung cancer immunotherapy"
    assert "query_rewrite" in [event.step for event in state.trace]


def test_graph_repairs_invalid_citation_at_most_once():
    retriever = FakeRetriever([make_hit()])
    answerer = FakeAnswerer([invalid_answer(), valid_answer()])

    state = run_agent_graph(
        AgentRequest(question="survival", top_k=1), retriever, answerer
    )

    assert state.status == "completed"
    assert state.answer_repair_count == 1
    assert answerer.calls == 2
    assert state.citations == [1]
    assert [event.step for event in state.trace].count("citation_check") == 2


def test_graph_stops_after_one_failed_answer_repair():
    retriever = FakeRetriever([make_hit()])
    answerer = FakeAnswerer([invalid_answer()])

    state = run_agent_graph(
        AgentRequest(question="survival", top_k=1), retriever, answerer
    )

    assert state.status == "failed"
    assert state.answer_repair_count == 1
    assert answerer.calls == 2
    assert state.trace[-1].step == "citation_check"
    assert state.trace[-1].status == "failed"


def test_create_graph_can_disable_both_recovery_paths():
    retriever = FakeRetriever([])
    answerer = FakeAnswerer()
    graph = create_agent_graph(
        retriever,
        answerer,
        max_query_rewrites=0,
        max_answer_repairs=0,
    )

    result = graph.invoke(
        {"agent": __import__("app.agent_state", fromlist=["AgentState"]).AgentState.from_request(
            AgentRequest(question="missing")
        )}
    )

    assert result["agent"].status == "abstained"
    assert answerer.calls == 0
