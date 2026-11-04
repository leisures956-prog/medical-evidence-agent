from app.agent_nodes import (
    answer_agent_node,
    citation_check_agent_node,
    evidence_gate_agent_node,
    query_analysis_agent_node,
    retrieval_agent_node,
)
from app.agent_state import AgentRequest, AgentState
from app.medical_models import CitationBinding, GroundedAnswer, MedicalDocument, RetrievalHit


def make_hit(abstract: str = "The intervention improved survival.") -> RetrievalHit:
    return RetrievalHit(
        document=MedicalDocument(
            id="pubmed:123456",
            source="pubmed",
            title="Example study",
            abstract=abstract,
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            metadata={"pmid": "123456"},
        ),
        score=0.9,
        retrieval_method="pubmed_hybrid",
    )


class LocalRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


class PubMedStyleRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, candidate_limit, top_k):
        self.calls.append((query, candidate_limit, top_k))
        return self.hits[:top_k]


class FakeAnswerer:
    def __init__(self, answer="The intervention improved survival [1:abstract-1]."):
        self.answer_text = answer
        self.calls = []

    def answer(self, question, hits, evidence_records):
        self.calls.append((question, hits, evidence_records))
        binding = CitationBinding(
            citation_number=1,
            pmid="123456",
            evidence_id="abstract-1",
            source_section="abstract",
            exact_text="The intervention improved survival.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        )
        return GroundedAnswer(
            answer=self.answer_text,
            sources=[hit.document for hit in hits],
            citations=[1],
            citation_bindings=[binding],
            model="fake",
            prompt_version="test",
        )


def run_to_gate(state, retriever):
    query_analysis_agent_node(state)
    retrieval_agent_node(state, retriever)
    evidence_gate_agent_node(state)
    return state


def test_agent_nodes_complete_happy_path_and_keep_traceable_evidence():
    state = AgentState.from_request(AgentRequest(question="survival"))
    retriever = LocalRetriever([make_hit()])
    answerer = FakeAnswerer()

    run_to_gate(state, retriever)
    answer_agent_node(state, answerer)
    citation_check_agent_node(state)

    assert state.status == "completed"
    assert state.answer.startswith("The intervention")
    assert state.citations == [1]
    assert state.citation_bindings[0].evidence_id == "abstract-1"
    assert [event.step for event in state.trace] == [
        "query_analysis",
        "retrieval",
        "evidence_gate",
        "answer",
        "citation_check",
    ]
    assert len(answerer.calls) == 1


def test_pubmed_style_retriever_receives_candidate_limit_and_top_k():
    state = AgentState.from_request(
        AgentRequest(
            question="survival",
            retrieval_method="pubmed_hybrid",
            top_k=2,
            candidate_limit=25,
        )
    )
    retriever = PubMedStyleRetriever([make_hit(), make_hit()])

    query_analysis_agent_node(state)
    retrieval_agent_node(state, retriever)

    assert retriever.calls == [("survival", 25, 2)]
    assert len(state.hits) == 2
    assert state.current_step == "evidence_gate"


def test_no_retrieval_results_fail_closed_without_calling_answerer():
    state = AgentState.from_request(AgentRequest(question="missing"))
    answerer = FakeAnswerer()

    run_to_gate(state, LocalRetriever([]))
    answer_agent_node(state, answerer)
    citation_check_agent_node(state)

    assert state.status == "abstained"
    assert state.abstention_reason == "no_retrieval_results"
    assert state.answer == "没有检索到可用于回答该问题的资料。"
    assert answerer.calls == []
    assert state.trace[-1].status == "skipped"


def test_retrieval_and_answer_errors_are_recorded_in_state():
    state = AgentState.from_request(AgentRequest(question="survival"))

    class BrokenRetriever:
        def search(self, query, top_k):
            raise RuntimeError("offline")

    retrieval_agent_node(state, BrokenRetriever())

    assert state.status == "failed"
    assert "检索节点失败" in state.errors[0]
    assert state.trace[-1].status == "failed"

    state = AgentState.from_request(AgentRequest(question="survival"))
    state.hits = [make_hit()]
    evidence_gate_agent_node(state)

    class BrokenAnswerer:
        def answer(self, question, hits, evidence_records):
            raise RuntimeError("model unavailable")

    answer_agent_node(state, BrokenAnswerer())

    assert state.status == "failed"
    assert "回答节点失败" in state.errors[0]
    assert state.trace[-1].status == "failed"
