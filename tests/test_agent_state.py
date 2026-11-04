import pytest
from pydantic import ValidationError

from app.agent_state import AgentRequest, AgentState, AgentTraceEvent
from app.medical_models import CitationBinding


def test_agent_request_has_safe_defaults_and_round_trips():
    request = AgentRequest(question="lung cancer immunotherapy")

    assert request.retrieval_method == "hybrid"
    assert request.top_k == 5
    assert request.candidate_limit == 30
    assert AgentRequest.model_validate(request.model_dump()) == request


def test_agent_request_rejects_empty_question_and_invalid_limits():
    with pytest.raises(ValidationError):
        AgentRequest(question="")
    with pytest.raises(ValidationError):
        AgentRequest(question="test", top_k=0)
    with pytest.raises(ValidationError):
        AgentRequest(question="test", candidate_limit=501)


def test_state_starts_without_side_effects_and_projects_result():
    state = AgentState.from_request(
        AgentRequest(question="sepsis biomarkers", retrieval_method="pubmed_hybrid")
    )

    assert state.status == "pending"
    assert state.current_step == "query_analysis"
    assert state.hits == []
    assert state.evidence_records == []
    assert state.answer == ""
    assert state.to_result().status == "pending"
    assert state.to_result().attempt_count == 0


def test_trace_and_citation_binding_are_serializable():
    state = AgentState.from_request(AgentRequest(question="test"))
    state.status = "completed"
    state.current_step = "final"
    state.answer = "The finding is reported [1:abstract-1]."
    state.citations = [1]
    state.citation_bindings = [
        CitationBinding(
            citation_number=1,
            pmid="123456",
            evidence_id="abstract-1",
            source_section="abstract",
            exact_text="The finding is reported.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        )
    ]
    state.trace.append(
        AgentTraceEvent(
            step="citation_check",
            status="completed",
            duration_ms=12.5,
            details={"valid": True},
        )
    )

    result = state.to_result()

    assert result.status == "completed"
    assert result.citations == [1]
    assert result.citation_bindings[0].evidence_id == "abstract-1"
    assert result.trace[0].details == {"valid": True}
    assert AgentState.model_validate(state.model_dump()) == state
