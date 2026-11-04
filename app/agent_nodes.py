from __future__ import annotations

import re
import inspect
from time import perf_counter
from typing import Any, TypedDict

from app.agent_state import AgentState, AgentTraceEvent
from app.citation_validation import validate_citations
from app.evidence_gate import evaluate_evidence
from app.grounded_llm import GroundedAnswerer
from app.hybrid_retriever import HybridRetriever
from app.medical_models import CitationBinding, RetrievalHit, StructuredEvidence


class RAGState(TypedDict, total=False):
    question: str
    hits: list[RetrievalHit]
    answer: str
    citations: list[int]
    evidence_records: list[StructuredEvidence]
    citation_bindings: list[CitationBinding]
    errors: list[str]


def retrieve_node(state: RAGState, retriever: HybridRetriever, top_k: int = 5) -> RAGState:
    state["hits"] = retriever.search(state["question"], top_k=top_k)
    return state


def answer_node(state: RAGState, answerer: GroundedAnswerer) -> RAGState:
    result = answerer.answer(state["question"], state.get("hits", []))
    state["answer"] = result.answer
    state["citations"] = result.citations
    return state


def citation_check_node(state: RAGState) -> RAGState:
    answer = state.get("answer", "")
    evidence_records = state.get("evidence_records")
    if evidence_records is not None:
        validation = validate_citations(answer, evidence_records)
        state["citations"] = sorted(
            {binding.citation_number for binding in validation.bindings}
        )
        state["citation_bindings"] = validation.bindings
        if validation.errors:
            state.setdefault("errors", []).extend(validation.errors)
        return state

    hit_count = len(state.get("hits", []))
    citations = sorted({int(item) for item in re.findall(r"\[(\d+)\]", answer)})
    invalid = [number for number in citations if number < 1 or number > hit_count]
    if not citations:
        state.setdefault("errors", []).append("回答没有引用来源。")
    if invalid:
        state.setdefault("errors", []).append(f"存在无效引用：{invalid}")
    state["citations"] = [number for number in citations if number not in invalid]
    return state


def _trace_agent_step(
    state: AgentState,
    step: str,
    status: str,
    started: float,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one trace event without coupling the nodes to LangGraph."""

    state.trace.append(
        AgentTraceEvent(
            step=step,
            status=status,  # type: ignore[arg-type]
            duration_ms=round((perf_counter() - started) * 1000, 3),
            details=details or {},
        )
    )


def query_analysis_agent_node(state: AgentState) -> AgentState:
    """Normalize the request before retrieval; this node never calls an LLM."""

    started = perf_counter()
    state.current_step = "query_analysis"
    state.status = "running"
    state.request.question = state.request.question.strip()
    if not state.request.question:
        state.status = "failed"
        state.errors.append("问题不能为空。")
        _trace_agent_step(state, "query_analysis", "failed", started)
        return state

    _trace_agent_step(
        state,
        "query_analysis",
        "completed",
        started,
        {"retrieval_method": state.request.retrieval_method},
    )
    state.current_step = "retrieval"
    return state


def _call_agent_retriever(retriever: object, state: AgentState) -> list[RetrievalHit]:
    search = getattr(retriever, "search", retriever)
    if not callable(search):
        raise TypeError("retriever 必须提供 search 方法或可调用接口。")

    parameters = inspect.signature(search).parameters
    kwargs: dict[str, Any] = {"top_k": state.request.top_k}
    if "candidate_limit" in parameters:
        kwargs["candidate_limit"] = state.request.candidate_limit
    return list(search(state.request.question, **kwargs))


def retrieval_agent_node(state: AgentState, retriever: object) -> AgentState:
    """Run an existing retriever without changing its ranking semantics."""

    started = perf_counter()
    state.current_step = "retrieval"
    state.status = "running"
    state.attempt_count += 1
    try:
        state.hits = _call_agent_retriever(retriever, state)
    except Exception as exc:  # noqa: BLE001 - graph state must capture node failures
        state.status = "failed"
        state.errors.append(f"检索节点失败：{type(exc).__name__}: {exc}")
        _trace_agent_step(state, "retrieval", "failed", started)
        return state

    _trace_agent_step(
        state,
        "retrieval",
        "completed",
        started,
        {"hit_count": len(state.hits)},
    )
    state.current_step = "evidence_gate"
    return state


def evidence_gate_agent_node(state: AgentState) -> AgentState:
    """Build verbatim evidence spans and fail closed when evidence is unusable."""

    started = perf_counter()
    state.current_step = "evidence_gate"
    if state.status == "failed":
        _trace_agent_step(state, "evidence_gate", "skipped", started)
        return state

    decision = evaluate_evidence(state.hits)
    state.evidence_records = decision.records
    state.metadata["evidence_gate_allowed"] = decision.allowed
    if not decision.allowed:
        state.answer = decision.message
        state.abstained = True
        state.abstention_reason = decision.reason
        state.status = "abstained"
        state.current_step = "final"
        _trace_agent_step(
            state,
            "evidence_gate",
            "completed",
            started,
            {"allowed": False, "reason": decision.reason},
        )
        return state

    state.status = "running"
    state.current_step = "answer"
    _trace_agent_step(
        state,
        "evidence_gate",
        "completed",
        started,
        {"allowed": True, "record_count": len(state.evidence_records)},
    )
    return state


def answer_agent_node(state: AgentState, answerer: GroundedAnswerer) -> AgentState:
    """Generate a grounded answer using the evidence records built by the gate."""

    started = perf_counter()
    state.current_step = "answer"
    if state.status in {"failed", "abstained"}:
        _trace_agent_step(state, "answer", "skipped", started)
        return state

    try:
        result = answerer.answer(
            state.request.question,
            state.hits,
            evidence_records=state.evidence_records,
        )
    except Exception as exc:  # noqa: BLE001 - graph state must capture node failures
        state.status = "failed"
        state.errors.append(f"回答节点失败：{type(exc).__name__}: {exc}")
        _trace_agent_step(state, "answer", "failed", started)
        return state

    state.answer = result.answer
    state.citations = list(result.citations)
    state.citation_bindings = list(result.citation_bindings)
    state.abstained = result.abstained
    state.abstention_reason = result.abstention_reason
    state.metadata.update(
        {
            "model": result.model,
            "prompt_version": result.prompt_version,
            "cache_hit": result.cache_hit,
        }
    )
    state.status = "abstained" if result.abstained else "running"
    state.current_step = "final" if result.abstained else "citation_check"
    _trace_agent_step(
        state,
        "answer",
        "completed",
        started,
        {"abstained": result.abstained},
    )
    return state


def citation_check_agent_node(state: AgentState) -> AgentState:
    """Validate exact evidence bindings before marking the agent completed."""

    started = perf_counter()
    state.current_step = "citation_check"
    if state.status in {"failed", "abstained"}:
        _trace_agent_step(state, "citation_check", "skipped", started)
        state.current_step = "final"
        return state

    validation = validate_citations(state.answer, state.evidence_records)
    state.citations = sorted({binding.citation_number for binding in validation.bindings})
    state.citation_bindings = validation.bindings
    if validation.errors:
        state.errors.extend(validation.errors)
        state.status = "failed"
        _trace_agent_step(
            state,
            "citation_check",
            "failed",
            started,
            {"errors": validation.errors},
        )
    else:
        state.status = "completed"
        _trace_agent_step(
            state,
            "citation_check",
            "completed",
            started,
            {"citation_count": len(state.citations)},
        )
    state.current_step = "final"
    return state
