"""LangGraph orchestration for the Stage 7 agent workflow.

The graph is intentionally a thin orchestration layer around the deterministic
state and node functions from :mod:`app.agent_nodes`.  It does not change the
retrieval ranking, the evidence model, or the existing ``/answer`` endpoint.

The graph has two bounded recovery paths:

* evidence failure -> one deterministic query rewrite -> retrieval again;
* citation failure -> one answer repair -> citation check again.

All tests can use local fakes, so importing and testing this module does not
require PubMed, Ollama, or a network connection.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent_nodes import (
    answer_agent_node,
    citation_check_agent_node,
    evidence_gate_agent_node,
    query_analysis_agent_node,
    retrieval_agent_node,
)
from app.agent_state import AgentRequest, AgentState, AgentTraceEvent
from app.query_rewriter import MedicalQueryRewriter


class AgentGraphState(TypedDict):
    """LangGraph state wrapper around the existing Pydantic agent state."""

    agent: AgentState


RouteName = Literal[
    "retrieval",
    "evidence_gate",
    "query_rewrite",
    "answer",
    "citation_check",
    "answer_repair",
    "end",
]


def _trace(
    state: AgentState,
    step: str,
    status: Literal["started", "completed", "skipped", "failed"],
    started: float,
    details: dict[str, Any] | None = None,
) -> None:
    state.trace.append(
        AgentTraceEvent(
            step=step,
            status=status,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            details=details or {},
        )
    )


def _route_after_query_analysis(graph_state: AgentGraphState) -> RouteName:
    return "end" if graph_state["agent"].status == "failed" else "retrieval"


def _route_after_retrieval(graph_state: AgentGraphState) -> RouteName:
    return "end" if graph_state["agent"].status == "failed" else "evidence_gate"


def _route_after_evidence_gate(
    graph_state: AgentGraphState,
    *,
    max_query_rewrites: int,
) -> RouteName:
    state = graph_state["agent"]
    if state.status == "running":
        return "answer"
    if (
        state.status == "abstained"
        and state.query_rewrite_count < max_query_rewrites
    ):
        return "query_rewrite"
    return "end"


def _route_after_answer(graph_state: AgentGraphState) -> RouteName:
    return "citation_check"


def _route_after_citation_check(
    graph_state: AgentGraphState,
    *,
    max_answer_repairs: int,
) -> RouteName:
    state = graph_state["agent"]
    if state.status == "failed" and state.answer_repair_count < max_answer_repairs:
        return "answer_repair"
    return "end"


def _route_after_query_rewrite(graph_state: AgentGraphState) -> RouteName:
    return "retrieval" if graph_state["agent"].status == "running" else "end"


def _make_query_rewrite_node(
    query_rewriter: MedicalQueryRewriter,
) -> Callable[[AgentGraphState], dict[str, AgentState]]:
    def query_rewrite_node(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        started = perf_counter()
        state.current_step = "query_rewrite"

        original = str(state.metadata.get("original_question", state.request.question))
        state.metadata.setdefault("original_question", original)
        try:
            variants = list(query_rewriter.rewrite(original).queries)
        except Exception as exc:  # noqa: BLE001 - preserve failure in graph state
            state.status = "failed"
            state.errors.append(f"查询改写节点失败：{type(exc).__name__}: {exc}")
            _trace(state, "query_rewrite", "failed", started)
            state.current_step = "final"
            return {"agent": state}

        # The first variant is the original query.  query_rewrite_count is the
        # number of rewrites already attempted, so index 1 is the next variant.
        next_index = state.query_rewrite_count + 1
        if next_index >= len(variants):
            state.status = "abstained"
            state.abstained = True
            state.abstention_reason = "query_rewrite_exhausted"
            state.answer = "证据不足，查询改写后仍未找到可用于回答该问题的资料。"
            state.current_step = "final"
            _trace(
                state,
                "query_rewrite",
                "completed",
                started,
                {"rewritten": False, "reason": "no_new_variant"},
            )
            return {"agent": state}

        rewritten = variants[next_index]
        state.query_rewrite_count += 1
        state.request.question = rewritten
        state.hits = []
        state.evidence_records = []
        state.answer = ""
        state.citations = []
        state.citation_bindings = []
        state.abstained = False
        state.abstention_reason = None
        state.status = "running"
        state.current_step = "retrieval"
        state.metadata["query_variants"] = variants
        state.metadata["last_rewritten_question"] = rewritten
        _trace(
            state,
            "query_rewrite",
            "completed",
            started,
            {"original": original, "rewritten": rewritten},
        )
        return {"agent": state}

    return query_rewrite_node


def _make_answer_repair_node(
    answerer: object,
) -> Callable[[AgentGraphState], dict[str, AgentState]]:
    def answer_repair_node(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        started = perf_counter()
        state.answer_repair_count += 1
        state.status = "running"
        state.current_step = "answer_repair"

        # A provider-backed answerer may have cached the invalid completion.
        # Discard only that last provider cache entry, if the implementation
        # exposes the optional hook; fakes do not need to implement it.
        discard = getattr(answerer, "_discard_invalid_cache", None)
        if callable(discard):
            discard()

        state.errors = [
            error
            for error in state.errors
            if not ("引用" in error or "evidence_id" in error)
        ]
        _trace(
            state,
            "answer_repair",
            "completed",
            started,
            {"attempt": state.answer_repair_count},
        )
        answer_agent_node(state, answerer)  # type: ignore[arg-type]
        return {"agent": state}

    return answer_repair_node


def create_agent_graph(
    retriever: object,
    answerer: object,
    *,
    query_rewriter: MedicalQueryRewriter | None = None,
    max_query_rewrites: int = 1,
    max_answer_repairs: int = 1,
):
    """Create and compile the bounded Stage 7C LangGraph workflow.

    ``retriever`` and ``answerer`` are injected into node closures.  This keeps
    the graph easy to test with local fakes and avoids global provider state.
    """

    if max_query_rewrites < 0:
        raise ValueError("max_query_rewrites 不能小于 0。")
    if max_answer_repairs < 0:
        raise ValueError("max_answer_repairs 不能小于 0。")

    rewriter = query_rewriter or MedicalQueryRewriter()
    query_rewrite_node = _make_query_rewrite_node(rewriter)
    answer_repair_node = _make_answer_repair_node(answerer)

    def query_analysis(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        query_analysis_agent_node(state)
        return {"agent": state}

    def retrieval(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        retrieval_agent_node(state, retriever)
        return {"agent": state}

    def evidence_gate(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        evidence_gate_agent_node(state)
        return {"agent": state}

    def answer(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        answer_agent_node(state, answerer)  # type: ignore[arg-type]
        return {"agent": state}

    def citation_check(graph_state: AgentGraphState) -> dict[str, AgentState]:
        state = graph_state["agent"]
        citation_check_agent_node(state)
        return {"agent": state}

    builder = StateGraph(AgentGraphState)
    builder.add_node("query_analysis", query_analysis)
    builder.add_node("retrieval", retrieval)
    builder.add_node("evidence_gate", evidence_gate)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("answer", answer)
    builder.add_node("citation_check", citation_check)
    builder.add_node("answer_repair", answer_repair_node)

    builder.add_edge(START, "query_analysis")
    builder.add_conditional_edges(
        "query_analysis",
        _route_after_query_analysis,
        {"retrieval": "retrieval", "end": END},
    )
    builder.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {"evidence_gate": "evidence_gate", "end": END},
    )
    builder.add_conditional_edges(
        "evidence_gate",
        lambda graph_state: _route_after_evidence_gate(
            graph_state, max_query_rewrites=max_query_rewrites
        ),
        {
            "answer": "answer",
            "query_rewrite": "query_rewrite",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "query_rewrite",
        _route_after_query_rewrite,
        {"retrieval": "retrieval", "end": END},
    )
    builder.add_edge("answer", "citation_check")
    builder.add_conditional_edges(
        "citation_check",
        lambda graph_state: _route_after_citation_check(
            graph_state, max_answer_repairs=max_answer_repairs
        ),
        {"answer_repair": "answer_repair", "end": END},
    )
    builder.add_edge("answer_repair", "citation_check")
    return builder.compile()


def run_agent_graph(
    request: AgentRequest | AgentState,
    retriever: object,
    answerer: object,
    *,
    query_rewriter: MedicalQueryRewriter | None = None,
    max_query_rewrites: int = 1,
    max_answer_repairs: int = 1,
    config: dict[str, Any] | None = None,
) -> AgentState:
    """Run the compiled graph and return the existing stable ``AgentState``."""

    state = request if isinstance(request, AgentState) else AgentState.from_request(request)
    graph = create_agent_graph(
        retriever,
        answerer,
        query_rewriter=query_rewriter,
        max_query_rewrites=max_query_rewrites,
        max_answer_repairs=max_answer_repairs,
    )
    result = graph.invoke({"agent": state}, config=config)
    return result["agent"]


__all__ = ["AgentGraphState", "create_agent_graph", "run_agent_graph"]
