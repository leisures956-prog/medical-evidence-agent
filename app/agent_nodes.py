from __future__ import annotations

import re
from typing import Any, TypedDict

from app.citation_validation import validate_citations
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
