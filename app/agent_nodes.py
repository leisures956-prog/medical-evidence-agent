from __future__ import annotations

import re
from typing import Any, TypedDict

from app.grounded_llm import GroundedAnswerer
from app.hybrid_retriever import HybridRetriever
from app.medical_models import RetrievalHit


class RAGState(TypedDict, total=False):
    question: str
    hits: list[RetrievalHit]
    answer: str
    citations: list[int]
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
    hit_count = len(state.get("hits", []))
    citations = sorted({int(item) for item in re.findall(r"\[(\d+)\]", answer)})
    invalid = [number for number in citations if number < 1 or number > hit_count]
    if not citations:
        state.setdefault("errors", []).append("回答没有引用来源。")
    if invalid:
        state.setdefault("errors", []).append(f"存在无效引用：{invalid}")
    state["citations"] = [number for number in citations if number not in invalid]
    return state
