from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.search import (
    ExternalServiceError,
    SearchHit,
    SearchService,
    VectorSearchError,
)


app = FastAPI(title="Search Answer API", version="0.2.0")

SearchSource = Literal["local", "pubmed", "vector", "hybrid", "all"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)
    source: SearchSource = "local"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=10)
    source: SearchSource = "local"
    mode: Literal["extractive", "llm"] = "extractive"


class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    source: str
    url: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    journal: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    source: SearchSource


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    confidence: float
    citations: list[int] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str


search_service = SearchService()


def get_search_service() -> SearchService:
    return search_service


def _serialize_hit(hit: SearchHit) -> SearchResult:
    return SearchResult(**asdict(hit))


def _run_search(service: SearchService, query: str, limit: int, source: SearchSource):
    try:
        return service.search(query, limit, source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExternalServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VectorSearchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "service": "search-answer-api"}


@app.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
):
    results, total = _run_search(service, request.query, request.limit, request.source)
    return {
        "results": [_serialize_hit(item) for item in results],
        "total": total,
        "source": request.source,
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    service: SearchService = Depends(get_search_service),
):
    results, _ = _run_search(service, request.question, request.top_k, request.source)
    sources = [_serialize_hit(item) for item in results]

    if not results:
        return {"answer": "没有检索到相关资料。", "sources": [], "confidence": 0.0}

    if request.mode == "llm":
        from app.grounded_llm import GroundedAnswerer, OpenAICompatibleProvider, LLMProviderError
        from app.medical_models import RetrievalHit
        from app.search import medical_document
        hits = [RetrievalHit(document=medical_document(item), score=item.score,
                             retrieval_method=request.source) for item in results]
        try:
            generated = GroundedAnswerer(OpenAICompatibleProvider()).answer(request.question, hits)
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"answer": generated.answer, "sources": sources,
                "confidence": 0.0, "citations": generated.citations}

    top_result = results[0]
    citations = " ".join(
        f"[{index}] {item.title}" for index, item in enumerate(results, start=1)
    )
    return {
        "answer": (
            f"根据检索结果，最相关的内容是：{top_result.title}。"
            f"{top_result.content}\n来源：{citations}"
        ),
        "sources": sources,
        "confidence": min(0.99, 0.5 + 0.1 * top_result.score),
    }
