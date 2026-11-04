from __future__ import annotations

from dataclasses import asdict
import logging
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from app.search import (
    ExternalServiceError,
    SearchHit,
    SearchService,
    VectorSearchError,
    medical_document,
)
from app.evidence_extractor import EvidenceExtractionError, EvidenceExtractor
from app.comparison import build_comparison
from app.agent_state import AgentRequest, AgentResult
from app.medical_models import (
    AnswerQuality,
    CitationBinding,
    CompareResponse,
    EvidenceResponse,
    LLMUsage,
    RetrievalHit,
)
from app.metadata_filter import RetrievalFilters
from app.request_context import request_id_context


app = FastAPI(title="Search Answer API", version="0.2.6")
logger = logging.getLogger(__name__)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach a request ID and log total API latency for every request."""

    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request_id = request_id[:128]
    started = perf_counter()
    context_token = request_id_context.set(request_id)
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http_request_failed",
                extra={
                    "request_id": request_id,
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
            raise

        duration_ms = round((perf_counter() - started) * 1000, 3)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request_completed",
            extra={
                "request_id": request_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
    finally:
        request_id_context.reset(context_token)

SearchSource = Literal[
    "local",
    "pubmed",
    "vector",
    "hybrid",
    "vector_persisted",
    "hybrid_persisted",
    "hybrid_persisted_reranked",
    "hybrid_persisted_cross_reranked",
    "hybrid_persisted_cross_quality_reranked",
    "advanced_persisted",
    "vector_pg",
    "hybrid_pg",
    "pubmed_hybrid",
    "all",
]


class RetrievalFiltersRequest(BaseModel):
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    require_abstract: bool = False
    exclude_article_types: list[
        Literal["editorial", "correction", "review"]
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_order(self):
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from 不能大于 year_to。")
        return self


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)
    source: SearchSource = "local"
    candidate_limit: int = Field(default=30, ge=1, le=50)
    filters: RetrievalFiltersRequest | None = None


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=10)
    source: SearchSource = "local"
    mode: Literal["extractive", "llm"] = "extractive"
    candidate_limit: int = Field(default=30, ge=1, le=50)


class AgentAnswerRequest(BaseModel):
    """Public request contract for the LangGraph-backed agent endpoint."""

    question: str = Field(min_length=1, max_length=300)
    retrieval_method: SearchSource = "pubmed_hybrid"
    top_k: int = Field(default=3, ge=1, le=10)
    candidate_limit: int = Field(default=30, ge=1, le=50)
    metadata: dict[str, object] = Field(default_factory=dict)


class EvidenceRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=1, ge=1, le=3)
    candidate_limit: int = Field(default=30, ge=20, le=50)
    mode: Literal["metadata_only", "strict_llm"] = "metadata_only"


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=2, le=5)
    candidate_limit: int = Field(default=30, ge=20, le=50)
    mode: Literal["metadata_only", "strict_llm"] = "metadata_only"


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
    citation_bindings: list[CitationBinding] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    cache_hit: bool = False
    quality: AnswerQuality | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


search_service = SearchService()


def get_search_service() -> SearchService:
    return search_service


def get_llm_provider():
    from app.grounded_llm import OpenAICompatibleProvider

    return OpenAICompatibleProvider()


def _serialize_hit(hit: SearchHit) -> SearchResult:
    return SearchResult(**asdict(hit))


class _AgentSearchRetriever:
    """Adapt the existing SearchService to the Stage 7 agent contract."""

    def __init__(self, service: SearchService, source: SearchSource):
        self.service = service
        self.source = source

    def search(
        self,
        query: str,
        top_k: int,
        candidate_limit: int = 30,
    ) -> list[RetrievalHit]:
        results, _ = self.service.search(
            query,
            top_k,
            self.source,
            candidate_limit=candidate_limit,
        )
        return [
            RetrievalHit(
                document=medical_document(item),
                score=item.score,
                retrieval_method=self.source,
            )
            for item in results
        ]


def _run_search(
    service: SearchService,
    query: str,
    limit: int,
    source: SearchSource,
    candidate_limit: int = 30,
    filters: RetrievalFiltersRequest | None = None,
):
    try:
        retrieval_filters = (
            RetrievalFilters(
                year_from=filters.year_from,
                year_to=filters.year_to,
                require_abstract=filters.require_abstract,
                exclude_article_types=tuple(filters.exclude_article_types),
            )
            if filters is not None
            else None
        )
        return service.search(
            query,
            limit,
            source,
            candidate_limit=candidate_limit,
            filters=retrieval_filters,
        )
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
    results, total = _run_search(
        service,
        request.query,
        request.limit,
        request.source,
        request.candidate_limit,
        request.filters,
    )
    return {
        "results": [_serialize_hit(item) for item in results],
        "total": total,
        "source": request.source,
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    service: SearchService = Depends(get_search_service),
    provider=Depends(get_llm_provider),
):
    results, _ = _run_search(
        service,
        request.question,
        request.top_k,
        request.source,
        request.candidate_limit,
    )
    sources = [_serialize_hit(item) for item in results]

    if not results:
        return {
            "answer": "没有检索到相关资料。",
            "sources": [],
            "confidence": 0.0,
            "abstained": True,
            "abstention_reason": "no_retrieval_results",
        }

    if request.mode == "llm":
        from app.evidence_gate import evaluate_evidence
        from app.grounded_llm import GroundedAnswerer, LLMProviderError
        from app.medical_models import RetrievalHit
        from app.search import medical_document
        hits = [RetrievalHit(document=medical_document(item), score=item.score,
                             retrieval_method=request.source) for item in results]
        decision = evaluate_evidence(hits)
        if not decision.allowed:
            return {
                "answer": decision.message,
                "sources": sources,
                "confidence": 0.0,
                "abstained": True,
                "abstention_reason": decision.reason,
            }
        try:
            generated = GroundedAnswerer(
                provider,
                max_citation_repairs=1,
            ).answer(
                request.question,
                hits,
                evidence_records=decision.records,
            )
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "answer": generated.answer,
            "sources": sources,
            "confidence": 0.0,
            "citations": generated.citations,
            "citation_bindings": generated.citation_bindings,
            "abstained": generated.abstained,
            "abstention_reason": generated.abstention_reason,
            "model": generated.model,
            "prompt_version": generated.prompt_version,
            "usage": generated.usage,
            "estimated_cost_usd": generated.estimated_cost_usd,
            "latency_ms": generated.latency_ms,
            "cache_hit": generated.cache_hit,
            "quality": generated.quality,
        }

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


@app.post("/agent-answer", response_model=AgentResult)
def agent_answer(
    request: AgentAnswerRequest,
    service: SearchService = Depends(get_search_service),
    provider=Depends(get_llm_provider),
):
    """Run the bounded LangGraph workflow without changing the legacy /answer."""

    from app.agent_graph import run_agent_graph
    from app.grounded_llm import GroundedAnswerer

    agent_request = AgentRequest(
        question=request.question,
        retrieval_method=request.retrieval_method,
        top_k=request.top_k,
        candidate_limit=request.candidate_limit,
        metadata=request.metadata,
    )
    retriever = _AgentSearchRetriever(service, request.retrieval_method)
    # Citation repair is bounded by the graph itself, so the answerer performs
    # its own provider call once per graph answer node.
    answerer = GroundedAnswerer(provider, max_citation_repairs=0)
    state = run_agent_graph(agent_request, retriever, answerer)
    return state.to_result()


@app.post("/evidence", response_model=EvidenceResponse)
def evidence(
    request: EvidenceRequest,
    service: SearchService = Depends(get_search_service),
):
    """Retrieve PubMed records and return traceable title/abstract evidence."""

    results, _ = _run_search(
        service,
        request.question,
        request.top_k,
        "pubmed_hybrid",
        request.candidate_limit,
    )

    if request.mode == "strict_llm":
        from app.grounded_llm import OpenAICompatibleProvider

        extractor = EvidenceExtractor(provider=OpenAICompatibleProvider())
    else:
        extractor = EvidenceExtractor()

    try:
        records = [
            extractor.extract(
                medical_document(item),
                mode=request.mode,
            )
            for item in results
        ]
    except EvidenceExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "question": request.question,
        "retrieval_method": "pubmed_hybrid",
        "extraction_mode": request.mode,
        "records": records,
    }


@app.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    service: SearchService = Depends(get_search_service),
):
    """Compare several PubMed records while preserving article-level markers."""

    results, _ = _run_search(
        service,
        request.question,
        request.top_k,
        "pubmed_hybrid",
        request.candidate_limit,
    )

    if request.mode == "strict_llm":
        from app.grounded_llm import OpenAICompatibleProvider

        extractor = EvidenceExtractor(provider=OpenAICompatibleProvider())
    else:
        extractor = EvidenceExtractor()

    try:
        return build_comparison(
            question=request.question,
            results=results,
            extractor=extractor,
            mode=request.mode,
        )
    except EvidenceExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
