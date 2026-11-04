"""Multi-query retrieval with reciprocal-rank fusion and stable deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.cross_encoder_reranker import CrossEncoderReranker
from app.medical_models import MedicalDocument, RetrievalHit
from app.metadata_filter import RetrievalFilters, filter_hits
from app.query_rewriter import MedicalQueryRewriter


class RetrieverProtocol(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]: ...


def _normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


def document_identity(document: MedicalDocument) -> str:
    """Prefer PMID, then DOI, then a normalized title for deduplication."""

    pmid = document.metadata.get("pmid")
    if pmid:
        return f"pmid:{str(pmid).strip()}"
    if document.id.startswith("pubmed:"):
        return f"pmid:{document.id.split(':', 1)[1]}"
    if document.doi:
        return f"doi:{document.doi.strip().casefold()}"
    title = _normalized_title(document.title)
    return f"title:{title}" if title else f"id:{document.id}"


@dataclass(frozen=True)
class MultiQueryDiagnostics:
    queries: tuple[str, ...]
    raw_hit_count: int
    unique_hit_count: int
    eligible_hit_count: int
    excluded_by_year: int = 0
    excluded_by_abstract: int = 0
    excluded_by_article_type: int = 0

    @property
    def duplicates_removed(self) -> int:
        return self.raw_hit_count - self.unique_hit_count

    @property
    def filtered_out_count(self) -> int:
        return self.unique_hit_count - self.eligible_hit_count


@dataclass(frozen=True)
class MultiQueryResult:
    hits: tuple[RetrievalHit, ...]
    diagnostics: MultiQueryDiagnostics


class MultiQueryRetriever:
    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        cross_encoder_reranker: CrossEncoderReranker,
        *,
        query_rewriter: MedicalQueryRewriter | None = None,
        rrf_k: int = 60,
    ):
        if rrf_k < 1:
            raise ValueError("rrf_k 必须大于 0。")
        self.base_retriever = base_retriever
        self.cross_encoder_reranker = cross_encoder_reranker
        self.query_rewriter = query_rewriter or MedicalQueryRewriter()
        self.rrf_k = rrf_k

    def search_with_diagnostics(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_limit: int = 30,
        filters: RetrievalFilters | None = None,
    ) -> MultiQueryResult:
        if top_k < 1 or candidate_limit < 1:
            return MultiQueryResult(
                hits=(),
                diagnostics=MultiQueryDiagnostics((), 0, 0, 0),
            )

        rewrite = self.query_rewriter.rewrite(query)
        fused_scores: dict[str, float] = {}
        selected: dict[str, RetrievalHit] = {}
        raw_hit_count = 0

        for variant in rewrite.queries:
            hits = self.base_retriever.search(variant, top_k=candidate_limit)
            raw_hit_count += len(hits)
            for rank, hit in enumerate(hits, start=1):
                identity = document_identity(hit.document)
                fused_scores[identity] = fused_scores.get(identity, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )
                previous = selected.get(identity)
                if previous is None or (
                    not previous.document.abstract and hit.document.abstract
                ):
                    selected[identity] = hit

        fused = [
            hit.model_copy(
                update={
                    "score": fused_scores[identity],
                    "retrieval_method": "multi_query_rrf",
                }
            )
            for identity, hit in selected.items()
        ]
        fused.sort(key=lambda hit: (-hit.score, document_identity(hit.document)))
        filtered = filter_hits(fused, filters)
        reranked = self.cross_encoder_reranker.rerank_with_quality(
            rewrite.original_query,
            filtered.hits,
            top_k=top_k,
        )
        diagnostics = MultiQueryDiagnostics(
            queries=rewrite.queries,
            raw_hit_count=raw_hit_count,
            unique_hit_count=len(fused),
            eligible_hit_count=filtered.diagnostics.output_count,
            excluded_by_year=filtered.diagnostics.excluded_by_year,
            excluded_by_abstract=filtered.diagnostics.excluded_by_abstract,
            excluded_by_article_type=(
                filtered.diagnostics.excluded_by_article_type
            ),
        )
        return MultiQueryResult(hits=tuple(reranked), diagnostics=diagnostics)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_limit: int = 30,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        return list(
            self.search_with_diagnostics(
                query,
                top_k=top_k,
                candidate_limit=candidate_limit,
                filters=filters,
            ).hits
        )


__all__ = [
    "MultiQueryDiagnostics",
    "MultiQueryResult",
    "MultiQueryRetriever",
    "document_identity",
]
