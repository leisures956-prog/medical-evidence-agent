"""Explicit metadata filters for the advanced persisted retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.medical_models import RetrievalHit
from app.quality_reranker import quality_flags


ArticleType = Literal["editorial", "correction", "review"]
ALLOWED_ARTICLE_TYPES = frozenset({"editorial", "correction", "review"})


@dataclass(frozen=True)
class RetrievalFilters:
    year_from: int | None = None
    year_to: int | None = None
    require_abstract: bool = False
    exclude_article_types: tuple[ArticleType, ...] = ()

    def __post_init__(self):
        for value in (self.year_from, self.year_to):
            if value is not None and not 1900 <= value <= 2100:
                raise ValueError("年份必须在 1900 到 2100 之间。")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from 不能大于 year_to。")
        invalid = set(self.exclude_article_types) - ALLOWED_ARTICLE_TYPES
        if invalid:
            raise ValueError(f"不支持的文献类型：{sorted(invalid)}")

    @property
    def is_active(self) -> bool:
        return any(
            (
                self.year_from is not None,
                self.year_to is not None,
                self.require_abstract,
                bool(self.exclude_article_types),
            )
        )


@dataclass(frozen=True)
class FilterDiagnostics:
    input_count: int
    output_count: int
    excluded_by_year: int = 0
    excluded_by_abstract: int = 0
    excluded_by_article_type: int = 0

    @property
    def filtered_out_count(self) -> int:
        return self.input_count - self.output_count


@dataclass(frozen=True)
class FilterResult:
    hits: tuple[RetrievalHit, ...]
    diagnostics: FilterDiagnostics


def filter_hits(
    hits: list[RetrievalHit] | tuple[RetrievalHit, ...],
    filters: RetrievalFilters | None,
) -> FilterResult:
    candidates = list(hits)
    if filters is None or not filters.is_active:
        return FilterResult(
            hits=tuple(candidates),
            diagnostics=FilterDiagnostics(
                input_count=len(candidates),
                output_count=len(candidates),
            ),
        )

    kept: list[RetrievalHit] = []
    excluded_by_year = 0
    excluded_by_abstract = 0
    excluded_by_article_type = 0
    excluded_types = set(filters.exclude_article_types)

    for hit in candidates:
        document = hit.document
        if filters.year_from is not None or filters.year_to is not None:
            if document.year is None:
                excluded_by_year += 1
                continue
            if filters.year_from is not None and document.year < filters.year_from:
                excluded_by_year += 1
                continue
            if filters.year_to is not None and document.year > filters.year_to:
                excluded_by_year += 1
                continue

        if filters.require_abstract and not document.abstract.strip():
            excluded_by_abstract += 1
            continue

        if excluded_types.intersection(quality_flags(document)):
            excluded_by_article_type += 1
            continue

        kept.append(hit)

    return FilterResult(
        hits=tuple(kept),
        diagnostics=FilterDiagnostics(
            input_count=len(candidates),
            output_count=len(kept),
            excluded_by_year=excluded_by_year,
            excluded_by_abstract=excluded_by_abstract,
            excluded_by_article_type=excluded_by_article_type,
        ),
    )


__all__ = [
    "ALLOWED_ARTICLE_TYPES",
    "ArticleType",
    "FilterDiagnostics",
    "FilterResult",
    "RetrievalFilters",
    "filter_hits",
]
