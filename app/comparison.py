from __future__ import annotations

import re

from app.evidence_extractor import EvidenceExtractor, EvidenceMode
from app.medical_models import (
    ArticleMarkers,
    CompareResponse,
    ComparisonRow,
    MedicalDocument,
)
from app.search import SearchHit, medical_document


_CORRECTION_PATTERN = re.compile(
    r"\b(?:correction|erratum|corrigendum|retraction|retracted|expression of concern)\b",
    flags=re.IGNORECASE,
)
_REVIEW_PATTERN = re.compile(
    r"\b(?:review|systematic review|narrative review|scoping review|meta-analysis|meta analysis)\b",
    flags=re.IGNORECASE,
)


def classify_article(document: MedicalDocument) -> ArticleMarkers:
    """Add transparent quality markers without changing retrieval scores."""

    title = document.title or ""
    flags: list[str] = []
    if _CORRECTION_PATTERN.search(title):
        flags.append("correction")
    if _REVIEW_PATTERN.search(title):
        flags.append("review")
    abstract_available = bool(document.abstract.strip())
    if not abstract_available:
        flags.append("abstract_missing")

    return ArticleMarkers(flags=flags, abstract_available=abstract_available)


def build_comparison(
    question: str,
    results: list[SearchHit],
    extractor: EvidenceExtractor,
    mode: EvidenceMode,
) -> CompareResponse:
    rows: list[ComparisonRow] = []
    for rank, hit in enumerate(results, start=1):
        document = medical_document(hit)
        evidence = extractor.extract(document, mode=mode)
        rows.append(
            ComparisonRow(
                rank=rank,
                pmid=evidence.pmid,
                title=document.title,
                score=hit.score,
                source_url=evidence.source_url,
                markers=classify_article(document),
                evidence=evidence,
            )
        )

    return CompareResponse(
        question=question,
        extraction_mode=mode,
        compared_count=len(rows),
        comparison_table=rows,
        summary=_summary(rows, mode),
    )


def _summary(rows: list[ComparisonRow], mode: EvidenceMode) -> str:
    if not rows:
        return "没有检索到可以比较的 PubMed 文献。"

    correction_count = sum("correction" in row.markers.flags for row in rows)
    review_count = sum("review" in row.markers.flags for row in rows)
    missing_count = sum("abstract_missing" in row.markers.flags for row in rows)
    parts = [f"本次比较包含 {len(rows)} 篇 PubMed 文献。"]
    if correction_count:
        parts.append(f"其中 {correction_count} 篇被标题启发式标记为校正或撤稿相关文章。")
    if review_count:
        parts.append(f"其中 {review_count} 篇被标题启发式标记为综述类文章。")
    if missing_count:
        parts.append(f"其中 {missing_count} 篇没有摘要，不能进行摘要级字段比较。")
    if mode == "metadata_only":
        parts.append(
            "当前为 metadata_only 模式；未对摘要做语义字段提取，未明确提及的字段按 null 返回。"
        )
    else:
        parts.append("结构化字段仅保留经过原文片段校验的摘要级证据。")
    return "".join(parts)


__all__ = ["build_comparison", "classify_article"]
