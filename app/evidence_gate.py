from __future__ import annotations

from dataclasses import dataclass

from app.evidence_segmenter import segment_abstract
from app.medical_models import (
    EvidenceSpan,
    EvidenceValue,
    RetrievalHit,
    StructuredEvidence,
)


@dataclass(frozen=True)
class EvidenceGateDecision:
    allowed: bool
    reason: str | None
    message: str
    records: list[StructuredEvidence]


def _pmid(hit: RetrievalHit) -> str:
    document = hit.document
    metadata_pmid = document.metadata.get("pmid")
    if metadata_pmid:
        return str(metadata_pmid)
    if document.id.startswith("pubmed:"):
        return document.id.removeprefix("pubmed:")
    return document.id


def _source_url(hit: RetrievalHit, pmid: str) -> str:
    if hit.document.url:
        return hit.document.url
    if hit.document.id.startswith("pubmed:") or hit.document.metadata.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return f"urn:document:{hit.document.id}"


def build_answer_evidence_records(hits: list[RetrievalHit]) -> list[StructuredEvidence]:
    """Create verbatim title/abstract spans for answer generation and tracing."""

    records: list[StructuredEvidence] = []
    for hit in hits:
        document = hit.document
        pmid = _pmid(hit)
        source_url = _source_url(hit, pmid)
        spans = [
            EvidenceSpan(
                evidence_id="title-1",
                pmid=pmid,
                source_section="title",
                exact_text=document.title,
                source_url=source_url,
            )
        ]
        status = "abstract_unavailable"
        if document.abstract.strip():
            status = "not_reported_in_abstract"
            for index, segment in enumerate(segment_abstract(document.abstract), start=1):
                spans.append(
                    EvidenceSpan(
                        evidence_id=f"abstract-{index}",
                        pmid=pmid,
                        source_section="abstract",
                        exact_text=segment.text,
                        source_url=source_url,
                        start=segment.start,
                        end=segment.end,
                    )
                )
        empty = EvidenceValue(value=None, status=status, evidence_ids=[])
        records.append(
            StructuredEvidence(
                pmid=pmid,
                source_url=source_url,
                title=document.title,
                title_evidence_ids=["title-1"],
                research_object=empty,
                disease=empty,
                method=empty,
                dataset=empty,
                main_findings=empty,
                limitations=empty,
                evidence_spans=spans,
            )
        )
    return records


def evaluate_evidence(
    hits: list[RetrievalHit],
    *,
    minimum_abstract_characters: int = 10,
) -> EvidenceGateDecision:
    if not hits:
        return EvidenceGateDecision(
            allowed=False,
            reason="no_retrieval_results",
            message="没有检索到可用于回答该问题的资料。",
            records=[],
        )

    records = build_answer_evidence_records(hits)
    abstract_characters = sum(len(hit.document.abstract.strip()) for hit in hits)
    if abstract_characters == 0:
        return EvidenceGateDecision(
            allowed=False,
            reason="abstract_unavailable",
            message="检索结果缺少摘要，当前证据不足，无法生成可靠回答。",
            records=records,
        )
    if abstract_characters < minimum_abstract_characters:
        return EvidenceGateDecision(
            allowed=False,
            reason="insufficient_evidence",
            message="检索摘要信息过少，当前证据不足，无法生成可靠回答。",
            records=records,
        )

    return EvidenceGateDecision(
        allowed=True,
        reason=None,
        message="",
        records=records,
    )
