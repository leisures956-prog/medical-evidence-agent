from __future__ import annotations

from app.medical_models import MedicalDocument, StructuredEvidence


class EvidenceValidationError(ValueError):
    """Raised when extracted evidence cannot be traced to one document."""


def _document_pmid(document: MedicalDocument) -> str:
    metadata_pmid = document.metadata.get("pmid")
    if metadata_pmid is not None:
        return str(metadata_pmid)
    if document.id.startswith("pubmed:"):
        return document.id.removeprefix("pubmed:")
    raise EvidenceValidationError(f"文献 {document.id} 缺少 PMID，无法建立可追溯证据。")


def _document_url(document: MedicalDocument, pmid: str) -> str:
    return document.url or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def validate_evidence(
    document: MedicalDocument,
    evidence: StructuredEvidence,
) -> StructuredEvidence:
    """Check that every quoted span is an exact substring of title/abstract.

    This validator intentionally checks traceability rather than medical truth. A
    model may paraphrase a supported finding, but it must point to a verbatim span
    from the same PubMed record.
    """

    expected_pmid = _document_pmid(document)
    expected_url = _document_url(document, expected_pmid)
    if evidence.pmid != expected_pmid:
        raise EvidenceValidationError(
            f"PMID 不匹配：文献是 {expected_pmid}，证据写成 {evidence.pmid}。"
        )
    if evidence.source_url != expected_url:
        raise EvidenceValidationError("source_url 与原文链接不匹配。")
    if evidence.title.strip() != document.title.strip():
        raise EvidenceValidationError("证据中的 title 与 PubMed 标题不匹配。")

    spans_by_id = {}
    for span in evidence.evidence_spans:
        if span.pmid != expected_pmid:
            raise EvidenceValidationError(
                f"证据片段 {span.evidence_id} 的 PMID 与文献不匹配。"
            )
        if span.source_url != expected_url:
            raise EvidenceValidationError(
                f"证据片段 {span.evidence_id} 的 source_url 与文献不匹配。"
            )
        if span.evidence_id in spans_by_id:
            raise EvidenceValidationError(f"evidence_id 重复：{span.evidence_id}")

        source_text = document.title if span.source_section == "title" else document.abstract
        if not source_text:
            raise EvidenceValidationError(
                f"证据片段 {span.evidence_id} 引用了空的 {span.source_section}。"
            )
        if span.exact_text not in source_text:
            raise EvidenceValidationError(
                f"证据片段 {span.evidence_id} 不是标题或摘要中的原文片段。"
            )
        if span.start is not None and span.end is not None:
            if span.end > len(source_text) or source_text[span.start : span.end] != span.exact_text:
                raise EvidenceValidationError(
                    f"证据片段 {span.evidence_id} 的字符位置与原文不匹配。"
                )
        spans_by_id[span.evidence_id] = span

    if not evidence.title_evidence_ids:
        raise EvidenceValidationError("必须保留至少一个标题证据片段。")

    for evidence_id in evidence.title_evidence_ids:
        span = spans_by_id.get(evidence_id)
        if span is None or span.source_section != "title":
            raise EvidenceValidationError(
                f"title_evidence_ids {evidence_id} 没有指向标题片段。"
            )
        if span.exact_text != document.title:
            raise EvidenceValidationError(
                f"标题证据片段 {evidence_id} 必须保留完整标题原文。"
            )

    for field_name in (
        "research_object",
        "disease",
        "method",
        "dataset",
        "main_findings",
        "limitations",
    ):
        field = getattr(evidence, field_name)
        for evidence_id in field.evidence_ids:
            if evidence_id not in spans_by_id:
                raise EvidenceValidationError(
                    f"字段 {field_name} 引用了不存在的证据片段 {evidence_id}。"
                )

    return evidence
