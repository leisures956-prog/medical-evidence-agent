from __future__ import annotations

import re

from app.medical_models import CitationBinding, CitationValidation, StructuredEvidence


# Both forms are supported:
#   [1]             source-level citation (unambiguous for one evidence span)
#   [1:finding-1]  explicit evidence-span citation
_CITATION_PATTERN = re.compile(
    r"\[(?P<number>\d+)(?:(?:\:|\#)(?P<evidence_id>[A-Za-z0-9_.-]+))?\]"
)


def citation_matches(answer: str) -> list[tuple[int, str | None]]:
    return [
        (int(match.group("number")), match.group("evidence_id"))
        for match in _CITATION_PATTERN.finditer(answer)
    ]


def citation_numbers(answer: str) -> list[int]:
    return sorted({number for number, _ in citation_matches(answer)})


def validate_citations(
    answer: str,
    evidence_records: list[StructuredEvidence],
    *,
    require_evidence: bool = True,
) -> CitationValidation:
    """Validate citation indexes and bind each citation to an evidence span.

    A plain ``[n]`` is accepted when record ``n`` has exactly one evidence span.
    When a record has multiple spans, the answer must use the explicit form
    ``[n:evidence_id]`` so the cited conclusion remains traceable.
    """

    matches = citation_matches(answer)
    citations = sorted({number for number, _ in matches})
    errors: list[str] = []
    bindings: list[CitationBinding] = []
    seen_bindings: set[tuple[int, str]] = set()

    if not matches:
        errors.append("回答没有引用来源。")

    for number, evidence_id in matches:
        if number < 1 or number > len(evidence_records):
            errors.append(f"存在无效引用：[{number}]。")
            continue

        record = evidence_records[number - 1]
        spans_by_id = {span.evidence_id: span for span in record.evidence_spans}
        span = None
        if evidence_id is not None:
            span = spans_by_id.get(evidence_id)
            if span is None:
                errors.append(
                    f"引用 [{number}:{evidence_id}] 指向不存在的 evidence_id。"
                )
                continue
        elif require_evidence:
            if len(record.evidence_spans) != 1:
                errors.append(
                    f"引用 [{number}] 无法唯一定位证据片段，请使用 ["
                    f"{number}:evidence_id]。"
                )
                continue
            span = record.evidence_spans[0]

        if span is None:
            continue
        binding_key = (number, span.evidence_id)
        if binding_key in seen_bindings:
            continue
        seen_bindings.add(binding_key)
        bindings.append(
            CitationBinding(
                citation_number=number,
                pmid=record.pmid,
                evidence_id=span.evidence_id,
                source_section=span.source_section,
                exact_text=span.exact_text,
                source_url=span.source_url,
            )
        )

    return CitationValidation(
        valid=not errors,
        citations=citations,
        bindings=bindings,
        errors=errors,
    )
