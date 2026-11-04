from __future__ import annotations

import re

from app.citation_validation import citation_matches
from app.medical_models import AnswerQuality, CitationBinding


_ANSWER_CITATION = re.compile(r"\[\d+(?:(?:\:|\#)[A-Za-z0-9_.-]+)?\]")
_SENTENCE_END = re.compile(r"[。！？!?]|\.(?!\d)")
_NOT_REPORTED = re.compile(
    r"(?:\b(?:not reported|not specified|not mentioned|not provided|not available|"
    r"does not (?:report|specify|mention|include|provide)|did not (?:report|specify|mention|"
    r"include|provide)|cannot (?:be )?determined|insufficient evidence|outside the abstract)\b|"
    r"未报告|未提及|未说明|未提供|摘要中没有|无法确定|无法判断|证据不足)",
    re.IGNORECASE,
)
_ABSTENTION_CONTEXT = re.compile(
    r"(?:\bfocused on\b|\bthe study (?:examined|evaluated)\b|"
    r"\bthe abstract (?:reports|describes)\b|\bprovided abstract\b|"
    r"研究重点|研究关注|摘要描述)",
    re.IGNORECASE,
)
_MEASUREMENT_NEGATION = re.compile(
    r"\b(?:no|did\s+not|does\s+not|was\s+not|were\s+not|not)\s+"
    r"(?:measure|measured|report|reported|evaluate|evaluated)\b",
    re.IGNORECASE,
)


def _statements(answer: str) -> list[str]:
    statements: list[str] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(answer):
        end = match.end()
        citation_end = end
        suffix = answer[end:]
        for citation_match in re.finditer(r"^\s*\[\d+(?:(?:\:|\#)[A-Za-z0-9_.-]+)?\]", suffix):
            citation_end = end + citation_match.end()
            suffix = answer[citation_end:]
        text = answer[cursor:citation_end].strip()
        if text:
            statements.append(text)
        cursor = citation_end
    tail = answer[cursor:].strip()
    if tail:
        statements.extend(line.strip() for line in tail.splitlines() if line.strip())
    return statements


def _is_claim(statement: str) -> bool:
    plain = _ANSWER_CITATION.sub("", statement)
    plain = re.sub(r"^[\s\-*•#\d.、]+", "", plain).strip(" ：:")
    if not plain or len(plain) < 8:
        return False
    headings = {"证据概述", "研究证据", "局限性", "总结", "回答"}
    lowered = plain.lower()
    support_scaffolding = (
        "this is supported by the following",
        "supported by the following abstract",
        "supporting evidence",
        "支持证据如下",
        "证据如下",
    )
    return plain not in headings and not any(item in lowered for item in support_scaffolding)


def is_unreported_answer(answer: str) -> bool:
    """Detect a refusal saying the requested detail is absent."""

    statements = [item for item in _statements(answer) if _is_claim(item)]
    if not statements:
        return False
    marked = [
        bool(_NOT_REPORTED.search(_ANSWER_CITATION.sub("", item)))
        for item in statements
    ]
    if any(marked):
        return all(
            marked[index] or bool(_ABSTENTION_CONTEXT.search(statement))
            for index, statement in enumerate(statements)
        )
    return False


def abstention_quality(answer: str) -> AnswerQuality:
    return AnswerQuality(
        valid=True,
        claim_count=0,
        cited_claim_count=0,
        citation_coverage=0.0,
        uncited_claims=[],
        warnings=["回答明确表示摘要未报告所请求信息。"],
    )


def assess_answer_quality(answer: str, bindings: list[CitationBinding]) -> AnswerQuality:
    statements = [item for item in _statements(answer) if _is_claim(item)]
    # A sentence explicitly limited to what the abstract reports is a permitted
    # missing-observation note. Stronger wording such as "was not measured"
    # remains an unsupported claim and is still checked normally.
    uncited = [
        item
        for item in statements
        if not citation_matches(item)
        and not _NOT_REPORTED.search(_ANSWER_CITATION.sub("", item))
    ]
    cited_count = len(statements) - len(uncited)
    coverage = cited_count / len(statements) if statements else 0.0
    warnings: list[str] = []

    if "evidence_id" in answer:
        warnings.append("回答包含未替换的 evidence_id 占位词。")
    if uncited:
        warnings.append("存在没有引用的实质性结论。")
    if any(
        _MEASUREMENT_NEGATION.search(_ANSWER_CITATION.sub("", statement))
        and not _NOT_REPORTED.search(_ANSWER_CITATION.sub("", statement))
        and not any(
            _MEASUREMENT_NEGATION.search(binding.exact_text)
            for binding in bindings
        )
        for statement in statements
    ):
        warnings.append("回答把摘要未报告的信息写成了未测量/未评估的事实。")
    uses_title_evidence = any(binding.source_section == "title" for binding in bindings)
    if uses_title_evidence:
        warnings.append("回答使用标题证据；具体医学结论应优先使用摘要证据。")

    return AnswerQuality(
        valid=(
            bool(statements)
            and not uncited
            and "evidence_id" not in answer
            and not uses_title_evidence
            and "回答把摘要未报告的信息写成了未测量/未评估的事实。" not in warnings
        ),
        claim_count=len(statements),
        cited_claim_count=cited_count,
        citation_coverage=round(coverage, 6),
        uncited_claims=uncited,
        warnings=warnings,
    )
