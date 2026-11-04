"""Evaluation utilities for grounded answers and citation support.

This module deliberately keeps semantic checking conservative.  It checks whether
the text of a cited abstract span overlaps with the claim and whether the claim
uses a stronger causal/effect verb than the cited evidence.  It is not a medical
fact checker and must be reported as a heuristic evaluation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.citation_validation import citation_matches
from app.medical_models import CitationBinding
from app.text_normalization import normalize_for_prompt


AnswerBehavior = Literal["answer", "abstain"]
AnswerScenario = Literal["answerable", "no_answer", "insufficient", "conflicting"]


class ExpectedClaim(BaseModel):
    """A manually specified claim that an answer should cover."""

    id: str = Field(min_length=1)
    support_terms: list[str] = Field(min_length=1)
    expected_source_section: Literal["abstract", "title"] = "abstract"
    allowed_pmids: list[str] = Field(default_factory=list)
    note: str | None = None
    exact_text: str | None = None


class AnswerEvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    scenario: AnswerScenario
    expected_behavior: AnswerBehavior
    expected_abstention_reasons: list[str] = Field(default_factory=list)
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    notes: str | None = None
    context_pmids: list[str] = Field(default_factory=list)
    reference_spans: list[dict[str, str]] = Field(default_factory=list)
    corpus_sha256: str | None = None


class AnswerObservation(BaseModel):
    """The small, stable subset accepted from an /answer JSON response."""

    answer: str = ""
    abstained: bool = False
    abstention_reason: str | None = None
    citations: list[int] = Field(default_factory=list)
    citation_bindings: list[CitationBinding] = Field(default_factory=list)
    quality: dict[str, Any] | None = None
    error: str | None = None


class AnswerCaseResult(BaseModel):
    id: str
    question: str
    scenario: AnswerScenario
    expected_behavior: AnswerBehavior
    actual_abstained: bool
    abstention_correct: bool
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    citation_coverage: float = Field(ge=0, le=1)
    answer_faithfulness: float = Field(ge=0, le=1)
    unsupported_claim_count: int = Field(ge=0)
    expected_claims_found: int = Field(ge=0)
    expected_claim_count: int = Field(ge=0)
    citation_traceable: bool
    scope_warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    semantic_support_verified: bool | None = None


class AnswerEvaluationSummary(BaseModel):
    evaluation_version: str = "answer-eval-v1-stage6.1"
    annotation_status: str = "assistant_draft_not_human_gold"
    semantic_metric_type: str = "lexical_proxy_not_entailment"
    total_cases: int = Field(ge=0)
    answerable_cases: int = Field(ge=0)
    abstention_cases: int = Field(ge=0)
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    citation_coverage: float = Field(ge=0, le=1)
    answer_faithfulness: float = Field(ge=0, le=1)
    abstention_accuracy: float = Field(ge=0, le=1)
    average_unsupported_claim_count: float = Field(ge=0)
    citation_traceability_rate: float = Field(ge=0, le=1)
    generation_error_cases: int = Field(ge=0)
    missing_observation_cases: int = Field(ge=0)
    cases: list[AnswerCaseResult] = Field(default_factory=list)
    metric_note: str


_CITATION = re.compile(r"\[\d+(?:(?:\:|\#)[A-Za-z0-9_.-]+)?\]")
_SENTENCE_END = re.compile(r"[。！？!?]|\.(?!\d)")
_MISSING_OBSERVATION = re.compile(
    r"(?:the\s+abstract|abstract|摘要).{0,80}"
    r"(?:does\s+not\s+report|did\s+not\s+report|not\s+reported|not\s+specified|"
    r"not\s+mentioned|not\s+provided|未报告|未提及|未说明|未提供)",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "were", "was", "are",
    "for", "into", "than", "study", "abstract", "reported", "result",
    "results", "evidence", "research", "patients", "patient", "may", "can",
    "有", "研究", "结果", "显示", "表明", "提示", "患者",
}
_ASSOCIATION_TERMS = {
    "associated", "association", "correlated", "correlate", "observational",
    "suggest", "suggests", "prognostic", "risk", "linked", "关系", "相关",
}
_EFFECT_TERMS = {
    "improve", "improved", "improves", "improvement", "benefit", "beneficial",
    "effective", "efficacy", "caused", "cause", "causes", "prevent", "prevents",
    "diagnose", "diagnostic", "治疗获益", "改善", "导致", "预防", "诊断",
}
_NEGATION_MEASURED = re.compile(
    r"(?:"
    r"\bno\s+(?:auc(?:\s+value)?|value|values|measurement|measurements)\b"
    r"|\b(?:auc|value|values|measurement|measurements)\b[^.!?]{0,40}"
    r"\b(?:not|never)\s+(?:reported|measured|provided|available)\b"
    r"|\b(?:not|never)\s+(?:reported|measured|provided|available)\b[^.!?]{0,40}"
    r"\b(?:auc|value|values|measurement|measurements)\b"
    r")",
    re.IGNORECASE,
)
_TERM_SYNONYMS = {
    "canadian": "canada",
    "children": "child",
    "patients": "patient",
    "studies": "study",
    "observational": "observe",
    "retrospective": "retrospect",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(normalize_for_prompt(text))
        if token.lower() not in _STOPWORDS and len(token) > 1
    }


def _term_tokens(term: str) -> set[str]:
    return {
        _TERM_SYNONYMS.get(token, token)
        for token in _tokens(term)
    }


def _normalized_terms(text: str) -> set[str]:
    return {
        _TERM_SYNONYMS.get(token, token)
        for token in _tokens(text)
    }


def _contains_term(text: str, term: str) -> bool:
    normalized = normalize_for_prompt(text).lower()
    return (
        term.lower() in normalized
        or _term_tokens(term).issubset(_normalized_terms(text))
    )


def _claim_sentences(answer: str) -> list[tuple[str, list[tuple[int, str | None]]]]:
    """Split an answer into claim-like sentences and their inline citations."""

    rows: list[tuple[str, list[tuple[int, str | None]]]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(answer):
        end = match.end()
        suffix = answer[end:]
        citation_end = end
        trailing = re.match(r"^(?:\s*" + _CITATION.pattern + r")+", suffix)
        if trailing:
            citation_end = end + trailing.end()
        text = answer[cursor:citation_end].strip()
        if text:
            rows.append((text, citation_matches(text)))
        cursor = citation_end
    tail = answer[cursor:].strip()
    if tail:
        for line in tail.splitlines():
            line = line.strip()
            if line:
                rows.append((line, citation_matches(line)))
    return rows


def _is_claim(text: str) -> bool:
    plain = _CITATION.sub("", text)
    plain = re.sub(r"^[\s\-*•#\d.、]+", "", plain).strip(" ：:")
    if len(plain) < 8:
        return False
    return plain.lower() not in {"summary", "answer", "总结", "回答"}


def _is_missing_observation(text: str) -> bool:
    """Identify a bounded statement about what the abstract does not report."""

    plain = _CITATION.sub("", text)
    return bool(_MISSING_OBSERVATION.search(plain))


def _binding_map(
    bindings: list[CitationBinding],
) -> tuple[dict[tuple[int, str], CitationBinding], dict[int, list[CitationBinding]]]:
    by_key: dict[tuple[int, str], CitationBinding] = {}
    by_number: dict[int, list[CitationBinding]] = {}
    for binding in bindings:
        by_key[(binding.citation_number, binding.evidence_id)] = binding
        by_number.setdefault(binding.citation_number, []).append(binding)
    return by_key, by_number


def _bindings_for_citations(
    citations: list[tuple[int, str | None]],
    by_key: dict[tuple[int, str], CitationBinding],
    by_number: dict[int, list[CitationBinding]],
) -> list[CitationBinding]:
    selected: list[CitationBinding] = []
    seen: set[tuple[int, str]] = set()
    for number, evidence_id in citations:
        candidates = (
            [by_key[(number, evidence_id)]]
            if evidence_id is not None and (number, evidence_id) in by_key
            else (by_number.get(number, []) if evidence_id is None else [])
        )
        for binding in candidates:
            key = (binding.citation_number, binding.evidence_id)
            if key not in seen:
                selected.append(binding)
                seen.add(key)
    return selected


def _overlap_score(claim: str, evidence: str) -> float:
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _supported_by_binding(claim: str, binding: CitationBinding) -> bool:
    if binding.source_section != "abstract":
        return False
    plain_claim = _CITATION.sub("", claim)
    score = _overlap_score(plain_claim, binding.exact_text)
    overlap = _tokens(plain_claim) & _tokens(binding.exact_text)
    claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", plain_claim))
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", binding.exact_text))
    if claim_numbers and not claim_numbers.issubset(evidence_numbers):
        return False
    if _NEGATION_MEASURED.search(plain_claim):
        claim_lower = plain_claim.lower()
        evidence_lower = binding.exact_text.lower()
        if "auc" in claim_lower and re.search(r"\d+(?:\.\d+)?", binding.exact_text):
            return False
        if "value" in claim_lower and re.search(r"\d+(?:\.\d+)?", binding.exact_text):
            return False
    return len(overlap) >= 2 or score >= 0.35


def _scope_warning(claim: str, bindings: list[CitationBinding]) -> str | None:
    claim_tokens = _tokens(claim)
    if not claim_tokens & _EFFECT_TERMS:
        return None
    for binding in bindings:
        evidence_tokens = _tokens(binding.exact_text)
        if evidence_tokens & _ASSOCIATION_TERMS and not evidence_tokens & _EFFECT_TERMS:
            return (
                "回答使用了因果/获益/诊断性措辞，但引用摘要仅使用关联、观察性或风险分层措辞。"
            )
    return None


def _expected_claim_found(
    expected: ExpectedClaim,
    rows: list[tuple[str, list[tuple[int, str | None]]]],
    by_key: dict[tuple[int, str], CitationBinding],
    by_number: dict[int, list[CitationBinding]],
) -> bool:
    expected_terms = [term.lower() for term in expected.support_terms]
    selected_rows = [
        (text, citations)
        for text, citations in rows
        if citations and any(_contains_term(text, term) for term in expected_terms)
    ]
    if not selected_rows:
        return False
    combined_answer = " ".join(text for text, _ in selected_rows)
    if not all(_contains_term(combined_answer, term) for term in expected_terms):
        return False

    bindings: list[CitationBinding] = []
    for _, citations in selected_rows:
        bindings.extend(_bindings_for_citations(citations, by_key, by_number))
    unique = {
        (binding.citation_number, binding.evidence_id): binding
        for binding in bindings
    }
    for binding in unique.values():
        if binding.source_section != expected.expected_source_section:
            continue
        if expected.allowed_pmids and binding.pmid not in expected.allowed_pmids:
            continue
        if expected.exact_text and binding.exact_text != expected.exact_text:
            continue
        evidence_text = " ".join(item.exact_text for item in unique.values())
        if all(_contains_term(evidence_text, term) for term in expected_terms):
            return True
    return False


def evaluate_answer_case(
    case: AnswerEvaluationCase,
    observation: AnswerObservation | dict[str, Any],
) -> AnswerCaseResult:
    """Evaluate one answer without making a medical truth claim."""

    observed = (
        observation
        if isinstance(observation, AnswerObservation)
        else AnswerObservation.model_validate(observation)
    )
    # An abstention message is not a medical claim.  Ignore its boilerplate
    # text so a correct refusal is not penalized for lacking citations.
    rows = (
        []
        if observed.abstained
        else [
            (text, citations)
            for text, citations in _claim_sentences(observed.answer)
            if _is_claim(text) and not _is_missing_observation(text)
        ]
    )
    by_key, by_number = _binding_map(observed.citation_bindings)
    cited_rows = [(text, citations) for text, citations in rows if citations]
    supported_rows = 0
    scope_warnings: list[str] = []
    traceable = bool(cited_rows)
    supported_citations = 0
    citation_count = 0
    for text, citations in cited_rows:
        bindings = _bindings_for_citations(citations, by_key, by_number)
        row_traceable = True
        for citation in citations:
            citation_count += 1
            matched = _bindings_for_citations([citation], by_key, by_number)
            if len(matched) != 1:
                traceable = False
                row_traceable = False
                continue
            binding = matched[0]
            if case.reference_spans and not any(
                binding.pmid == ref['pmid']
                and binding.source_section == ref['source_section']
                and binding.exact_text == ref['exact_text']
                for ref in case.reference_spans
            ):
                traceable = False
                row_traceable = False
                continue
            if _supported_by_binding(text, binding) and not _scope_warning(text, [binding]):
                supported_citations += 1
        if row_traceable and bindings and all(_supported_by_binding(text, binding) for binding in bindings):
            warning = _scope_warning(text, bindings)
            if warning:
                scope_warnings.append(warning)
            else:
                supported_rows += 1
        else:
            traceable = False if bindings == [] else traceable

    uncited_count = len(rows) - len(cited_rows)
    unsupported_count = uncited_count + (len(cited_rows) - supported_rows)
    citation_precision = (
        supported_citations / citation_count if citation_count else 0.0
    )
    citation_coverage = len(cited_rows) / len(rows) if rows else 0.0

    found = sum(
        _expected_claim_found(expected, rows, by_key, by_number)
        for expected in case.expected_claims
    )
    expected_count = len(case.expected_claims)
    if expected_count:
        citation_recall = found / expected_count
    else:
        citation_recall = 0.0

    if case.expected_behavior == "abstain":
        abstention_correct = observed.abstained and (
            not case.expected_abstention_reasons
            or observed.abstention_reason in case.expected_abstention_reasons
        )
    else:
        abstention_correct = not observed.abstained and bool(rows)
    if observed.error:
        abstention_correct = False

    faithfulness = (
        supported_rows / len(rows)
        if rows
        else (1.0 if observed.abstained and case.expected_behavior == "abstain" else 0.0)
    )
    errors: list[str] = [observed.error] if observed.error else []
    if case.expected_behavior == "answer" and observed.abstained:
        errors.append("期望回答，但实际拒答。")
    if case.expected_behavior == "abstain" and not observed.abstained:
        errors.append("期望拒答，但实际生成了回答。")
    if not traceable and cited_rows:
        errors.append("存在无法绑定到 PMID/evidence_id 的引用。")
    if scope_warnings:
        errors.extend(scope_warnings)

    return AnswerCaseResult(
        id=case.id,
        question=case.question,
        scenario=case.scenario,
        expected_behavior=case.expected_behavior,
        actual_abstained=observed.abstained,
        abstention_correct=abstention_correct,
        citation_precision=round(citation_precision, 6),
        citation_recall=round(citation_recall, 6),
        citation_coverage=round(citation_coverage, 6),
        answer_faithfulness=round(max(0.0, min(1.0, faithfulness)), 6),
        unsupported_claim_count=unsupported_count,
        expected_claims_found=found,
        expected_claim_count=expected_count,
        citation_traceable=traceable,
        scope_warnings=list(dict.fromkeys(scope_warnings)),
        errors=list(dict.fromkeys(errors)),
    )


def summarize_answer_evaluation(
    cases: list[AnswerEvaluationCase],
    observations: dict[str, AnswerObservation | dict[str, Any]],
) -> AnswerEvaluationSummary:
    results: list[AnswerCaseResult] = []
    for case in cases:
        observation = observations.get(case.id, AnswerObservation(error='missing_observation'))
        results.append(evaluate_answer_case(case, observation))
    total = len(results)
    denominator = total or 1
    answer_results = [item for item in results if item.expected_behavior == 'answer']
    answer_denominator = len(answer_results) or 1
    return AnswerEvaluationSummary(
        total_cases=total,
        answerable_cases=sum(case.expected_behavior == "answer" for case in cases),
        abstention_cases=sum(case.expected_behavior == "abstain" for case in cases),
        citation_precision=round(sum(item.citation_precision for item in answer_results) / answer_denominator, 6),
        citation_recall=round(sum(item.citation_recall for item in answer_results) / answer_denominator, 6),
        citation_coverage=round(sum(item.citation_coverage for item in answer_results) / answer_denominator, 6),
        answer_faithfulness=round(sum(item.answer_faithfulness for item in answer_results) / answer_denominator, 6),
        abstention_accuracy=round(sum(item.abstention_correct for item in results) / denominator, 6),
        average_unsupported_claim_count=round(
            sum(item.unsupported_claim_count for item in results) / denominator,
            6,
        ),
        citation_traceability_rate=round(
            sum(item.citation_traceable for item in answer_results) / answer_denominator,
            6,
        ),
        generation_error_cases=sum(
            bool(item.errors and any(
                not error.startswith("期望")
                and not error.startswith("存在无法")
                for error in item.errors
            ))
            for item in results
        ),
        missing_observation_cases=sum(
            "missing_observation" in item.errors
            for item in results
        ),
        cases=results,
        metric_note=(
            "词汇支持/期望片段覆盖的启发式代理指标，不是语义蕴含或医学准确率；标签为助手草稿，须人工复核。"
            "引用及忠实度均值仅在预期可回答题计算，拒答题不贡献引用满分。拒答准确率指全体题目的回答/拒答决策正确率。"
            "无依据结论数量为启发式估计；否定、数值、人群和因果仍须人工审核。"
        ),
    )


def load_answer_evaluation_cases(path: Path | None = None) -> list[AnswerEvaluationCase]:
    path = path or Path(__file__).resolve().parents[1] / "data" / "benchmark" / "answer_eval_v1.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"回答评测集读取失败：{path}") from exc
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise RuntimeError("回答评测集必须包含 cases 数组。")
    return [AnswerEvaluationCase.model_validate(item) for item in raw_cases]
