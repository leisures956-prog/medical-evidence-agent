from __future__ import annotations

import math
import re

from app.medical_models import StructuredEvidence
from app.text_normalization import normalize_for_prompt


_PLAIN_CITATION = re.compile(r"\[(?P<number>\d+)\]")
_ANY_CITATION = re.compile(r"\[\d+(?:(?:\:|\#)[A-Za-z0-9_.-]+)?\]")
_SENTENCE = re.compile(r".*?(?:[。！？!?]|\.(?!\d))", re.DOTALL)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,}|\d+(?:\.\d+)?")
_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "were", "was", "are", "for",
    "into", "than", "study", "abstract", "reported", "result", "results", "evidence",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(normalize_for_prompt(text))
        if token.lower() not in _STOPWORDS and len(token) > 1
    }


def _claim_context(answer: str, start: int, end: int) -> str:
    left = max(answer.rfind(mark, 0, start) for mark in ("。", "！", "？", "!", "?", "\n"))
    left = 0 if left < 0 else left + 1
    right_candidates = [answer.find(mark, end) for mark in ("。", "！", "？", "!", "?", "\n")]
    right_candidates = [value for value in right_candidates if value >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(answer)
    return answer[left:right]


def _best_span(claim: str, record: StructuredEvidence) -> str | None:
    candidates = [span for span in record.evidence_spans if span.source_section == "abstract"]
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return None

    scored: list[tuple[float, str]] = []
    for span in candidates:
        span_tokens = _tokens(span.exact_text)
        overlap = len(claim_tokens & span_tokens)
        score = overlap / math.sqrt(max(1, len(claim_tokens) * len(span_tokens)))
        scored.append((score, span.evidence_id))
    scored.sort(reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    if len(scored) > 1:
        best_score, _ = scored[0]
        second_score, _ = scored[1]
        if best_score - second_score < 0.05 or best_score < second_score * 1.25:
            return None
    return scored[0][1]


def _best_global_span(
    claim: str,
    records: list[StructuredEvidence],
) -> tuple[int, str] | None:
    resolved = _best_global_spans(claim, records)
    if resolved is None:
        return None
    number, evidence_ids = resolved
    return number, evidence_ids[0]


def _best_global_spans(
    claim: str,
    records: list[StructuredEvidence],
) -> tuple[int, list[str]] | None:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return None
    by_record: dict[int, list[tuple[float, str, set[str]]]] = {}
    for number, record in enumerate(records, start=1):
        for span in record.evidence_spans:
            if span.source_section != "abstract":
                continue
            span_tokens = _tokens(span.exact_text)
            overlap = len(claim_tokens & span_tokens)
            score = overlap / math.sqrt(max(1, len(claim_tokens) * len(span_tokens)))
            by_record.setdefault(number, []).append((score, span.evidence_id, span_tokens))
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    if claim_numbers:
        for number, spans in list(by_record.items()):
            numeric_spans = [
                item for item in spans if claim_numbers.issubset(item[2])
            ]
            if numeric_spans:
                by_record[number] = numeric_spans
    record_scores = sorted(
        [
            (max(item[0] for item in spans), number)
            for number, spans in by_record.items()
        ],
        reverse=True,
    )
    if not record_scores or record_scores[0][0] <= 0:
        return None
    if len(record_scores) > 1:
        best_score = record_scores[0][0]
        second_score = record_scores[1][0]
        if best_score - second_score < 0.05 or best_score < second_score * 1.25:
            return None

    number = record_scores[0][1]
    scored = sorted(by_record[number], reverse=True)
    best_score, best_id, best_tokens = scored[0]
    if len(scored) > 1:
        second_score, _, second_tokens = scored[1]
        new_terms = (second_tokens - best_tokens) & claim_tokens
        if best_score - second_score < 0.05 or (
            not new_terms and best_score < second_score * 1.25
        ):
            return None
    selected = [best_id]
    if len(scored) > 1:
        for score, evidence_id, span_tokens in scored[1:]:
            if score <= 0 or score < best_score * 0.75:
                continue
            if (span_tokens - best_tokens) & claim_tokens:
                selected.append(evidence_id)
            if len(selected) == 3:
                break
    return number, selected


def resolve_plain_citations(
    answer: str,
    records: list[StructuredEvidence],
) -> tuple[str, list[str]]:
    """Resolve source-only citations only when one abstract span is uniquely supported."""

    replacements: list[tuple[int, int, str]] = []
    notes: list[str] = []
    for match in _PLAIN_CITATION.finditer(answer):
        number = int(match.group("number"))
        if number < 1 or number > len(records):
            continue
        claim = _claim_context(answer, match.start(), match.end())
        evidence_id = _best_span(claim, records[number - 1])
        if evidence_id is None:
            continue
        explicit = f"[{number}:{evidence_id}]"
        replacements.append((match.start(), match.end(), explicit))
        notes.append(f"来源级引用 [{number}] 已唯一解析为 {explicit}。")

    resolved = answer
    for start, end, replacement in reversed(replacements):
        resolved = resolved[:start] + replacement + resolved[end:]
    return resolved, notes


def attach_unique_evidence_to_uncited_claims(
    answer: str,
    records: list[StructuredEvidence],
) -> tuple[str, list[str]]:
    """Attach a citation only when an uncited sentence has one unique lexical match."""

    insertions: list[tuple[int, str]] = []
    notes: list[str] = []
    segments: list[tuple[str, int]] = [
        (match.group(), match.end())
        for match in _SENTENCE.finditer(answer)
    ]
    last_end = segments[-1][1] if segments else 0
    if last_end < len(answer) and answer[last_end:].strip():
        segments.append((answer[last_end:], len(answer)))

    for raw_sentence, segment_end in segments:
        sentence = raw_sentence.strip()
        if not sentence or _ANY_CITATION.search(sentence):
            continue
        lowered = sentence.lower()
        if "supported by the following" in lowered or "支持证据如下" in sentence:
            continue
        resolved = _best_global_spans(sentence, records)
        if resolved is None:
            continue
        number, evidence_ids = resolved
        citation = " " + "".join(f"[{number}:{evidence_id}]" for evidence_id in evidence_ids)
        stripped_raw = raw_sentence.rstrip()
        has_terminal_punctuation = bool(
            stripped_raw and stripped_raw[-1] in "。！？!?"
        ) or bool(stripped_raw.endswith(".") and not re.search(r"\.\d+$", stripped_raw))
        punctuation_index = segment_end - 1 if has_terminal_punctuation else segment_end
        insertions.append((punctuation_index, citation))
        notes.append(f"无引用事实句已唯一匹配并附加 {citation.strip()}。")

    grounded = answer
    for index, citation in reversed(insertions):
        grounded = grounded[:index] + citation + grounded[index:]
    return grounded, notes
