"""Deterministic medical query expansion for reproducible retrieval tests."""

from __future__ import annotations

import re
from dataclasses import dataclass


MEDICAL_QUERY_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("non-small cell lung cancer", ("NSCLC",)),
    ("nsclc", ("non-small cell lung cancer",)),
    ("lung cancer", ("non-small cell lung cancer", "NSCLC")),
    ("immunotherapy", ("immune checkpoint inhibitors", "PD-1 PD-L1 treatment")),
    ("immune checkpoint inhibitor", ("immunotherapy", "PD-1 PD-L1 inhibitor")),
    ("glp-1", ("glucagon-like peptide-1 receptor agonist",)),
    ("sglt2", ("sodium-glucose cotransporter 2 inhibitor",)),
    ("alzheimer disease", ("Alzheimer's disease",)),
    ("biomarkers", ("diagnostic markers", "prognostic markers")),
    ("mechanical thrombectomy", ("endovascular thrombectomy",)),
)


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _replace_phrase(query: str, phrase: str, replacement: str) -> str:
    return re.sub(
        re.escape(phrase),
        replacement,
        query,
        count=1,
        flags=re.IGNORECASE,
    )


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    queries: tuple[str, ...]


class MedicalQueryRewriter:
    """Generate a small stable set of medically equivalent query variants."""

    def __init__(self, max_queries: int = 4):
        if max_queries < 1:
            raise ValueError("max_queries 必须大于 0。")
        self.max_queries = max_queries

    def rewrite(self, query: str) -> QueryRewriteResult:
        original = _normalize_spaces(query)
        if not original:
            raise ValueError("query 不能为空。")

        candidates: list[str] = [original]
        matched: list[tuple[str, tuple[str, ...]]] = []
        lowered = original.lower()
        for phrase, replacements in MEDICAL_QUERY_EXPANSIONS:
            if phrase.lower() in lowered:
                matched.append((phrase, replacements))
                for replacement in replacements:
                    candidates.append(_replace_phrase(original, phrase, replacement))

        if len(matched) >= 2:
            combined = original
            for phrase, replacements in matched:
                combined = _replace_phrase(combined, phrase, replacements[0])
            candidates.append(combined)

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = _normalize_spaces(candidate)
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                unique.append(normalized)
            if len(unique) >= self.max_queries:
                break

        return QueryRewriteResult(original_query=original, queries=tuple(unique))


__all__ = [
    "MEDICAL_QUERY_EXPANSIONS",
    "MedicalQueryRewriter",
    "QueryRewriteResult",
]
