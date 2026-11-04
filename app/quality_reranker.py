"""可解释的文献质量感知重排。

这里的标记只来自标题和摘要是否存在，不判断医学结论真伪，也不改变原有
``hybrid`` 排序。它作为一个可选的后处理层，专门用于比较“重排前后”的
文献质量与检索指标。
"""

from __future__ import annotations

import re
from typing import Iterable

from app.medical_models import MedicalDocument, RetrievalHit


_CORRECTION_PATTERN = re.compile(
    r"\b(?:correction|erratum|corrigendum|retraction|retracted|expression of concern)\b",
    flags=re.IGNORECASE,
)
_EDITORIAL_PATTERN = re.compile(
    r"\b(?:editorial|commentary|perspective|letter to the editor)\b",
    flags=re.IGNORECASE,
)
_REVIEW_PATTERN = re.compile(
    r"\b(?:review|systematic review|narrative review|scoping review|meta-analysis|meta analysis)\b",
    flags=re.IGNORECASE,
)


DEFAULT_QUALITY_PENALTIES: dict[str, float] = {
    "abstract_missing": 0.12,
    "editorial": 0.08,
    "correction": 0.18,
    "review": 0.04,
}


def quality_flags(document: MedicalDocument) -> tuple[str, ...]:
    """Return transparent title/abstract quality flags in stable order."""

    title = document.title or ""
    flags: list[str] = []
    if not document.abstract.strip():
        flags.append("abstract_missing")
    if _CORRECTION_PATTERN.search(title):
        flags.append("correction")
    if _EDITORIAL_PATTERN.search(title):
        flags.append("editorial")
    if _REVIEW_PATTERN.search(title):
        flags.append("review")
    return tuple(flags)


class QualityReranker:
    """Apply small, configurable quality penalties after retrieval."""

    def __init__(self, penalties: dict[str, float] | None = None):
        values = dict(DEFAULT_QUALITY_PENALTIES)
        if penalties:
            values.update(penalties)
        if any(value < 0 for value in values.values()):
            raise ValueError("质量惩罚不能为负数。")
        self.penalties = values

    def explain(self, document: MedicalDocument) -> dict[str, object]:
        flags = quality_flags(document)
        penalty = sum(self.penalties.get(flag, 0.0) for flag in flags)
        return {
            "flags": list(flags),
            "quality_penalty": round(penalty, 8),
        }

    def rerank(
        self,
        hits: Iterable[RetrievalHit],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if top_k is not None and top_k < 1:
            return []

        scored: list[tuple[RetrievalHit, float, int]] = []
        for position, hit in enumerate(hits):
            explanation = self.explain(hit.document)
            penalty = float(explanation["quality_penalty"])
            final_score = max(0.0, float(hit.score) - penalty)
            scored.append(
                (
                    hit.model_copy(update={"score": final_score}),
                    float(hit.score),
                    position,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0].score,
                -item[1],
                item[0].document.id,
                item[2],
            )
        )
        results = [item[0] for item in scored]
        return results[:top_k] if top_k is not None else results


__all__ = ["DEFAULT_QUALITY_PENALTIES", "QualityReranker", "quality_flags"]
