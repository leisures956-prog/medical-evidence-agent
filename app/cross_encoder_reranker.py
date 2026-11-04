"""Lazy, offline Cross-Encoder reranking for retrieved medical documents."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from app.medical_models import RetrievalHit
from app.quality_reranker import quality_flags


class CrossEncoderRerankerError(RuntimeError):
    """Raised when the optional local Cross-Encoder cannot be used."""


DEFAULT_CROSS_ENCODER_QUALITY_PENALTIES: dict[str, float] = {
    "abstract_missing": 2.0,
    "editorial": 3.0,
    "correction": 4.0,
    "review": 0.5,
}


class CrossEncoderReranker:
    """Rerank hybrid candidates with query-document relevance logits.

    The model is deliberately lazy-loaded and restricted to local files.  This
    keeps ``/health`` and ordinary retrieval independent from model startup and
    prevents an API request from silently downloading weights.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model: Any | None = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name or os.getenv("CROSS_ENCODER_MODEL")
        self._model = model
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0。")

    def load_model(self):
        """Load the configured model once and never access the network."""

        if self._model is not None:
            return self._model
        if not self.model_name:
            raise CrossEncoderRerankerError(
                "未配置 CROSS_ENCODER_MODEL；请指向本地 Cross-Encoder 目录。"
            )
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                local_files_only=True,
            )
        except Exception as exc:
            raise CrossEncoderRerankerError(
                f"Cross-Encoder 加载失败：{self.model_name}。"
            ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        hits: Iterable[RetrievalHit],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip():
            raise ValueError("query 不能为空。")
        if top_k is not None and top_k < 1:
            return []

        ranked = self._score(
            query,
            hits,
            "hybrid_persisted_cross_reranked",
        )
        ranked.sort(
            key=lambda item: (
                -item[0].score,
                -item[1],
                item[0].document.id,
            )
        )
        output = [item[0] for item in ranked]
        return output[:top_k] if top_k is not None else output

    def rerank_with_quality(
        self,
        query: str,
        hits: Iterable[RetrievalHit],
        *,
        top_k: int | None = None,
        penalties: dict[str, float] | None = None,
    ) -> list[RetrievalHit]:
        """Combine relevance logits with explicit title/abstract penalties."""

        if top_k is not None and top_k < 1:
            return []
        values = dict(DEFAULT_CROSS_ENCODER_QUALITY_PENALTIES)
        if penalties:
            values.update(penalties)
        if any(value < 0 for value in values.values()):
            raise ValueError("质量惩罚不能为负数。")

        scored = self._score(
            query,
            hits,
            "hybrid_persisted_cross_quality_reranked",
        )
        adjusted: list[tuple[RetrievalHit, float]] = []
        for hit, base_score in scored:
            penalty = sum(
                values.get(flag, 0.0) for flag in quality_flags(hit.document)
            )
            adjusted.append(
                (hit.model_copy(update={"score": hit.score - penalty}), base_score)
            )
        adjusted.sort(
            key=lambda item: (
                -item[0].score,
                -item[1],
                item[0].document.id,
            )
        )
        output = [item[0] for item in adjusted]
        return output[:top_k] if top_k is not None else output

    def _score(
        self,
        query: str,
        hits: Iterable[RetrievalHit],
        retrieval_method: str,
    ) -> list[tuple[RetrievalHit, float]]:
        if not query.strip():
            raise ValueError("query 不能为空。")
        candidates = list(hits)
        if not candidates:
            return []

        pairs = [(query, hit.document.retrieval_text()) for hit in candidates]
        try:
            scores = self.load_model().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except CrossEncoderRerankerError:
            raise
        except Exception as exc:
            raise CrossEncoderRerankerError("Cross-Encoder 重排失败。") from exc

        flattened = [float(score) for score in scores]
        if len(flattened) != len(candidates):
            raise CrossEncoderRerankerError(
                "Cross-Encoder 返回的分数数量与候选文献数量不一致。"
            )
        return [
            (
                hit.model_copy(
                    update={
                        "score": score,
                        "retrieval_method": retrieval_method,
                    }
                ),
                float(hit.score),
            )
            for hit, score in zip(candidates, flattened, strict=True)
        ]


__all__ = [
    "CrossEncoderReranker",
    "CrossEncoderRerankerError",
    "DEFAULT_CROSS_ENCODER_QUALITY_PENALTIES",
]
