from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MedicalDocument(BaseModel):
    """Normalized medical evidence record used by every retrieval stage."""

    id: str
    source: str = "unknown"
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    url: str | None = None
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def retrieval_text(self) -> str:
        parts = [self.title, self.abstract, " ".join(self.keywords)]
        return "\n".join(part for part in parts if part).strip()


class RetrievalHit(BaseModel):
    document: MedicalDocument
    score: float
    retrieval_method: str
    semantic_score: float | None = None
    keyword_score: float | None = None


class GroundedAnswer(BaseModel):
    answer: str
    sources: list[MedicalDocument]
    citations: list[int] = Field(default_factory=list)
