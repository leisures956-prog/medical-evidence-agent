from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


EvidenceSection = Literal["title", "abstract"]
EvidenceStatus = Literal[
    "reported",
    "not_reported_in_abstract",
    "abstract_unavailable",
    "extraction_failed",
]


class EvidenceSpan(BaseModel):
    """A verbatim, traceable fragment from a PubMed title or abstract."""

    evidence_id: str = Field(min_length=1)
    pmid: str = Field(min_length=1)
    source_section: EvidenceSection
    exact_text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("start 和 end 必须同时提供，或同时省略。")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end 必须大于 start。")
        return self


class EvidenceValue(BaseModel):
    """One structured field and the evidence IDs that support it."""

    value: str | list[str] | None = None
    status: EvidenceStatus
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_and_evidence(self):
        if self.value is None and self.status == "reported":
            raise ValueError("reported 状态必须提供 value。")
        if self.value is not None and self.status != "reported":
            raise ValueError("只有 reported 状态可以提供 value。")
        if self.value is not None and not self.evidence_ids:
            raise ValueError("有 value 的字段必须绑定 evidence_ids。")
        return self


class StructuredEvidence(BaseModel):
    """Summary-level evidence extracted only from a PubMed title/abstract."""

    pmid: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    title_evidence_ids: list[str] = Field(default_factory=list)
    research_object: EvidenceValue
    disease: EvidenceValue
    method: EvidenceValue
    dataset: EvidenceValue
    main_findings: EvidenceValue
    limitations: EvidenceValue
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_references(self):
        span_ids = [span.evidence_id for span in self.evidence_spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("evidence_id 必须唯一。")

        known_ids = set(span_ids)
        references = list(self.title_evidence_ids)
        for field_name in (
            "research_object",
            "disease",
            "method",
            "dataset",
            "main_findings",
            "limitations",
        ):
            references.extend(getattr(self, field_name).evidence_ids)

        missing = sorted(set(references) - known_ids)
        if missing:
            raise ValueError(f"引用了不存在的 evidence_id：{missing}")
        return self


class EvidenceResponse(BaseModel):
    question: str
    retrieval_method: Literal["pubmed_hybrid"] = "pubmed_hybrid"
    extraction_mode: Literal["metadata_only", "strict_llm"]
    records: list[StructuredEvidence] = Field(default_factory=list)
