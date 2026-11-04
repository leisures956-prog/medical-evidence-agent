"""State and request/result contracts for the Stage 7 agent graph.

Stage 7A deliberately contains no graph execution and no provider calls.  The
models here are the stable boundary that later nodes will read and update.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.medical_models import CitationBinding, RetrievalHit, StructuredEvidence


AgentStatus = Literal["pending", "running", "completed", "abstained", "failed"]
TraceStatus = Literal["started", "completed", "skipped", "failed"]


class AgentRequest(BaseModel):
    """Stable input accepted by a future agent entry point."""

    question: str = Field(min_length=1)
    retrieval_method: str = Field(default="hybrid", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_limit: int = Field(default=30, ge=1, le=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTraceEvent(BaseModel):
    """One observable state transition in the agent workflow."""

    step: str = Field(min_length=1)
    status: TraceStatus
    duration_ms: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Mutable graph state shared by retrieval, answer and check nodes."""

    request: AgentRequest
    status: AgentStatus = "pending"
    current_step: str | None = None

    hits: list[RetrievalHit] = Field(default_factory=list)
    evidence_records: list[StructuredEvidence] = Field(default_factory=list)
    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    citation_bindings: list[CitationBinding] = Field(default_factory=list)

    abstained: bool = False
    abstention_reason: str | None = None
    errors: list[str] = Field(default_factory=list)

    attempt_count: int = Field(default=0, ge=0)
    query_rewrite_count: int = Field(default=0, ge=0)
    answer_repair_count: int = Field(default=0, ge=0)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_request(cls, request: AgentRequest) -> "AgentState":
        """Create a clean state without performing retrieval or model calls."""

        return cls(request=request, status="pending", current_step="query_analysis")

    def to_result(self) -> "AgentResult":
        """Project internal state to the externally returnable result contract."""

        return AgentResult(
            status=self.status,
            answer=self.answer,
            citations=list(self.citations),
            citation_bindings=list(self.citation_bindings),
            abstained=self.abstained,
            abstention_reason=self.abstention_reason,
            errors=list(self.errors),
            attempt_count=self.attempt_count,
            trace=list(self.trace),
            metadata=dict(self.metadata),
        )


class AgentResult(BaseModel):
    """Stable final response shape for a future `/agent-answer` endpoint."""

    status: AgentStatus
    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    citation_bindings: list[CitationBinding] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    errors: list[str] = Field(default_factory=list)
    attempt_count: int = Field(default=0, ge=0)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
